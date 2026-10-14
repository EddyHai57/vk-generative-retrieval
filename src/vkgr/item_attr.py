"""FEAT-02 ITEM_ATTR_V1: item -> author table and the train-defined author vocabulary for the F6 variant.

Everything here is a function of the frozen catalog, the train-defined ``support`` set and ``items_metadata.parquet``
(columns ``item_id`` / ``author_id`` only). There is no query, holdout, target or test input anywhere: the vocabulary is the
sorted set of authors of SUPPORTED catalog items, so a new author on an unsupported item never changes it.
Author rows: 0 = PAD, 1 = out of vocabulary (OOV), 2 + rank = vocabulary author.
The sidecar is bound by sha256 to the metadata file, the catalog and the support set; its vocabulary size and sha256
must equal the hard-coded audit constants (the caller passes them in; they are never defaulted here).
"""
import hashlib
import json
from pathlib import Path

from .history_v2 import _read_pinned, _write_json

ATTR_VERSION = 'ITEM_ATTR_V1'
MAX_AUTHOR = (1 << 32) - 1                      # author_id is uint32


def _author(value):
    if type(value) is not int or not 0 <= value <= MAX_AUTHOR:
        raise ValueError('author id must be an integer in uint32 range')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def vocab_digest(vocab):
    """sha256 of the compact ascending JSON list of vocabulary authors (the form hard-coded as AUTHOR_VOCAB_SHA256)."""
    return hashlib.sha256(json.dumps(list(vocab), separators=(',', ':')).encode()).hexdigest()


def load_authors(path, catalog):
    """{item_id: author_id} for every catalog item, read from the two metadata columns only.

    Rows are filtered by the catalog before they are turned into python objects (the file has millions of rows);
    rows of items outside the catalog are ignored. Non-integer columns, null values, duplicates and missing items fail.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    catalog = list(catalog)
    if len(set(catalog)) != len(catalog) or any(type(i) is not int for i in catalog):
        raise ValueError('catalog must be unique integers')
    table = pq.read_table(path, columns=['item_id', 'author_id'])
    for name in ('item_id', 'author_id'):
        if not pa.types.is_integer(table.schema.field(name).type):
            raise ValueError(f'{name} must be an integer column')
    wanted = pa.array(catalog, type=table.schema.field('item_id').type)
    kept = table.filter(pc.is_in(table.column('item_id'), value_set=wanted))
    authors = {}
    for item, author in zip(kept.column('item_id').to_pylist(), kept.column('author_id').to_pylist()):
        if item in authors:
            raise ValueError('duplicate item metadata')
        authors[item] = _author(author)
    if set(authors) != set(catalog):
        raise ValueError('catalog author coverage')
    return authors


def author_vocab(authors, support):
    """Sorted authors of the SUPPORTED catalog items (``authors`` maps item -> author)."""
    if any(item not in authors for item in support):
        raise ValueError('supported item without author')
    return sorted({_author(authors[item]) for item in support})


def author_rows(authors, vocab):
    """Row per author: ``None`` (PAD slot) -> 0, author outside the vocabulary -> 1, vocabulary author -> 2 + rank."""
    vocab = list(vocab)
    for author in vocab:
        _author(author)
    if any(b <= a for a, b in zip(vocab, vocab[1:])):
        raise ValueError('vocabulary must be strictly ascending')
    rank = {author: k for k, author in enumerate(vocab)}
    return [0 if author is None else 2 + rank[author] if _author(author) in rank else 1 for author in authors]


def build_item_attr(authors, catalog, support):
    """Authors in catalog order plus the train-defined vocabulary and its digest."""
    catalog = list(catalog)
    if len(set(catalog)) != len(catalog) or any(type(i) is not int for i in catalog):
        raise ValueError('catalog must be unique integers')
    if not set(support) <= set(catalog):
        raise ValueError('support outside the catalog')
    if any(item not in authors for item in catalog):
        raise ValueError('author missing for a catalog item')
    values = [_author(authors[item]) for item in catalog]
    vocab = author_vocab({item: authors[item] for item in catalog}, support)
    return {'authors': values, 'vocab': vocab, 'vocab_size': len(vocab), 'vocab_sha256': vocab_digest(vocab)}


def _check_pins(size, digest, expected_size, expected_sha256):
    if size != expected_size or digest != expected_sha256:
        raise ValueError('author vocabulary differs from the hard-coded audit constants')


def write_item_attr(output, items_metadata_sha256, catalog, support, authors, binding, expected_size, expected_sha256):
    built = build_item_attr(authors, catalog, support)
    _check_pins(built['vocab_size'], built['vocab_sha256'], expected_size, expected_sha256)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    artifacts = {'authors.json': _write_json(output / 'authors.json', built['authors'])}
    manifest = {'schema_version': 1, 'kind': 'item_attr_prepare', 'attr_version': ATTR_VERSION, 'fields': ['author_id'],
                'items_metadata_sha256': items_metadata_sha256, 'catalog_sha256': _digest(list(catalog)), 'support_sha256': _digest(sorted(support)),
                'catalog_items': len(list(catalog)), 'supported_items': len(set(support)),
                'vocab_size': built['vocab_size'], 'vocab_sha256': built['vocab_sha256'],
                'definition': 'author_id of every catalog item (catalog order); vocabulary = sorted authors of supported catalog items; rows 0 PAD, 1 OOV, 2 + rank',
                'binding': binding, 'artifacts': artifacts, 'test_gate': 'closed_hash_only'}
    return {'manifest': manifest, 'manifest_sha256': _write_json(output / 'manifest.json', manifest)}


def load_item_attr(path, digest, items_metadata_sha256, catalog, support, expected_size, expected_sha256):
    """Load and re-derive the sidecar for this catalog/support; any identity, binding or pin difference raises ValueError."""
    path = Path(path); value = _read_pinned(path, digest); catalog = list(catalog)
    if value.get('kind') != 'item_attr_prepare' or value.get('attr_version') != ATTR_VERSION or value.get('fields') != ['author_id'] \
            or value.get('test_gate') != 'closed_hash_only':
        raise ValueError('item attr manifest identity')
    if value.get('items_metadata_sha256') != items_metadata_sha256:
        raise ValueError('attr not bound to this items metadata')
    if value.get('catalog_sha256') != _digest(catalog) or value.get('catalog_items') != len(catalog):
        raise ValueError('attr not bound to this catalog')
    if not set(support) <= set(catalog) or value.get('support_sha256') != _digest(sorted(support)):
        raise ValueError('attr not bound to this support set')
    authors = _read_pinned(path.parent / 'authors.json', value['artifacts']['authors.json'])
    if type(authors) is not list or len(authors) != len(catalog):
        raise ValueError('attr/catalog length mismatch')
    vocab = author_vocab(dict(zip(catalog, authors)), support)
    if value.get('vocab_size') != len(vocab) or value.get('vocab_sha256') != vocab_digest(vocab) or any(_author(a) != a for a in authors):
        raise ValueError('stored authors do not reproduce the stored vocabulary')
    _check_pins(value['vocab_size'], value['vocab_sha256'], expected_size, expected_sha256)
    return {'manifest': value, 'manifest_sha256': digest, 'authors': authors, 'vocab': vocab, 'vocab_size': len(vocab), 'vocab_sha256': value['vocab_sha256']}
