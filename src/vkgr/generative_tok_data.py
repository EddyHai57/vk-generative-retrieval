"""GTOK-01 G data path: HIST_V2_OBS_L50 history overlay plus a SIDMAP_V2 codec.

``build_tok_context`` has the same logic as ``generative_data_v2.build_v2_context`` (examples and queries keep their
v1 identity, only the history fields are replaced), but it takes the codec from a verified SIDMAP_V2 bundle instead of
the two registered P5 labels, and it records the bundle identity in the context binding. ``build_v2_context`` itself
cannot be reused (its SID_VERSIONS whitelist and the ``codec is base_codec`` assertion), and it must stay byte-identical
because its bytes feed the binding of every existing anchor run.
"""
from . import generative_experiment as ge
from . import history_v2
from .generative_data_v2 import ListExamples

HISTORY_LENGTH = 50


def build_tok_context(context, loaded, length, sid):
    """RunContext with v2 histories and the codec of ``sid`` (a ``sid_v2.LoadedSID``)."""
    if length != HISTORY_LENGTH or loaded.get('length') != length:
        raise ValueError('registered history length required')
    info = sid.info
    if type(info) is not dict or sid.codec is None or type(sid.manifest) is not dict:
        raise ValueError('verified SIDMAP_V2 bundle required')
    context.check()
    catalog = set(context.catalog)
    originals = context.examples.rows(list(range(len(context.examples))))
    rows = history_v2.overlay_examples(originals, loaded['train'], length, catalog)
    queries = history_v2.overlay_queries(context.queries, loaded['validation'], length, catalog)
    base_codec, codec = context.codec, sid.codec
    sids_sha = lambda c: ge._digest(c.catalog.sids.tolist())
    binding = {**context.binding, 'queries_sha256': ge._digest(queries), 'codec_sha256': sids_sha(codec),
               'b1': {'history_version': f'HIST_V2_OBS_L{length}', 'history_manifest_sha256': loaded['manifest_sha256'],
                      'train_history_sha256': ge._digest(rows), 'base_queries_sha256': context.binding['queries_sha256'],
                      'sid_version': info['sid_version'], 'base_sid_sha256': sids_sha(base_codec), 'sid_sha256': sids_sha(codec)},
               'tok': {'sid_version': info['sid_version'], 'sid_manifest_sha256': info['manifest_sha256'],
                       'content_feature_sha256': sid.manifest['content_feature_sha256'], 'kind': info['kind'], 'branching': info['branching'],
                       'suffix_vocabulary': info['suffix_vocabulary'], 'vocab_size': info['vocab_size'], 'engineering': info['engineering']}}
    return ge.RunContext(ge._CONTEXT_KEY, examples=ListExamples(rows), codec=codec, queries=queries, binding=binding,
                         stats=context.slice_definition).check()
