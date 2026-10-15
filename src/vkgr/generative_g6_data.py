"""G6-PREP G data path: HIST_V2_OBS_L50 history overlay plus a SIDMAP_V2D codec of any registered depth.

Same logic as ``generative_tok_data.build_tok_context`` (examples and queries keep their v1 identity, only the history fields are
replaced), but the codec comes from a verified SIDMAP_V2D bundle and the binding records the bundle identity and its depth.
"""
from . import generative_experiment as ge
from . import history_v2
from .generative_data_v2 import ListExamples
from .sid_depth import ValidatedCodecD

HISTORY_LENGTH = 50


def build_g6_context(context, loaded, length, sid):
    """RunContext with v2 histories and the codec of ``sid`` (a ``sid_v2.LoadedSID`` produced by ``sid_v2d.load_bundle_d``)."""
    if length != HISTORY_LENGTH or loaded.get('length') != length:
        raise ValueError('registered history length required')
    info = sid.info
    if type(info) is not dict or not isinstance(sid.codec, ValidatedCodecD) or type(sid.manifest) is not dict or info.get('depth') != sid.codec.depth:
        raise ValueError('verified SIDMAP_V2D bundle required')
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
               'g6': {'sid_version': info['sid_version'], 'sid_manifest_sha256': info['manifest_sha256'],
                      'content_feature_sha256': sid.manifest['content_feature_sha256'], 'kind': info['kind'], 'branching': info['branching'],
                      'depth': info['depth'], 'suffix_vocabulary': info['suffix_vocabulary'], 'vocab_size': info['vocab_size'],
                      'engineering': info['engineering']}}
    return ge.RunContext(ge._CONTEXT_KEY, examples=ListExamples(rows), codec=codec, queries=queries, binding=binding,
                         stats=context.slice_definition).check()
