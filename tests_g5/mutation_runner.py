"""Generic isolated mutation runner. Each mutant is applied to a fresh copy of the repository subset and run with a unique bytecode cache.

Usage: python tests_g5/mutation_runner.py <suite>   (suite = qmeta | enqueue)
Output: markdown ledger on stdout. Exit 1 on any unexpected outcome.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = {
    'qmeta': {'target': 'src/vkgr_g5/query_meta_p.py', 'copy': ['src', 'tests_g5'], 'cmd': ['-m', 'unittest', 'tests_g5.test_query_meta_p'], 'mutants': [
        ('dup-user-gate-off', "    if len({r['user_id'] for r in rows}) != len(rows):", "    if False:", 'kill'),
        ('history-gate-off', "    if any(not 1 <= len(r['history']) <= history_len for r in rows):", "    if False:", 'kill'),
        ('support-flip', "'supported' if q['target'] in support else 'unsupported'", "'unsupported' if q['target'] in support else 'supported'", 'kill'),
        ('oov-label', "'oov' if q['oov'] else (", "'oov' if not q['oov'] else (", 'kill'),
        ('query-digest-over-rows', "'queries_digest': b1._digest(queries)", "'queries_digest': b1._digest(rows)", 'kill'),
        ('catalog-digest-wrong', "'catalog_digest': b1._digest(list(data['catalog']))", "'catalog_digest': b1._digest(sorted(data['catalog'])[::-1])", 'kill'),
        ('support-digest-unsorted', "b1._digest(sorted(data['support']))", "b1._digest(list(data['support']))", 'kill'),
        ('manifest-queries-skipped', "binding['queries_sha256'] != record['queries_digest'] or ", "False or ", 'kill'),
        ('manifest-catalog-skipped', "or binding['catalog_sha256'] != record['catalog_digest'] or ", "or False or ", 'kill'),
        ('manifest-support-skipped', "or binding['support_sha256'] != support_sha256:", "or False:", 'kill'),
        ('manifest-required-off', "    if not manifests:", "    if False:", 'kill'),
        ('manifest-pair-length-off', "        if len(args.manifest) != len(args.manifest_sha):", "        if False:", 'kill'),
        ('only-first-manifest', "for value in manifests:", "for value in manifests[:1]:", 'kill'),
        ('query-count-lt', "if record['query_count'] != args.expect_queries or", "if record['query_count'] < args.expect_queries or", 'kill'),
        ('oov-count-lt', "!= args.expect_oov:", "< args.expect_oov:", 'kill'),
        ('manifest-arg-optional', "parser.add_argument('--manifest', action='append', required=True)", "parser.add_argument('--manifest', action='append', required=False)", 'equivalent'),
        ('count-check-off', "        if record['query_count'] != args.expect_queries or sum(1 for r in record['rows'] if r['oov']) != args.expect_oov:", "        if False:", 'kill'),
        ('oov-count-ignored', "or sum(1 for r in record['rows'] if r['oov']) != args.expect_oov:", ":", 'syntax'),
        ('mkdir-in-check-only', "        if args.check_only:\n", "        out.parent.mkdir(parents=True, exist_ok=True)\n        if args.check_only:\n", 'kill'),
        ('mkdir-exist-ok', "out.parent.mkdir(parents=True, exist_ok=False)", "out.parent.mkdir(parents=True, exist_ok=True)", 'kill'),
        ('check-only-writes', "        if args.check_only:\n", "        if False:\n", 'kill'),
        ('line-root-ancestor-ok', "        if Path(vkgr.__file__).resolve().parent != root / 'src' / 'vkgr':", "        if root not in Path(vkgr.__file__).resolve().parents:", 'kill'),
        ('output-in-deployment-ok', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if out.exists() or out.with_name(out.name + '.partial').exists():", 'kill'),
        ('existing-output-ok', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if root in out.parents or out.with_name(out.name + '.partial').exists():", 'kill'),
        ('partial-check-off', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if root in out.parents or out.exists():", 'kill'),
        ('resource-guard-on-output-parent', "b1.resource_guard(nearest_existing(out.parent), spec['disk_cap_bytes'])", "b1.resource_guard(out.parent, spec['disk_cap_bytes'])", 'kill'),
        ('subprocess-error-uncaught', "RuntimeError, subprocess.SubprocessError)", "RuntimeError)", 'kill'),
    ]},
    'enqueue': {'target': 'ops/g5/g5_fusion_enqueue_20261018.py', 'copy': ['ops'], 'cmd': ['-m', 'unittest', 'discover', '-s', 'ops/g5', '-p', 'test_g5_fusion_enqueue*'], 'mutants': [
        ('claimed-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'rejected')", 'kill'),
        ('rejected-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'claimed')", 'kill'),
        ('runs-dir-not-checked', "    if (root / 'runs' / rid).exists():\n        return True\n", "", 'kill'),
        ('reason-form-dropped', "(json|reason|json\\.reason)", "(json)", 'kill'),
        ('json-reason-form-dropped', "(json|reason|json\\.reason)", "(json|reason)", 'kill'),
        ('tmp-form-dropped', "(\\.tmp)?'", "'", 'kill'),
        ('config-from-template', "config=str(Path(p_deploy) / get('--config')),", "config=t['config'],", 'kill'),
        ('gen-limit-80', "if gen and (len(gen) > 8 or not gen.isalnum()):", "if gen and (len(gen) > 80 or not gen.isalnum()):", 'kill'),
        ('exact-match-off', "    return any(pattern.fullmatch(p.name) for sub", "    return any(rid in p.name for sub", 'kill'),
        ('inputs-positional', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=[get('--inputs')] + [i for i in t['inputs'][1:3]] + manifests,", 'kill'),
        ('manifest-kept-in-inputs', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=list(t['inputs']) + manifests,", 'kill'),
        ('manifests-dropped-from-inputs', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=[i for i in t['inputs'] if i != get('--manifest')],", 'kill'),
        ('manifest-sha-wrong', "'--manifest-sha', P_MANIFESTS[seed]]", "'--manifest-sha', P_MANIFESTS[42]]", 'kill'),
        ('expect-queries-wrong', "EXPECT_QUERIES, EXPECT_OOV = 2520, 623", "EXPECT_QUERIES, EXPECT_OOV = 2519, 623", 'kill'),
        ('expect-oov-wrong', "EXPECT_QUERIES, EXPECT_OOV = 2520, 623", "EXPECT_QUERIES, EXPECT_OOV = 2520, 622", 'kill'),
        ('root-from-request', "root = Path(root)      # the queue root", "root = Path(req.get('root', root))      # the queue root", 'kill'),
        ('pythonpath-not-rewritten', "f'PYTHONPATH=src:{fusion_deploy}/src' if e.startswith('PYTHONPATH=') else e", "e", 'kill'),
        ('fusion-first', "f'PYTHONPATH=src:{fusion_deploy}/src'", "f'PYTHONPATH={fusion_deploy}/src:src'", 'kill'),
        ('config-sha-check-off', "    if get('--config-sha') != CONFIG_SHA:", "    if False:", 'kill'),
        ('check-only-missing', "    if STAGES[stage][1]:\n        cmd.append('--check-only')", "    if False:\n        cmd.append('--check-only')", 'kill'),
        ('vram-floor-off', "or type(vram_mib) is not int or vram_mib < 256", "or type(vram_mib) is not int", 'kill'),
        ('wall-cap-off', "or not 0 < wall_hours <= 4:", "or not 0 < wall_hours:", 'kill'),
        ('p-deploy-regex-open', "P_DEPLOY_PATTERN = r'/workspace/vkgr-g5-p-export100-v[0-9]+'", "P_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
        ('fusion-regex-open', "FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'", "FUSION_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
        ('duplicate-check-off', "    if exists_anywhere(root, job['run_id']):", "    if False:", 'kill'),
    ]},
    'asm_enqueue': {'target': 'ops/g5/g5_assemble_enqueue_20261018.py', 'copy': ['ops'], 'cmd': ['-m', 'unittest', 'discover', '-s', 'ops/g5', '-p', 'test_g5_assemble_enqueue*'], 'mutants': [
        ('prefix-match', "    return any(pattern.fullmatch(p.name) for sub", "    return any(rid in p.name for sub", 'kill'),
        ('claimed-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'rejected')", 'kill'),
        ('rejected-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'claimed')", 'kill'),
        ('runs-dir-not-checked', "    if (root / 'runs' / rid).exists():\n        return True\n", "", 'kill'),
        ('manifest-sha-check-off', "get('--manifest-sha') != MANIFEST_SHA[seed] or ", "False or ", 'kill'),
        ('cwd-check-off', "or t['cwd'] != g_deploy or ", "or ", 'kill'),
        ('entry-check-off', "or argv[argv.index('-m') + 1:argv.index('-m') + 3] != ['vkgr.coll_sid_a1_run', 'verify']:", ":", 'syntax'),
        ('check-only-missing', "    if STAGES[stage][1]:\n        cmd.append('--check-only')", "    if False:\n        cmd.append('--check-only')", 'kill'),
        ('fusion-first', "f'PYTHONPATH={g_deploy}/src:{fusion_deploy}/src'", "f'PYTHONPATH={fusion_deploy}/src:{g_deploy}/src'", 'kill'),
        ('pythonpath-check-off', "    if env == t['env'] or f'PYTHONPATH={g_deploy}/src' not in t['env']:", "    if False:", 'kill'),
        ('flag-dropped', "    for flag in FLAGS:\n        cmd += [flag, get(flag)]", "    for flag in FLAGS[:-2]:\n        cmd += [flag, get(flag)]", 'kill'),
        ('split-wrong', "'--split', 'validation']", "'--split', 'train']", 'kill'),
        ('vram-floor-off', "not 256 <= vram_mib <= 6656", "not vram_mib <= 6656", 'kill'),
        ('vram-ceiling-off', "not 256 <= vram_mib <= 6656", "not 256 <= vram_mib", 'kill'),
        ('wall-hours-ceiling', "not 0 < wall_hours <= 4", "not 0 < wall_hours <= 400", 'kill'),
        ('template-count-lt', "    if len(matches) != 1:", "    if len(matches) < 1:", 'kill'),
        ('submit-without-tmp', "tmp = pending / ('.' + name + '.tmp'); tmp.write_text(json.dumps(job)); os.replace(tmp, pending / name)", "tmp = pending / ('.' + name + '.tmp'); (pending / name).write_text(json.dumps(job)); os.replace(tmp, pending / name) if tmp.exists() else None", 'kill'),
        ('stdin-unbounded', "sys.stdin.read(65537)", "sys.stdin.read()", 'kill'),
        ('flags-first-renamed', "FLAGS = ('--prepared',", "FLAGS = ('--prepared-x',", 'kill'),
        ('unknown-action-ok', "    elif req['action'] != 'dry':\n        raise ValueError('unknown action')", "", 'kill'),
        ('job-sha-unsorted', "hashlib.sha256(json.dumps(j, sort_keys=True).encode())", "hashlib.sha256(json.dumps(j).encode())", 'kill'),
        ('seed-duplicates-ok', " or len(set(req['seeds'])) != len(req['seeds'])", "", 'kill'),
        ('seed-table-check-off', "or seed not in MANIFEST_SHA", "", 'equivalent'),   # unobservable: MANIFEST_SHA[seed] raises KeyError at the very next template identity check
        ('stage-check-off', "if stage not in STAGES or ", "if ", 'kill'),   # the error contract is ValueError; without the guard a KeyError would escape
        ('gen-limit-80', "if gen and (len(gen) > 8 or not gen.isalnum()):", "if gen and (len(gen) > 80 or not gen.isalnum()):", 'kill'),
        ('duplicate-check-off', "        if exists_anywhere(root, job['run_id']):", "        if False:", 'kill'),
        ('g-deploy-regex-open', "G_DEPLOY_PATTERN = r'/workspace/vkgr-coll-a1-g-v[0-9]+'", "G_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
    ]},
    'fuse_enqueue': {'target': 'ops/g5/g5_fuse_enqueue_20261018.py', 'copy': ['ops'], 'cmd': ['-m', 'unittest', 'discover', '-s', 'ops/g5', '-p', 'test_g5_fuse_enqueue*'], 'mutants': [
        ('prefix-match', "    return any(pattern.fullmatch(p.name) for sub", "    return any(rid in p.name for sub", 'kill'),
        ('claimed-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'rejected')", 'kill'),
        ('rejected-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'claimed')", 'kill'),
        ('runs-dir-not-checked', "    if (root / 'runs' / rid).exists():\n        return True\n", "", 'kill'),
        ('reason-form-dropped', "(json|reason|json\\.reason)", "(json)", 'kill'),
        ('tmp-form-dropped', "(\\.tmp)?'", "'", 'kill'),
        ('check-only-missing', "    if STAGES[stage][1]:\n        cmd.append('--check-only')", "    if False:\n        cmd.append('--check-only')", 'kill'),
        ('p-export-sha-wrong', "f'{seed}={export}@{P_EXPORT_MANIFEST_SHA[seed]}'", "f'{seed}={export}@{P_EXPORT_MANIFEST_SHA[42]}'", 'kill'),
        ('g-assembly-sha-wrong', "@{g_assembly_sha[str(seed)]}'", "@{g_assembly_sha['42']}'", 'kill'),
        ('meta-sha-wrong', "f'{meta}@{meta_sha}'", "f'{meta}@{\"0\" * 64}'", 'kill'),
        ('output-dir-wrong', "'--output-dir', str(runs / rid / 'out')", "'--output-dir', str(runs / rid)", 'kill'),
        ('split-wrong', "'--split', 'validation']", "'--split', 'train']", 'kill'),
        ('inputs-without-top100', "inputs += [str(export), str(export.with_name('top100.json')), str(assembly)]", "inputs += [str(export), str(assembly)]", 'kill'),
        ('pythonpath-not-fusion', "f'PYTHONPATH={fusion_deploy}/src'", "f'PYTHONPATH={fusion_deploy}'", 'kill'),
        ('pinned-files-kept', "disk_gib=1, pinned_files=[])", "disk_gib=1)", 'kill'),
        ('cwd-kept', "config=None, cwd=fusion_deploy,", "config=None,", 'kill'),
        ('module-wrong', "'-m', 'vkgr_g5.fuse']", "'-m', 'vkgr_g5.assemble_g']", 'kill'),
        ('deploy-regex-open', "FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'", "FUSION_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
        ('assembly-shas-check-off', "set(g_assembly_sha) != {'42', '2024', '2025'} or ", "", 'kill'),
        ('meta-sha-check-off', "or type(meta_sha) is not str or SHA.fullmatch(meta_sha) is None:", ":", 'kill'),
        ('vram-ceiling-off', "not 256 <= vram_mib <= 6656", "not 256 <= vram_mib", 'kill'),
        ('vram-floor-off', "not 256 <= vram_mib <= 6656", "not vram_mib <= 6656", 'kill'),
        ('gen-limit-80', "if gen and (len(gen) > 8 or not gen.isalnum()):", "if gen and (len(gen) > 80 or not gen.isalnum()):", 'kill'),
        ('template-count-lt', "    if len(matches) != 1:", "    if len(matches) < 1:", 'kill'),
        ('stdin-unbounded', "sys.stdin.read(65537)", "sys.stdin.read()", 'kill'),
        ('unknown-action-ok', "    elif req['action'] != 'dry':\n        raise ValueError('unknown action')", "", 'kill'),
        ('job-sha-unsorted', "hashlib.sha256(json.dumps(job, sort_keys=True).encode())", "hashlib.sha256(json.dumps(job).encode())", 'kill'),
        ('duplicate-check-off', "    if exists_anywhere(root, job['run_id']):", "    if False:", 'kill'),
    ]},
    'fuse': {'target': 'src/vkgr_g5/fuse.py', 'copy': ['src', 'tests_g5'], 'cmd': ['-m', 'unittest', 'tests_g5.test_fuse'], 'mutants': [
        ('p-gate-flag-off', "gate.get('first50_equal') is not True or ", "False or ", 'kill'),
        ('p-mismatch-count-off', "or gate.get('mismatched_queries') != 0", "", 'kill'),
        ('p-depth-off', "record.get('depth') != 100 or ", "False or ", 'kill'),
        ('p-binding-off', "    if record.get('train_manifest_sha256') != P_MANIFEST_SHA[seed]:", "    if False:", 'kill'),
        ('p-shape-off', "    if len(lists) != expected_queries or any(len(row) != 100 for row in lists):", "    if False:", 'kill'),
        ('p-short-row-ok', "any(len(row) != 100 for row in lists)", "any(len(row) < 99 for row in lists)", 'kill'),
        ('g-binding-off', "value.get('run_manifest_sha256') != G_MANIFEST_SHA[seed]", "False", 'kill'),
        ('g-checks-off', " or value['checks'] != {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True}", "", 'kill'),
        ('g-seed-off', "value.get('seed') != seed or ", "False or ", 'kill'),
        ('align-history-off', "(p['user_id'], p['target'], p['oov'], p['history']) != (g['user_id'], g['target'], g['oov'], g['history'])", "(p['user_id'], p['target'], p['oov']) != (g['user_id'], g['target'], g['oov'])", 'kill'),
        ('align-target-off', "(p['user_id'], p['target'], p['oov'], p['history']) != (g['user_id'], g['target'], g['oov'], g['history'])", "(p['user_id'], p['oov'], p['history']) != (g['user_id'], g['oov'], g['history'])", 'kill'),
        ('align-only-seed-42', "for seed, assembly in assemblies.items():", "for seed, assembly in list(assemblies.items())[:1]:", 'kill'),
        ('hits-check-off', "    if got != expect_hits:", "    if False:", 'kill'),
        ('hits-g-skipped', "got.update({('G', s): single_route_hits([r['items'] for r in assemblies[s]['rows']], targets) for s in SEEDS})", "got.update({('G', s): expect_hits[('G', s)] for s in SEEDS})", 'kill'),
        ('hits-uses-top30', "return sum(1 for ranked, target in zip(lists, targets) if target in ranked[:50])", "return sum(1 for ranked, target in zip(lists, targets) if target in ranked[:30])", 'kill'),
        ('sha-pin-off', "sha256_file(path) != digest", "False", 'kill'),
        ('held-out-check-off', "            if HELD_OUT_COMPONENT.search(part):", "            if False:", 'kill'),
        ('held-out-substring', "HELD_OUT_COMPONENT = re.compile(r'(?i)(^|[-_.])(test|week[-_]?26)([-_.]|$)')", "HELD_OUT_COMPONENT = re.compile(r'(?i)test')", 'kill'),
        ('output-exists-ok', "        if out.exists():\n            raise ValueError('output directory must not exist')", "", 'kill'),
        ('check-only-writes', "        if args.check_only:\n            print(json.dumps({'status': 'preflight_ok'", "        if False:\n            print(json.dumps({'status': 'preflight_ok'", 'kill'),
        ('catalog-size-check-off', "a['catalog_size'] != meta['catalog_size'] or ", "False or ", 'kill'),
        ('pop-from-wrong-seed', "routes[('POP', None)] = base['pop']['ranking']", "routes[('POP', None)] = assemblies[2025]['pop']['ranking']", 'equivalent'),   # unobservable: the cross-seed Pop-equality gate (tested) already forces every seed's list to be identical
        ('queries-from-wrong-labels', "'popularity_group': r['popularity_group'], 'user_group': r['user_group'], 'history_len': len(r['history'])}", "'popularity_group': r['popularity_group'], 'user_group': r['user_group'], 'history_len': 1}", 'kill'),
        ('support-mismatch-not-counted', "support_mismatch = sum(1 for p, g in zip(meta['rows'], base['rows']) if p['support_p'] != g['support'])", "support_mismatch = 0", 'kill'),
        ('query-count-meta-off', " or meta.get('query_count') != expected_queries:", ":", 'syntax'),
        ('seeds-required-off', "    if set(out) != set(SEEDS):", "    if False:", 'kill'),
        ('duplicate-seed-ok', "or int(seed) in out:", ":", 'syntax'),
        ('fusion-hash-not-recorded', "'fusion_v2_script_sha256': fv.script_sha256(),", "'fusion_v2_script_sha256': '0' * 64,", 'kill'),
        ('p-status-off', "record.get('status') != 'complete' or record.get('schema')", "False or record.get('schema')", 'kill'),
        ('p-schema-off', "record.get('schema') != 'p-seqm1-export100-v1' or ", "False or ", 'kill'),
        ('p-queries-off', "or record.get('queries') != expected_queries", "", 'kill'),
        ('p-seed-off', "or record.get('seed') != seed:", ":", 'kill'),
        ('g-split-off', "value.get('split') != 'validation' or ", "False or ", 'kill'),
        ('g-schema-off', "value.get('schema') != 'g5-g-assembly-v1' or ", "False or ", 'kill'),
        ('g-query-count-off', "or value.get('query_count') != expected_queries", "", 'kill'),
        ('meta-schema-off', "meta.get('schema') != 'g5-p-query-meta-v1' or ", "False or ", 'kill'),
        ('meta-split-off', "or meta.get('split') != 'validation'", "", 'kill'),
        ('meta-query-count-off', "or meta.get('query_count') != expected_queries", "", 'kill'),
        ('meta-manifests-checked-off', "or meta.get('manifests_checked') != 3", "", 'kill'),
        ('meta-history-len-off', "or meta.get('history_len') != 50", "", 'kill'),
        ('p-ordinal-off', "p['ordinal'] != ordinal or ", "False or ", 'kill'),
        ('g-ordinal-off', "or g['ordinal'] != ordinal", "", 'kill'),
        ('row-count-gate-off', "        if len(grows) != len(rows):", "        if False:", 'kill'),
        ('symlink-gate-off', "path.is_symlink() or ", "False or ", 'kill'),
        ('regular-file-gate-off', "or not path.is_file() or ", "or ", 'kill'),
        ('held-out-output-unchecked', "refuse_held_out_paths([out, ", "refuse_held_out_paths([", 'kill'),
        ('held-out-meta-unchecked', "meta_input[0], *[v[0] for v in p_inputs.values()]", "*[v[0] for v in p_inputs.values()]", 'kill'),
        ('held-out-p-unchecked', "*[v[0] for v in p_inputs.values()], *[v[0] for v in g_inputs.values()]]", "*[v[0] for v in g_inputs.values()]]", 'kill'),
        ('held-out-g-unchecked', ", *[v[0] for v in g_inputs.values()]])", "])", 'kill'),
        ('top100-sha-unchecked', "read_pinned_json(Path(export_manifest).with_name('top100.json'), record['top100_sha256'])", "json.loads(Path(export_manifest).with_name('top100.json').read_text())", 'kill'),
        ('meta-sha-unchecked', "    meta = read_pinned_json(*meta_input)", "    meta = json.loads(Path(meta_input[0]).read_text())", 'kill'),
        ('g-sha-unchecked', "    value = read_pinned_json(path, digest)\n    if value.get('schema')", "    value = json.loads(Path(path).read_text())\n    if value.get('schema')", 'kill'),
        ('support-label-constant', "'support': r['support'], 'popularity_group'", "'support': 'supported', 'popularity_group'", 'kill'),
        ('oov-flag-constant-false', "queries = [{'target': r['target'], 'oov': r['oov'],", "queries = [{'target': r['target'], 'oov': False,", 'kill'),
        ('g-routes-top50-only', "routes.update({('G', s): [r['items'] for r in assemblies[s]['rows']] for s in SEEDS})", "routes.update({('G', s): [r['items'][:50] for r in assemblies[s]['rows']] for s in SEEDS})", 'kill'),
        ('utc-dropped', "datetime.now(timezone.utc).isoformat()", "datetime.now().isoformat()", 'kill'),
        ('nan-allowed-in-output', "payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()", "payload = json.dumps(result, sort_keys=True).encode()", 'equivalent'),
        ('nan-allowed-in-preflight', "serialized = len(json.dumps(result, sort_keys=True, allow_nan=False))", "serialized = len(json.dumps(result, sort_keys=True))", 'kill'),
        ('preflight-skips-serialisation', "serialized = len(json.dumps(result, sort_keys=True, allow_nan=False))", "serialized = 1", 'kill'),
        ('catalog-digest-unchecked', " or a.get('catalog_digest') != meta['catalog_digest']", "", 'kill'),
        ('shared-fields-unchecked', "any(assembly[k] != base[k] for k in shared) or ", "False or ", 'kill'),
        ('pop-lists-unchecked', " or assembly['pop']['ranking'] != base['pop']['ranking']:", ":", 'kill'),
        ('label-agreement-unchecked', "for r in assembly['rows']] != [(r['support']", "for r in base['rows']] != [(r['support']", 'kill'),
        ('mkdir-exist-ok', "out.mkdir(parents=True, exist_ok=False)", "out.mkdir(parents=True, exist_ok=True)", 'equivalent'),
        ('json-xb-to-wb', "(out / 'fusion-results.json').open('xb')", "(out / 'fusion-results.json').open('wb')", 'equivalent'),
    ]},
    'assemble': {'target': 'src/vkgr_g5/assemble_g.py', 'copy': ['src', 'tests_g5'], 'cmds': [['-m', 'unittest', 'tests_g5.test_assemble_g.LabelTests', 'tests_g5.test_assemble_g.GateTests', 'tests_g5.test_assemble_g.CoreTests', 'tests_g5.test_assemble_g.CliTests'], ['-m', 'unittest', 'tests_g5.test_assemble_g.IntegrationTests']], 'mutants': [
        ('popularity-boundary', "count <= stats.thresholds[0] else item_groups[2] if count <= stats.thresholds[1]", "count < stats.thresholds[0] else item_groups[2] if count <= stats.thresholds[1]", 'kill'),
        ('user-boundary', "users.append(", "users.append", 'syntax'),
        ('support-flip', "'supported' if count > 0 else 'unsupported'", "'unsupported' if count > 0 else 'supported'", 'kill'),
        ('oov-popularity', "popularity.append('oov'); support.append('oov')", "popularity.append('oov'); support.append('unsupported')", 'kill'),
        ('history-gate-off', "        if not 1 <= length <= 50:", "        if False:", 'kill'),
        ('gate-section-skipped', "    for section, names in spec:", "    for section, names in spec[:2]:", 'kill'),
        ('pop-tie-rule', "key=lambda i: (-stats.item_counts[i], i))", "key=lambda i: (-stats.item_counts[i], -i))", 'kill'),
        ('pop-depth-gate-off', "    if len(ranking) < k:", "    if False:", 'kill'),
        ('ordinal-order-gate-off', " or [r['ordinal'] for r in rows] != list(range(len(queries))):", ":", 'syntax'),
        ('metric-gate-off', "    if [main['recall@50'], main['ndcg@50']] != list(best['metric']):", "    if False:", 'kill'),
        ('archive-replay-gate-off', "    if canonical_bytes(aggregate) != canonical_bytes(record['metrics']) or computation != record['evaluation']['computation_sha256']:", "    if False:", 'kill'),
        ('kind-gate-off', "    if record.get('epoch') != epoch or record['evaluation']['kind'] != 'query_evaluation':", "    if False:", 'kill'),
        ('consistency-gate-off', "    consistency_gate(items, queries, context.catalog, assigned, aggregate, ev.ITEM_GROUPS, ev.USER_GROUPS, metrics, canonical_bytes)", "    pass", 'kill'),
        ('items-copied-from-wrong-row', "items = [list(r['items']) for r in rows]", "items = [list(r['items']) for r in rows[::-1]]", 'kill'),
        ('best-hash-gate-off', "    if best.get('sha256') != record['checkpoint_sha256']:", "    if False:", 'kill'),
        ('best-epoch-type-off', "type(epoch) is not int or ", "", 'kill'),
        ('best-epoch-range-off', "or not 1 <= epoch <= len(value['epochs'])", "", 'kill'),
        ('record-epoch-eq-off', "record.get('epoch') != epoch or ", "", 'kill'),
        ('metric-only-recall', "if [main['recall@50'], main['ndcg@50']] != list(best['metric']):", "if [main['recall@50']] != list(best['metric'])[:1]:", 'kill'),
        ('g-short-boundary', "if len(r['items']) < 100)", "if len(r['items']) <= 100)", 'kill'),
        ('pop-hits-top100', "set(pop[:50])", "set(pop[:100])", 'kill'),
        ('pop-hits-top49', "set(pop[:50])", "set(pop[:49])", 'kill'),
        ('oov-output-false', "'oov': bool(query['oov'])", "'oov': False", 'kill'),
        ('history-truncated-49', "'history': [int(i) for i in query['history']]", "'history': [int(i) for i in query['history']][:49]", 'kill'),
        ('context-binding-gate-off', "        if cr._digest(context.binding) != recorded['context_binding_sha256']:", "        if False:", 'kill'),
        ('source-files-gate-off', "        if fresh['files'] != recorded['source']['files'] or ", "        if ", 'kill'),
        ('source-sha-gate-off', " or fresh['source_sha256'] != recorded['source']['source_sha256']:", ":", 'kill'),
        ('check-only-writes', "        if args.check_only:\n", "        if False:\n", 'kill'),
        ('scratch-exists-ok', "            if root in path.parents or path.exists() or", "            if root in path.parents or", 'kill'),
        ('scratch-partial-ok', "or path.with_name(path.name + '.partial').exists():", ":", 'kill'),
        ('output-in-deployment-ok', "            if root in path.parents or path.exists()", "            if path.exists()", 'kill'),
        ('out-scratch-nesting-ok', " or out in scratch.parents or scratch in out.parents:", ":", 'kill'),
        ('seed-recorded-wrong', "seed=config['seed']", "seed=42", 'kill'),
        ('manifest-sha-recorded-wrong', "run_manifest_sha256=args.manifest_sha", "run_manifest_sha256='0' * 64", 'kill'),
        ('schema-renamed', "SCHEMA = 'g5-g-assembly-v1'", "SCHEMA = 'g5-g-assembly-v2'", 'kill'),
        ('catalog-digest-dropped', "digest=cr._digest)", "digest=None)", 'kill'),
        ('script-sha-wrong', "script_sha256=script_sha256())", "script_sha256='0' * 64)", 'kill'),
        ('mkdir-parent-dropped', "        out.parent.mkdir(parents=True, exist_ok=True)\n", "", 'kill'),
        ('vkgr-ancestor-ok', "        if Path(vkgr.__file__).resolve().parent != root / 'src' / 'vkgr':", "        if root not in Path(vkgr.__file__).resolve().parents:", 'kill'),
        ('engineering-ok', "        if engineering:\n            raise ValueError('formal (non-engineering) context required')", "", 'kill'),
    ]},
}


def suite_run(work, cmd):
    if cmd and isinstance(cmd[0], list):          # several isolated interpreters (the P and G code bases both provide a package named vkgr)
        results = [suite_run(work, c) for c in cmd]
        return max(r[0] for r in results), sorted({t for r in results for t in r[1]})
    cache = tempfile.mkdtemp(prefix='mut-pyc-')
    env = {**os.environ, 'PYTHONPYCACHEPREFIX': cache, 'PYTHONDONTWRITEBYTECODE': '1', **EXTRA_ENV}
    proc = subprocess.run([sys.executable, '-B', *cmd], cwd=work, env=env, capture_output=True, text=True)
    return proc.returncode, sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+) ', proc.stderr, re.M)))


EXTRA_ENV = {}


def main(name):
    spec = SUITES[name]
    EXTRA_ENV.update(spec.get('env', {}))
    if name == 'assemble' and not os.environ.get('VKGR_G5_G_ROOT'):
        raise SystemExit('the assemble suite needs VKGR_G5_G_ROOT (the G code base) in the environment')
    print(f'# Mutation ledger: {name} ({spec["target"]})\n')
    print('Isolation: every run uses a fresh copy of the sources, `python -B`, `PYTHONDONTWRITEBYTECODE=1` and a unique `PYTHONPYCACHEPREFIX`.\n')
    def fresh():
        work = Path(tempfile.mkdtemp(prefix='mut-'))
        for part in spec['copy']:
            shutil.copytree(ROOT / part, work / part, ignore=shutil.ignore_patterns('__pycache__'))
        return work
    base_code, failed = suite_run(fresh(), spec.get('cmds') or spec['cmd'])
    print(f'Baseline: returncode {base_code}, failing tests {failed}\n')
    print('| id | change | expected | result | killed by |\n|---|---|---|---|---|')
    bad = 0
    for mid, old, new, expected in spec['mutants']:
        if expected == 'syntax':
            continue
        work = fresh(); target = work / spec['target']; source = target.read_text()
        if source.count(old) < 1:
            print(f'| {mid} | {old[:50]!r} | {expected} | PATTERN_NOT_FOUND | |'); bad += 1; continue
        target.write_text(source.replace(old, new, 1))
        code, failed = suite_run(work, spec.get('cmds') or spec['cmd'])
        result = 'killed' if code != 0 else 'SURVIVED'
        if (expected == 'kill') != (result == 'killed'):
            bad += 1
        print(f'| {mid} | {new[:60]!r} | {expected} | {result} | {", ".join(failed[:3]) or "-"} |')
    print(f'\nMutants: {len([m for m in spec["mutants"] if m[3] != "syntax"])}; unexpected outcomes: {bad}.')
    return 1 if bad or base_code else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
