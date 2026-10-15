"""FEAT-04b stage 2 (E = f4_recency_behavior) detached verify driver. NO model calls. Stdlib only. Polls every POLL seconds (>= 900).
For each of the 3 E runs: when runs/<train>/terminal.json says succeeded -> enqueue <train>-verify (derived from the claimed P-HP-02 s42 verify job: only argv/config/cwd/inputs/run_id differ),
then check it landed in queue/claimed (not rejected, runs/<verify> exists) and finally record the verify terminal status. Train failed -> log, no verify.
Differences from the stage 1 (C/D) driver: E inputs spec, the deploy dir x2, NO stats manifest in the verify inputs (E has no stats pin), and a heartbeat line every poll.
Writes only: queue/pending (verify jobs, atomic tmp + os.replace, never overwrites) and its own dir (driver.pid / driver.log / state.json). Never touches other jobs. Never reads test."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
POLL = 900; DEADLINE_H = 36
R = Path('/workspace/vkgr-infra01-root'); Q = R / 'queue'; DEP = Path('/workspace/vkgr-feat04b-p-x2')
HERE = Path(__file__).resolve().parent
SPEC = '/workspace/vkgr-b1-inputs/p-feat04b-inputs-e-v1.json'; SPEC_SHA = '54c48b310424d7b28581ad3f33c52f9b141d08fab6dc88fcec6374fbb0407f3b'
CFGS = {
 "p-feat04b-e-q2-behavior-s2024": "ac8fcb244a92c7ae3c63b3fe120a1ca65ed83afd9b3cbacb43f29c2ca4782b12",
 "p-feat04b-e-q2-behavior-s2025": "27e67a3e42afc53b189e83d6162bac5661295d433c0207bedac7d4d57b130800",
 "p-feat04b-e-q2-behavior-s42": "484e6aab15b6a7bfd30909432b963f25a5486a89e66abb0829aedbfb0d07938f"
}
FINAL = ('done', 'train_not_succeeded', 'verify_rejected')
STATE = HERE / 'state.json'
def log(msg):
    with open(HERE / 'driver.log', 'a') as f: f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')
def fsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {rid: {'phase': 'wait_train'} for rid in CFGS}
def save_state(s):
    tmp = HERE / '.state.tmp'; tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, STATE)
def queue_where(rid):
    return {d: [p.name for p in (Q / d).glob(f'*-{rid}.json*')] for d in ('pending', 'claimed', 'rejected')}
def reason(rid):
    out = {}
    for n in queue_where(rid)['rejected']:
        rp = Q / 'rejected' / n
        for c in (Path(str(rp) + '.reason'), rp.with_suffix('.reason')):
            if c.exists(): out[c.name] = c.read_text()[:600]
    return out
def terminal(rid):
    t = R / 'runs' / rid / 'terminal.json'
    return json.loads(t.read_text()).get('status') if t.exists() else None
def enqueue_verify(rid, sha):
    v = rid + '-verify'
    if (R / 'runs' / v).exists() or any(queue_where(v).values()): return 'exists'
    fs = sorted(Q.glob('claimed/*-p-php02-t005-s42-verify.json')); assert len(fs) == 1
    T = json.loads(fs[0].read_text()); ta = T['argv']
    man = R / 'runs' / rid / 'out/manifest.json'; assert man.is_file()
    argv = list(ta)
    argv[argv.index('-m') + 1] = 'vkgr.retrieval_feat04b_run'
    argv[argv.index('--inputs') + 1] = SPEC; argv[argv.index('--inputs-sha') + 1] = SPEC_SHA
    argv[argv.index('--config') + 1] = f'configs/feat04b/{rid}.json'; argv[argv.index('--config-sha') + 1] = sha
    argv[argv.index('--manifest') + 1] = str(man); argv[argv.index('--manifest-sha') + 1] = fsha(man)
    assert len(T['inputs']) == 4 and T['inputs'][3].endswith('/p-php02-t005-s42/out/manifest.json')
    inputs = [SPEC, T['inputs'][1], T['inputs'][2], str(man)]            # E has no stats manifest (stage 1 C/D inserted it before the train manifest)
    d = {**T, 'run_id': v, 'cwd': str(DEP), 'config': str(DEP / 'configs/feat04b' / (rid + '.json')), 'argv': argv, 'inputs': inputs}
    diff = sorted(k for k in set(d) | set(T) if d.get(k) != T.get(k))
    assert diff == ['argv', 'config', 'cwd', 'inputs', 'run_id'], diff
    assert fsha(d['config']) == sha and 'php02' not in json.dumps(d['argv']) and 'php02' not in json.dumps(d['inputs'])
    assert d['argv'][4] == 'verify' and fsha(SPEC) == SPEC_SHA
    assert not any('stats' in str(x) for x in d['inputs']) and 'x1' not in json.dumps(d['argv']) and 'x1' not in d['cwd']
    assert (d['cpu_cores'], d['ram_mib'], d['vram_mib'], d['disk_gib']) == (1, 9216, 1024, 1)
    name = f'{time.time_ns():020d}-{v}.json'; tmp = Q / 'pending' / ('.' + name + '.tmp')
    tmp.write_text(json.dumps(d)); os.replace(tmp, Q / 'pending' / name)
    return 'written'
def main():
    (HERE / 'driver.pid').write_text(str(os.getpid()))
    log(f'driver start pid={os.getpid()} ppid={os.getppid()} poll={POLL}s deadline={DEADLINE_H}h')
    t0 = time.time(); st = load_state()
    while time.time() - t0 < DEADLINE_H * 3600:
        for rid, sha in CFGS.items():
            s = st[rid]; v = rid + '-verify'
            try:
                if s['phase'] == 'wait_train':
                    t = terminal(rid)
                    if t is None: continue
                    if t != 'succeeded': s['phase'] = 'train_not_succeeded'; s['train_terminal'] = t; log(f'{rid} train terminal={t}: no verify'); continue
                    r = enqueue_verify(rid, sha); s['phase'] = 'verify_enqueued'; log(f'{rid} train succeeded -> verify {r}')
                elif s['phase'] == 'verify_enqueued':
                    w = queue_where(v)
                    if w['rejected']: s['phase'] = 'verify_rejected'; s['reason'] = reason(v); log(f'{v} REJECTED {s["reason"]}'); continue
                    if w['claimed'] and (R / 'runs' / v).is_dir():
                        s['phase'] = 'wait_verify'; log(f'{v} in claimed, runs dir exists')
                elif s['phase'] == 'wait_verify':
                    t = terminal(v)
                    if t is not None: s['phase'] = 'done'; s['verify_terminal'] = t; log(f'{v} terminal={t}')
            except Exception as e:  # keep the driver alive; the next poll retries
                log(f'{rid} ERROR {type(e).__name__}: {e}')
        save_state(st)
        log('heartbeat ' + json.dumps({k.rsplit('-', 1)[1]: x['phase'] for k, x in st.items()}))
        if all(x['phase'] in FINAL for x in st.values()):
            log('all runs final: ' + json.dumps({k: x['phase'] for k, x in st.items()})); return
        time.sleep(POLL)
    log('deadline reached, driver exits (state kept)')
if __name__ == '__main__':
    try: main()
    except BaseException:
        log('FATAL ' + traceback.format_exc()[-1500:]); raise
