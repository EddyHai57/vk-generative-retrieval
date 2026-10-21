"""Original PF wait only; no live scheduler or scientific execution."""
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from ops.tools.remote import p_first_batch_chain as m

@pytest.mark.parametrize('fault',['','missing','rejected','failed','timeout','job','request','identity'])
def test_wait_original_pf_then_existing_acceptance_only(tmp_path,fault):
    ops=tmp_path/'ops';run=ops/'runs/pf';run.mkdir(parents=True)
    claimed=ops/'queue/claimed/000-pf.json';claimed.parent.mkdir(parents=True);claimed.write_text('{}')
    base={'ops_root':str(ops),'run_ids':{'preflight':'pf'},'frozen':True}
    expected={'run_id':'pf','seed':42}
    (run/'run.json').write_text(json.dumps(expected))
    backend=SimpleNamespace(read_transport=lambda *a:base,stage_inputs=lambda *a:[],job_contract=lambda *a:expected,
                            strict_json=json.loads)
    args=SimpleNamespace(inputs=tmp_path/'inputs',request=tmp_path/'request',request_sha256='a'*64,evidence=tmp_path)
    events=[];clock=[0]
    def identity():
        events.append('identity')
        if fault=='identity':raise ValueError('identity drift')
    def sleep(seconds):
        assert seconds>=900;clock[0]+=seconds;events.append('sleep')
        if fault=='missing':claimed.unlink()
        elif fault=='rejected':
            rejected=ops/'queue/rejected'/claimed.name;rejected.parent.mkdir(parents=True);claimed.rename(rejected)
        elif fault=='job':(run/'run.json').write_text(json.dumps({**expected,'seed':2024}))
        elif fault=='request':backend.read_transport=lambda *a:{**base,'frozen':False}
        elif fault!='timeout':
            (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':'pf','status':'failed' if fault=='failed' else 'succeeded','exit_code':2 if fault=='failed' else 0,'signal':None,'wall_cap_exceeded':False}))
    if fault:
        with pytest.raises(TimeoutError if fault=='timeout' else ValueError):
            m.wait_preflight(args,base,backend,identity,1800,now=lambda:clock[0],sleep=sleep)
    else:
        m.wait_preflight(args,base,backend,identity,1800,now=lambda:clock[0],sleep=sleep)
        assert events==['identity','sleep','identity']
    assert not (tmp_path/'train-request.json').exists()
    assert not (ops/'queue/pending').exists()
    assert not (tmp_path/'summary.json').exists()


def test_wait_terminal_needs_claimed_and_does_not_reset_deadline(tmp_path):
    args=SimpleNamespace(evidence=tmp_path)
    with pytest.raises(TimeoutError):
        m.wait_preflight(args,{},None,lambda:None,0,now=lambda:0,sleep=lambda _:pytest.fail('deadline reset'))


def test_formal_run_parser_accepts_wait_option_and_keeps_dry_receipt(monkeypatch,tmp_path):
    import subprocess,sys
    run=subprocess.run([sys.executable,'-B','-m','ops.tools.remote.p_first_batch_chain','run','--wait-preflight','--inputs',str(tmp_path/'i'),'--request',str(tmp_path/'r'),'--request-sha256','a'*64,'--ops-code',str(tmp_path/'ops'),'--consumer-code',str(tmp_path/'code'),'--evidence',str(tmp_path/'e'),'--deadline-hours','200','--driver-sha256','b'*64],capture_output=True,text=True)
    assert run.returncode!=0
    assert 'unrecognized arguments' not in run.stderr
    assert 'detached PPID=1 required' in run.stderr

@pytest.mark.parametrize('wait_failure',[False,True])
def test_chain_waits_before_unchanged_pf_acceptance_and_any_submit(tmp_path,monkeypatch,wait_failure):
    import analysis.ur001_derive_cli as derive
    import analysis.new_instance_prepare as environment
    base={'predecessor':None,'config':{'seed':42},'native_root':str(tmp_path/'native'),'source':{},'runtime':{},'lock':{},
          'job_contract':{s:{'ops_source':{},'resources':{'wall_seconds':345600}} for s in ('preflight','train','verify')}}
    backend=SimpleNamespace(read_transport=lambda *a:base,native_for=lambda p:SimpleNamespace(default_config=lambda:base['config']),record_path=lambda a:None)
    monkeypatch.setattr(derive,'source_identity',lambda:{})
    monkeypatch.setattr(environment,'runtime',lambda p:{})
    events=[]
    def wait(*args,**kwargs):
        events.append('wait')
        assert args[4]==720000
        if wait_failure:raise ValueError('wait stopped')
    def complete(*args):
        events.append('complete')
        raise ValueError('existing acceptance reached')
    monkeypatch.setattr(m,'wait_preflight',wait);monkeypatch.setattr(m,'complete',complete)
    args=SimpleNamespace(inputs=tmp_path/'i',request=tmp_path/'r',request_sha256='a'*64,deadline_hours=200,wait_preflight=True)
    with pytest.raises(ValueError,match='wait stopped' if wait_failure else 'existing acceptance reached'):
        m.run_chain(args,backend,None,SimpleNamespace(check_consumer=lambda *a:None),SimpleNamespace(_ops_source=lambda:{}),SimpleNamespace(submit=lambda *a:pytest.fail('premature submit')),now=lambda:0,sleep=lambda _:pytest.fail('unexpected sleep'))
    assert events==(['wait'] if wait_failure else ['wait','complete'])
