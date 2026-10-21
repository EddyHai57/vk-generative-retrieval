"""Prepare only ad1c428 subset A1 new-family inputs; no training or test reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from vkgr import sgns_subset_a1 as a1,sgns_subset_a1_inputs as core
ROOT=Path(__file__).resolve().parents[1]
RUN_ID='sgns-gpu-subset-a1-inputs-v1'


def invocation(path,sha):
    return [sys.executable,'-B','-m','analysis.sgns_subset_a1_inputs','derive','--request',str(path),'--request-sha256',sha]


def derive(path,sha):
    from analysis.ur001_derive_cli import source_identity,guards,progress,encoded
    from analysis.new_instance_prepare import runtime
    value=core._json({'path':str(path),'sha256':sha})
    keys={'schema','source_commit','lock','raw_data','ur001','content','gpu_sid_manifest_sha256','output_root','ops_root'}
    if type(value) is not dict or set(value)!=keys or value['schema']!='gpu-sgns-subset-a1-input-request-v1':raise ValueError('explicit input request required')
    source=source_identity()
    if source['commit']!=value['source_commit'] or value['lock']['sha256']!=a1.NATIVE_LOCK_SHA or value['content']['sha256']!=a1.INPUT_PINS['content']:raise ValueError('fixed source/runtime/content required')
    a1.check_native();env=runtime(a1.absolute(value['lock']['path']))
    records=[{'path':str(path),'sha256':sha},value['lock'],value['ur001'],value['content']]
    raw=(a1.absolute(value['ops_root'])/'runs'/RUN_ID/'run.json').read_bytes();run=json.loads(raw)
    if (os.environ.get('VKGR_RUN_ID')!=RUN_ID or run.get('run_id')!=RUN_ID or run.get('kind')!='ops' or run.get('seed')!=42
            or run.get('cwd')!=str(ROOT) or run.get('git')!={'commit':source['commit'],'dirty':False,'dirty_files':[]}
            or run.get('argv')!=invocation(path,sha) or run.get('config',{}).get('sha256')!=sha
            or run.get('config',{}).get('path')!=str(path)
            or [{'path':r['path'],'sha256':r['sha256']} for r in run.get('inputs',[])]!=records):
        raise ValueError('original input preparation job/source/argv/inputs differs')
    for record in records:a1.file_bytes(record)
    output=a1.absolute(value['output_root'])
    for immutable in ('raw_data','ops_root'):
        data=a1.absolute(value[immutable])
        if output==data or output in data.parents or data in output.parents:raise ValueError('exclusive output separate from raw/OPS required')
    if output==ROOT or output in ROOT.parents or ROOT in output.parents:raise ValueError('input output outside deployment required')
    guards(output)
    progress('a1_inputs_raw_start')
    payload,parts,raw_inputs,producer=core._raw_payloads(value['raw_data'],value['ur001'])
    guards(output);progress('a1_inputs_native_payloads_verified')
    result=core.write_family(output,payload,parts,raw_data=value['raw_data'],raw_records=raw_inputs['files'],ur001=value['ur001'],content=value['content'],
                             sid_sha=value['gpu_sid_manifest_sha256'],source=source,runtime=env)
    for record in records:a1.file_bytes(record)
    if source_identity()!=source or runtime(a1.absolute(value['lock']['path']))!=env:raise ValueError('input source/runtime changed')
    receipt={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,'run_id':RUN_ID,
             'run_manifest_sha256':hashlib.sha256(raw).hexdigest(),'request_sha256':sha,'source':source,'runtime':env,'output':result,
             'science_training':False,'test_read':False,'subset_four_gate_acceptance':False}
    with (output/'stage-receipt.json').open('xb') as f:f.write(encoded(receipt))
    return receipt


def prepare(path,sha,ops_code):
    from analysis.a40_ops import load_ops
    value=core._json({'path':str(path),'sha256':sha})
    if value.get('schema')!='gpu-sgns-subset-a1-input-request-v1':raise ValueError('fixed input request required')
    _,launcher,_,_=load_ops(ops_code)
    ops=a1.absolute(value['ops_root'])
    if (ops/'runs'/RUN_ID).exists():raise FileExistsError('fixed input job already exists')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+RUN_ID+'.json')):raise FileExistsError('fixed input job queued')
    argv=['--root',str(ops),'--run-id',RUN_ID,'--kind','ops','--cwd',str(ROOT),
          '--config',str(path),'--seed','42','--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONPATH='+str(ROOT/'src'),
          '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--cpu-cores','1','--ram-mib','98304',
          '--vram-mib','0','--disk-gib','4','--wall-hours','2']
    for record in [{'path':str(path),'sha256':sha},value['lock'],value['ur001'],value['content']]:
        a1.file_bytes(record);argv+=['--input',record['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*argv,'--',*invocation(path,sha)]))
    metadata=launcher.prepare(spec)
    if metadata['git']['commit']!=value['source_commit'] or metadata['git']['dirty_files']:raise ValueError('fixed clean input source required')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'real_preflight':False,
            'resource_scope':'not_measured conservative first real input preparation','subset_four_gate_acceptance':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);s.add_parser('dry')
    d=s.add_parser('derive');d.add_argument('--request',type=Path,required=True);d.add_argument('--request-sha256',required=True)
    d=s.add_parser('prepare');d.add_argument('--request',type=Path,required=True);d.add_argument('--request-sha256',required=True);d.add_argument('--ops-code',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':value=prepare(args.request,args.request_sha256,args.ops_code)
    else:value={'status':'dry','science_training':False,'test_read':False,'subset_only':True} if args.command=='dry' else derive(args.request,args.request_sha256)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
