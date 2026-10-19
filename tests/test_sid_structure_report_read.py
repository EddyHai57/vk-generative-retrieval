"""Fixed report contracts; tiny fixtures, no scientific input or model runs."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

def tool():
    p=ROOT/'analysis/sid_structure_report_read.py'
    assert p.is_file(),'missing read-only structure reporter'
    spec=importlib.util.spec_from_file_location('structure_report',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class StructureReportTests(unittest.TestCase):
    def test_exact_six_authorized_runs_no_subset_or_s2(self):
        t=tool();t.validate_run_ids(list(t.RUNS))
        for ids in (list(t.RUNS)[:3],list(reversed(t.RUNS)),[*t.RUNS[:-1],'g-learn-a1-s2025']):
            with self.assertRaises(ValueError):t.validate_run_ids(ids)

    def test_original_metrics_hand_calculated_bins_and_collision_definitions(self):
        t=tool();_,metrics=t.frozen(ROOT)
        mapping={1:(0,0,0,0),2:(0,0,0,1),3:(1,0,0,0)}
        a=t.catalog_structure(mapping,48,metrics)
        self.assertEqual(a['semantic_collision_items'],2)
        self.assertEqual(a['surplus_collision_items'],1)
        self.assertEqual(a['collision_pairs'],1)
        self.assertEqual(a['leaf_size_histogram'],[[1,1],[2,1]])
        self.assertEqual(a['S0']['layers'][0]['avg_items_per_node'],1.5)
        r=t.summarize_pair([1,2,99],[[1],[3],[1]],[[2],[2],[1]],mapping,mapping,
                           {1:10,2:1,3:3},{'median':2,'p90':5},metrics)
        self.assertEqual(r['delta_hits'],0)
        self.assertEqual(r['popularity_delta_hits'],{'oov':0,'unsupported':0,'low':-1,'mid':0,'high':1})
        self.assertEqual(r['new']['B']['n_in_catalog'],2)
        self.assertEqual(r['new']['B']['n_oov'],1)
        self.assertTrue(r['prefix_descriptive_only'])
        self.assertNotIn('normalized_prefix_gain',r)
        self.assertNotIn('predictions',r)

    def test_population_and_item_hits_are_independently_checked(self):
        t=tool();_,metrics=t.frozen(ROOT)
        b=metrics.prediction_metrics([1,99],[[1],[1]],{1:(0,0,0,0)})
        with self.assertRaises(ValueError):t.population(b,1)
        with self.assertRaises(ValueError):t.population(b,0,total=2,in_catalog=1,oov=1)
        t.population(b,1,total=2,in_catalog=1,oov=1)

    def test_safe_paths_reject_forbidden_ancestors_and_symlinks_before_read(self):
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();f=root/'good.json';f.write_text('{}');link=root/'alias.json';link.symlink_to(f)
            with self.assertRaises(ValueError):t.safe(link,[root])
            bad=root/'test'/'secret.json';bad.parent.mkdir();bad.write_text('{}')
            with self.assertRaises(ValueError):t.safe(bad,[root])
            with self.assertRaises(ValueError):t.safe(f,[root/'elsewhere'])
            self.assertEqual(t.safe(f,[root]),f)

    def test_exact_pinned_delta_rejects_old_byte_change_and_unlisted_path(self):
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();old=root/'old.py';old.write_text('old bytes')
            training={'old.py':t.sha(old)};delta={}
            for name in t.REPORT_FILES:
                if name in training:continue
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed '+name);delta[name]=t.sha(p)
            r=t.check_execution_ledger(root,training,delta)
            self.assertEqual(r,{**training,**delta})
            with self.assertRaises(ValueError):t.check_execution_ledger(root,training,{**delta,'unlisted.py':'0'*64})
            with self.assertRaises(ValueError):t.check_execution_ledger(root,training,{**delta,**training})
            old.write_text('changed bytes')
            with self.assertRaises(ValueError):t.check_execution_ledger(root,training,delta)

    def test_raw_verified_and_manifest_identity_cannot_be_replaced_by_auxiliary_json(self):
        t=tool();rid=t.RUNS[0];pin='a'*64
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();vr=root/'runs'/(rid+'-verify');logs=vr/'logs';logs.mkdir(parents=True)
            log=logs/'stdout.log';log.write_text(json.dumps({'status':'complete'}))
            with self.assertRaises(ValueError):t.guard_verifier(root,rid,pin,t.sha(log))
            raw={'status':'verified','run_id':rid,'verify_run_id':rid+'-verify','manifest_sha256':pin}
            log.write_text(json.dumps(raw));proof=t.guard_verifier(root,rid,pin,t.sha(log))
            self.assertEqual(proof['status'],'raw_stdout_verified')
            raw['manifest_sha256']='b'*64;log.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):t.guard_verifier(root,rid,pin,t.sha(log))

    def test_request_cannot_change_accepted_hits_or_seed_manifests(self):
        t=tool();self.assertTrue(hasattr(t,'ACCEPTED_STDOUTS'))
        req={'schema':'sid-structure-report-request-v1','frozen_root':str(ROOT),'source_commit':'a'*40,
             'scheduler_root':'/workspace/vkgr-infra01-root','run_ids':list(t.RUNS),
             'manifest_pins':dict(t.ACCEPTED_MANIFESTS),'stdout_pins':dict(t.ACCEPTED_STDOUTS),
             'history':{'path':'/allowed/history.json','sha256':'c'*64},'catalog':{'path':'/allowed/catalog.json','sha256':'d'*64},
             'sid_manifests':{k:{'path':'/allowed/'+k+'.json','sha256':pin} for k,pin in t.SID_PINS.items()},
             'source_delta':{'structure':{},'control':{}}}
        t.validate_request(req)
        bad={**req,'manifest_pins':{**req['manifest_pins'],t.RUNS[0]:'0'*64}}
        with self.assertRaises(ValueError):t.validate_request(bad)
        with self.assertRaises(ValueError):t.validate_request({**req,'accepted_hits':{}})
        with self.assertRaises(ValueError):t.validate_request({**req,'stdout_pins':{**req['stdout_pins'],t.RUNS[0]:'0'*64}})

class SourceBindingTests(unittest.TestCase):
    def test_original_sid_runtime_source_and_delta_are_all_bound(self):
        import copy
        t=tool();self.assertTrue(callable(getattr(t,'audit_source',None)))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'uv.lock').write_text('fixed lock')
            training={'uv.lock':t.sha(root/'uv.lock')};delta={}
            for name in t.REPORT_FILES:
                if name in training:continue
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed '+name);delta[name]=t.sha(p)
            own={'runtime':{'python':'fixture','exact_dependencies':{}},'lock_sha256':training['uv.lock']}
            binding={'source':{'git':{'git_base':t.SOURCES['structure'],'git_dirty':False},'files':training},
                     'sid':{'manifest_sha256':t.SID_PINS['structure']},'runtime':{'environment':own['runtime']}}
            self.assertEqual(t.audit_source(root,'structure',binding,delta,own),{**training,**delta})
            for key in ('sid','runtime','source'):
                bad=copy.deepcopy(binding)
                if key=='sid':bad[key]['manifest_sha256']='0'*64
                elif key=='runtime':bad[key]['environment']['python']='changed'
                else:bad[key]['git']['git_dirty']=True
                with self.assertRaises(ValueError):t.audit_source(root,'structure',bad,delta,own)

class LegacyLockTests(unittest.TestCase):
    def test_legacy_missing_lock_is_anchored_to_original_git_not_current_bytes(self):
        import subprocess
        t=tool();self.assertTrue(callable(getattr(t,'original_lock_sha',None)))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();old=root/'old.py';old.write_text('original source')
            training={'old.py':t.sha(old)};delta={}
            for name in t.REPORT_FILES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original '+name);delta[name]=t.sha(p)
            for args in (['init','-q'],['add','old.py',*t.REPORT_FILES],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']):
                subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True)
            commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();t.SOURCES['control']=commit
            expected=t.sha(root/'uv.lock');own={'runtime':{'python':'fixture','exact_dependencies':{}},'lock_sha256':expected}
            binding={'source':{'git':{'git_base':commit,'git_dirty':False},'files':training},'sid':{'manifest_sha256':t.SID_PINS['control']},'runtime':{'environment':own['runtime']}}
            self.assertEqual(t.original_lock_sha(root,'control',training),expected)
            t.audit_source(root,'control',binding,delta,own)
            (root/'uv.lock').write_text('changed lock');delta['uv.lock']=t.sha(root/'uv.lock');own['lock_sha256']=delta['uv.lock']
            self.assertEqual(t.original_lock_sha(root,'control',training),expected)
            with self.assertRaises(ValueError):t.audit_source(root,'control',binding,delta,own)

class OriginalJobBindingTests(unittest.TestCase):
    def test_report_stdout_binds_real_run_source_config_and_actual_argv(self):
        from unittest.mock import patch
        import copy
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();run=root/'runs'/t.PF_ID;run.mkdir(parents=True)
            request=root/'request.json';request.write_text('{}');pin=t.sha(request)
            raw=['--request',str(request),'--request-sha',pin,'--preflight']
            import sys
            meta={'run_id':t.PF_ID,'git':{'commit':'a'*40,'dirty':False},'cwd':str(ROOT),'config':{'path':str(request),'sha256':pin},'argv':[sys.executable,'-B',str(ROOT/'analysis/sid_structure_report_read.py'),*raw]}
            mp=run/'run.json';mp.write_text(json.dumps(meta));req={'scheduler_root':str(root),'source_commit':'a'*40}
            with patch.dict(t.os.environ,VKGR_RUN_ID=t.PF_ID):
                b=t.bind_job(req,request,pin,raw,True);self.assertEqual(b['run_manifest_sha256'],t.sha(mp));self.assertEqual(b['run_id'],t.PF_ID)
                for field in ('source','config','argv'):
                    bad=copy.deepcopy(meta)
                    if field=='source':bad['git']['commit']='b'*40
                    elif field=='config':bad['config']['sha256']='d'*64
                    else:bad['argv'][-1]='--other'
                    mp.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):t.bind_job(req,request,pin,raw,True)

class ArchivePathTests(unittest.TestCase):
    def test_lexical_evaluation_symlink_is_rejected_before_resolving_archive(self):
        t=tool();self.assertTrue(callable(getattr(t,'guard_archive_paths',None)))
        for link in (False,True):
            with self.subTest(link=link),tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();real=root/('real-evaluation' if link else 'evaluation-007');real.mkdir()
                if link:(root/'evaluation-007').symlink_to(real,target_is_directory=True)
                shard=real/'shard-000000-002520.json';shard.write_text('{}')
                evaluation=real/'manifest.json';evaluation.write_text(json.dumps({'complete':True,'completed_queries':2520,'shards':[{'path':str(shard),'sha256':t.sha(shard)}]}))
                manifest={'best':{'epoch':7},'selection':{'best_epoch':7},'epochs':[{'epoch':7,'evaluation':{'path':'evaluation-007','manifest_sha256':t.sha(evaluation)}}]}
                def reader(path,pin):
                    if link:raise AssertionError('opened symlinked evaluation')
                    self.assertEqual(t.sha(path),pin);return json.loads(Path(path).read_bytes()),pin
                if link:
                    with self.assertRaises(ValueError):t.guard_archive_paths(root/'manifest.json',manifest,reader)
                else:self.assertEqual(t.guard_archive_paths(root/'manifest.json',manifest,reader)['shards'],1)

if __name__=='__main__':unittest.main()

class FrozenSupplementTests(unittest.TestCase):
    def test_query_weighted_baseline_correction_and_no_top50_correction(self):
        t=tool();self.assertTrue(callable(getattr(t,'prefix_description',None)))
        _,m=t.frozen(ROOT)
        mapping={1:(0,0,0,0),2:(0,0,1,0),3:(1,1,1,0)}
        targets=[1,1,3,99];pred=[[1],[3],[1],[1]]
        r=t.prefix_description(targets,m.prediction_metrics(targets,pred,mapping),mapping)
        self.assertAlmostEqual(r['1']['q'],5/9)
        self.assertAlmostEqual(r['1']['C_top1'],-.5)
        self.assertAlmostEqual(r['1']['A_top1'],1/3)
        self.assertNotIn('C_top50',r['1'])
        one={1:(0,0,0,0)}
        r=t.prefix_description([1],m.prediction_metrics([1],[[1]],one),one)
        self.assertIsNone(r['2']['C_top1'])
    def test_positive_total_and_head_mid_half_rule_inclusive_boundary(self):
        t=tool();self.assertTrue(callable(getattr(t,'head_mid_check',None)))
        for deltas,ok in [({'high':0,'mid':0,'low':0,'unsupported':0,'oov':0},False),({'high':1,'mid':0,'low':1,'unsupported':0,'oov':0},True),({'high':0,'mid':0,'low':1,'unsupported':0,'oov':0},False),({'high':2,'mid':-3,'low':1,'unsupported':0,'oov':0},False),({'high':-2,'mid':1,'low':0,'unsupported':0,'oov':0},False)]:
            result=t.head_mid_check([deltas,deltas,deltas]);self.assertEqual(result['passed'],ok)
            self.assertEqual(result['delta_by_bin']['high'],3*deltas['high'])
    def test_integrity_pins_all_bin_denominators_zero_oov_and_original_fit(self):
        import copy
        t=tool();self.assertTrue(callable(getattr(t,'fit_integrity',None)))
        _,m=t.frozen(ROOT);mapping={1:(0,0,0,0),2:(0,0,0,1),3:(1,1,1,0)}
        structure=t.catalog_structure(mapping,48,m)
        audit={'items':3,'positions':[{'distinct':2,'utilization':2/48}]*3,'prefix_groups':{'1':2,'2':2,'3':2},'leaves':{'count':2,'max':2,'histogram_head':[[1,1],[2,1]],'distinct_sizes':2},'semantic_collision_items':2,'semantic_collision_groups':1,'full_sid_collisions':0}
        self.assertTrue(t.fit_integrity(structure,audit)['passed'])
        for key in ('positions','semantic_collision_items','prefix_groups'):
            bad=copy.deepcopy(audit)
            if key=='positions':bad[key][2]['distinct']=1
            elif key=='prefix_groups':bad[key]['2']=3
            else:bad[key]=1
            self.assertFalse(t.fit_integrity(structure,bad)['passed'])
        pair={'new':{'C':{'bins':{k:{'n':n,'hits':0} for k,n in t.BIN_DENOMINATORS.items()}}},'control':{'C':{'bins':{k:{'n':n,'hits':0} for k,n in t.BIN_DENOMINATORS.items()}}}}
        self.assertTrue(t.bin_integrity([pair])['passed'])
        pair['new']['C']['bins']['high']['n']-=1
        self.assertFalse(t.bin_integrity([pair])['passed'])

class SupplementAggregationTests(unittest.TestCase):
    def test_seed_means_and_anchor_ignore_prefix_values_but_require_completeness(self):
        import copy
        t=tool();_,m=t.frozen(ROOT)
        mapping={1:(0,0,0,0),2:(0,0,0,1),3:(1,1,1,0)}
        structure=t.catalog_structure(mapping,48,m)
        audit={'items':3,'positions':[{'distinct':2,'utilization':2/48}]*3,'prefix_groups':{'1':2,'2':2,'3':2},'leaves':{'count':2,'max':2,'histogram_head':[[1,1],[2,1]],'distinct_sizes':2},'semantic_collision_items':2,'semantic_collision_groups':1,'full_sid_collisions':0}
        reports={}
        for seed,value in zip(t.SEEDS,(.1,.2,.6)):
            r={role:{'C':{'bins':{k:{'n':n,'hits':0} for k,n in t.BIN_DENOMINATORS.items()}},'prefix_description':{k:{field:value for field in ('A_top1','A_top50','q','C_top1')} for k in ('1','2','3')}} for role in ('new','control')}
            r['popularity_delta_hits']={'high':2,'mid':1,'low':0,'unsupported':0,'oov':0};reports[str(seed)]=r
        out=t.supplement(reports,{'structure':structure,'control':structure},{'structure':audit,'control':audit})
        self.assertAlmostEqual(out['prefix_seed_means']['new']['2']['A_top1'],.3)
        self.assertTrue(out['anchor_rule_components_satisfied'])
        altered=copy.deepcopy(reports)
        for row in altered.values():
            row['new']['prefix_description']['2']['C_top1']=-1
            row['control']['prefix_description']['2']['C_top1']=1
        self.assertTrue(t.supplement(altered,{'structure':structure,'control':structure},{'structure':audit,'control':audit})['anchor_rule_components_satisfied'])
        altered['42']['new']['C']['bins']['oov']['hits']=1
        self.assertFalse(t.supplement(altered,{'structure':structure,'control':structure},{'structure':audit,'control':audit})['anchor_rule_components_satisfied'])
