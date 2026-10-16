"""Synthetic-only full R2 contract, mining and actor/reference training tests."""
import copy
import json
from pathlib import Path
import unittest
import torch
from vkgr import ranking_r2_config as cfg, ranking_r2_engine as engine
from vkgr import generative_g6_model as model
from vkgr import ranking_r2 as primitives
from vkgr.sid_depth import SIDTokensLD
from tests.test_generative_g6_unit import codec_of, g6_config


class Contract(unittest.TestCase):
    def test_three_frozen_configs_and_explicit_scientific_fields(self):
        for seed in (42, 2024, 2025):
            value = json.loads(Path(f'configs/ranking_r2/g-ranking-r2-s{seed}.json').read_text())
            cfg.validate(value)
            self.assertEqual(value['seed'], seed)
            self.assertEqual(value['negative_policy'], 'skip_without_resampling')
            self.assertEqual(value['dpo_min_epochs'], 1)
            for name in ('negative_policy', 'dpo_min_epochs', 'mining_admission'):
                with self.assertRaises(ValueError):
                    cfg.validate(dict(value, **{name: None}))
            with self.assertRaises(ValueError):
                cfg.validate(dict(value, dpo_learning_rate=.0003))
            self.assertEqual(value['mining_samples'], 20000)

    def test_sft_only_changes_schema_task_from_original_same_seed(self):
        from vkgr import ranking_r2_sft as sft
        value = cfg.default_config(2024)
        actual = cfg.sft_config(value)
        sft.validate_config(actual, False)
        base = json.loads(Path('configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
        self.assertEqual({k: v for k, v in actual.items() if k not in ('schema', 'task_id', 'seed')}, {k: v for k, v in base.items() if k not in ('schema', 'task_id', 'seed')})
        self.assertEqual(actual['seed'], 2024)


class Engine(unittest.TestCase):
    def setUp(self):
        self.ids, self.codec = codec_of(3, n=8)
        self.tokens = SIDTokensLD(self.codec, 50)
        self.config = cfg.default_config(42)
        self.small = g6_config(sid='SID_V2_REPLAY_BHKM_B16D3')
        torch.manual_seed(42)
        self.actor = model.make_model_arch_d(self.codec, self.small, device='cpu')
        self.reference = primitives.frozen_reference(self.actor)

    def test_one_dpo_step_updates_actor_only_and_reference_stays_eval(self):
        rows = [{'history': [int(self.ids[0])], 'target': int(self.ids[1]), 'negative': int(self.ids[2])},
                {'history': [int(self.ids[1])], 'target': int(self.ids[2]), 'negative': int(self.ids[3])}]
        prior_ref = {k: v.clone() for k, v in self.reference.state_dict().items()}
        prior_actor = {k: v.clone() for k, v in self.actor.state_dict().items()}
        optimizer = engine.make_optimizer(self.actor, self.config)
        self.assertEqual(optimizer.param_groups[0]['lr'], .00003)
        result = engine.train_step(self.actor, self.reference, optimizer, rows, self.tokens, microbatch=1, device='cpu')
        self.assertEqual(result.examples, 2)
        self.assertTrue(any(not torch.equal(v, self.actor.state_dict()[k]) for k, v in prior_actor.items()))
        self.assertTrue(all(torch.equal(v, self.reference.state_dict()[k]) for k, v in prior_ref.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.reference.parameters()))
        self.assertFalse(self.reference.training)
        self.assertTrue(set(id(p) for group in optimizer.param_groups for p in group['params']).isdisjoint(id(p) for p in self.reference.parameters()))

    def test_mining_skips_without_replacing_sample_and_uses_full_likes(self):
        from unittest.mock import patch
        ids = [int(i) for i in self.ids]
        rows = [{'example_id': 0, 'history': [ids[0]], 'target': ids[1], 'past_likes': ids[2:]},
                {'example_id': 1, 'history': [ids[0]], 'target': ids[1], 'past_likes': [ids[0]]}]
        # First candidate is legal for first row even though it is observed history;
        # second row has no legal candidate and must not be replaced or resampled.
        beams = [([ids[1], ids[2], ids[0]], {}), ([ids[1], ids[0]], {})]
        with patch.object(engine, 'decode_histories_batched_d', side_effect=[[beams[0]], [beams[1]]]):
            mined = engine.mine_rows(self.actor, self.tokens, rows, [0, 1], engineering=True)
        self.assertEqual([x['example_index'] for x in mined['records']], [0, 1])
        self.assertEqual(mined['records'][0]['negative'], ids[0])
        self.assertIsNone(mined['records'][1]['negative'])
        self.assertEqual(mined['skipped'], 1)
        self.assertEqual(mined['skip_fraction'], .5)
        engine.validate_mining(mined, rows, self.tokens, engineering=True)
        bad = copy.deepcopy(mined);bad['records'][0]['negative'] = ids[2]
        with self.assertRaises(ValueError):
            engine.validate_mining(bad, rows, self.tokens, engineering=True)

    def test_epoch_zero_is_excluded_from_dpo_selection(self):
        stop = engine.dpo_stop(self.config)
        self.assertIsNone(stop.best_epoch)
        stop.observe(1, .1, .1)
        self.assertEqual(stop.best_epoch, 1)
        self.assertEqual((stop.min_epochs, stop.max_epochs, stop.patience), (1, 5, 2))


class FullSynthetic(unittest.TestCase):
    def test_two_stages_own_checkpoints_epoch0_report_and_final_verify(self):
        import tempfile
        from types import MappingProxyType, SimpleNamespace
        from unittest.mock import patch
        from vkgr import ranking_r2_run as run, ranking_r2_sft as sft
        from vkgr import generative_experiment as ge, generative_evaluation as ev
        from vkgr.generative_data_v2 import ListExamples
        from vkgr.data_audit import file_sha256
        ids, codec = codec_of(3, n=8)
        ids = [int(i) for i in ids]
        rows = [{'example_id': i, 'history': [ids[i]], 'target': ids[i+1], 'past_likes': [ids[i]]} for i in range(3)]
        queries = [{'user_id': 1, 'history': [ids[0]], 'history_positions': [[0, 0]], 'target': ids[1], 'target_position': [25, 0], 'oov': False}]
        stats = ev.SliceStatistics((1,1),MappingProxyType({i:1 for i in ids}),MappingProxyType({1:2}),'a'*64,'b'*64,'c'*64)
        context = ge.RunContext(ge._CONTEXT_KEY,examples=ListExamples(rows),codec=codec,queries=queries,
            binding={'queries_sha256':ge._digest(queries),'catalog_sha256':ge._digest(ids),'synthetic':True,'engineering':True,'b1':{}},stats=stats).check()
        info={'sid_version':'SID_V2_REPLAY_BHKM_B16D3','manifest_sha256':'d'*64,'branching':codec.catalog.branching,'depth':3,
              'suffix_vocabulary':codec.catalog.bases[-1],'vocab_size':codec.vocab_size,'kind':'bhkm','engineering':True}
        sid=SimpleNamespace(info=info,manifest={'content_feature_sha256':'e'*64})
        config=cfg.default_config(42);config['sft']=g6_config(sid=info['sid_version'])
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp, patch.object(sft,'_contexts',return_value=(context,True,sid)):
            temp=Path(temp)
            control=temp/'a1-control.json'
            control.write_text(json.dumps({'schema':'g-g6-train-manifest-v1','kind':'g_g6_train','status':'complete','config':config['sft'],
                'best':{'epoch':1},'epochs':[{'epoch':1,'metrics':{'metrics':{'all_eligible':{'recall@50':0.,'ndcg@50':0.}}}}]}))
            common=(root,Path('synthetic'),'1'*64,Path('synthetic'),'2'*64,Path('synthetic'),'3'*64,Path('synthetic'),'4'*64)
            value=run.run(*common,config,temp/'run',control_manifest=control,control_sha=file_sha256(control),engineering_indices=[0,1,2])
            self.assertEqual(value['status'],'complete')
            self.assertGreaterEqual(value['dpo']['best']['epoch'],1)
            self.assertNotIn(0,[record['epoch'] for record in value['dpo']['epochs']])
            self.assertIn('control_delta',value['epoch0'])
            self.assertEqual(value['epoch0']['control_delta']['recall@50'],1.)
            result=run.verify(*common,temp/'run/manifest.json',value['manifest_sha256'],temp/'verify',control_manifest=control,control_sha=file_sha256(control))
            self.assertEqual(result['status'],'verified')
            self.assertEqual(sorted(p.name for p in (temp/'run/dpo').glob('*.pt')),['best.pt','last.pt'])
            mining=temp/'run/mining/records.json'
            mining.write_text('{}')
            with self.assertRaises(ValueError):
                run.verify(*common,temp/'run/manifest.json',value['manifest_sha256'],temp/'bad-verify',control_manifest=control,control_sha=file_sha256(control))


class FailClosed(unittest.TestCase):
    def test_cross_seed_sft_metadata_rejected_before_checkpoint_io(self):
        from vkgr import ranking_r2_run as run, ranking_r2_config as contract
        from unittest.mock import patch
        value = {'config': contract.sft_config(contract.default_config(2024))}
        with patch.object(run.sft, '_load_best') as load:
            with self.assertRaises(ValueError):
                run._own_sft(None, Path('never-open'), value, None, {}, contract.default_config(42))
            load.assert_not_called()

    def test_dpo_cuda_proof_cannot_omit_reference_or_actor_only_optimizer(self):
        from tests.test_ranking_r2_ops import module
        from unittest.mock import patch
        m = module()
        req = {'preflight_receipt': 'p', 'cuda_receipt': 'c', 'source_commit': 'a'*40}
        pf = {'status': 'preflight_ok', 'source_commit': 'a'*40}
        proof = {'status': 'cuda_engineering_verified', 'source_commit': 'a'*40,
                 'reload_decode_equal': True, 'synthetic_grad_steps': 2, 'measured_mib': 6000, 'declared_vram_mib': 6912}
        with patch.object(m, 'identity'), patch.object(m, 'load', side_effect=[pf, proof]), patch.object(m, 'launcher_prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'actor/reference/optimizer'):
                m.dry_run(req)
            prepare.assert_not_called()


class ControlMetadata(unittest.TestCase):
    def test_original_control_comparison_reads_metrics_only_never_checkpoint(self):
        import tempfile
        from unittest.mock import patch
        from vkgr import ranking_r2_run as run
        from vkgr.data_audit import file_sha256
        config=cfg.default_config(42)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            path=Path(temp)/'control.json'
            path.write_text(json.dumps({'schema':'g-g6-train-manifest-v1','kind':'g_g6_train','status':'complete','config':config['sft'],
                'best':{'epoch':1,'path':'do-not-open.pt'},'epochs':[{'epoch':1,'metrics':{'metrics':{'all_eligible':{'recall@50':.1,'ndcg@50':.05}}}}]}))
            with patch.object(torch,'load',side_effect=AssertionError('old checkpoint forbidden')) as load:
                self.assertEqual(run._control(path,file_sha256(path),config)['recall@50'],.1)
                load.assert_not_called()


class FrozenTypes(unittest.TestCase):
    def test_admission_boolean_cannot_be_replaced_with_numeric_truthiness(self):
        value=cfg.default_config(42)
        value['mining_admission']['within_four_hours']=1
        with self.assertRaises(ValueError):
            cfg.validate(value)


class EarlyCrossSeedGate(unittest.TestCase):
    def test_final_verifier_rejects_cross_seed_sft_before_any_stage_checkpoint_verify(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from vkgr import ranking_r2_run as run
        from vkgr.data_audit import file_sha256
        config=cfg.default_config(42)
        value={'schema':run.SCHEMA,'kind':run.KIND,'status':'complete','config':config,
               'sft':{'path':'sft/manifest.json','sha256':'a'*64},'mining':{'path':'mining/records.json'}}
        wrong={'config':cfg.sft_config(cfg.default_config(2024))}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            temp=Path(temp);manifest=temp/'manifest.json';manifest.write_text(json.dumps(value))
            with patch.object(run.sft,'_contexts',return_value=(SimpleNamespace(),True,None)), patch.object(run,'_control',return_value={}), \
                 patch.object(run.sft,'_runtime',return_value={'device':'cpu'}), patch.object(run.sft,'_read_run',return_value=wrong), \
                 patch.object(run.sft,'verify',side_effect=AssertionError('cross-seed checkpoint verifier must not start')) as stage_verify:
                with self.assertRaises(ValueError):
                    run.verify(Path('.'),Path('p'),'1'*64,Path('h'),'2'*64,Path('s'),'3'*64,Path('c'),'4'*64,
                               manifest,file_sha256(manifest),temp/'verify',control_manifest=Path('control'),control_sha='5'*64)
                stage_verify.assert_not_called()


class CudaConfigBinding(unittest.TestCase):
    def test_cuda_reference_proof_config_must_match_seed42_preflight_before_prepare(self):
        from tests.test_ranking_r2_ops import module
        from unittest.mock import patch
        m=module();req={'preflight_receipt':'p','cuda_receipt':'c','source_commit':'a'*40}
        common={'source_commit':'a'*40,'data_binding_sha256':'b'*64,'sid_manifest_sha256':'c'*64,'runtime':{},'model_shape':{}}
        pf=dict(common,status='preflight_ok',entries=[{'config_sha256':'a'*64}]*3)
        proof=dict(common,status='cuda_engineering_verified',reference_unchanged=True,optimizer_actor_only=True,synthetic_sft_steps=2,
                   actor_parameter_count=10,reference_parameter_count=10,reload_decode_equal=True,synthetic_grad_steps=2,
                   config_sha256='b'*64,measured_mib=6000.,declared_vram_mib=6912,
                   memory_measurement={'pid':123,'context_process_mib':1.,'context_reserved_mib':1.,
                    'samples':[{'stage':'dpo','own_process_mib':6000.,'reserved_mib':6000.,'max_reserved_mib':6000.,'max_allocated_mib':5900.}]})
        with patch.object(m,'identity'),patch.object(m,'load',side_effect=[pf,proof]),patch.object(m,'derive_job',side_effect=AssertionError('must reject proof before job preparation')) as derive:
            with self.assertRaisesRegex(ValueError,'config'):
                m.dry_run(req)
            derive.assert_not_called()
