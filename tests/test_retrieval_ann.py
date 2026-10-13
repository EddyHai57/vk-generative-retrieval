import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from vkgr.retrieval_models import TwoTower
from vkgr.retrieval_ann import export_vectors, build_index, audit_index, read_export, DependencyUnavailable, file_hash, select_audit_setting, validate_audit_queries

class ExportTests(unittest.TestCase):
    def fixture(self, root):
        content=torch.zeros((4,64),dtype=torch.float32)
        model=TwoTower('id_mean',content,[1,2,3])
        with torch.no_grad():
            model.embedding.weight.zero_()
            model.embedding.weight[1,0]=1
            model.embedding.weight[2,1]=1
            model.embedding.weight[3,0]=-1
        ck=root/'checkpoint.pt';torch.save({'schema_version':1,'binding':{'original':'producer'},'model':model.state_dict()},ck)
        binding={k:'a'*64 for k in ('source_sha256','catalog_sha256','idmap_sha256','query_sha256')}
        binding['checkpoint_sha256']=file_hash(ck)
        ctx={'checkpoint':str(ck),'checkpoint_binding':{'original':'producer'},'config':{'architecture':'id_mean','seed':42,'batch_size':512},'content':content,'supported_indices':[1,2,3],'catalog':[10,20,30],'idmap':{10:1,20:2,30:3},'queries':[{'query_id':7,'history':[10],'target':10,'oov':False},{'query_id':8,'history':[20],'target':20,'oov':False}]}
        return ctx,binding

    def test_actual_checkpoint_export_roundtrip_and_pin_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,binding=self.fixture(root)
            manifest=export_vectors(ctx,root/'export',binding)
            ids,users,items=read_export(root/'export'/'manifest.json',file_hash(root/'export'/'manifest.json'),binding)
            self.assertEqual(ids.tolist(),[10,20,30])
            self.assertEqual(users[0,0],1.)
            self.assertEqual(items[2,0],-1.)
            wrong={**binding,'source_sha256':'b'*64}
            with self.assertRaises(ValueError):read_export(root/'export'/'manifest.json',file_hash(root/'export'/'manifest.json'),wrong)
            with self.assertRaises(FileExistsError):export_vectors(ctx,root/'export',binding)
            ctx['checkpoint_binding']={'original':'other'}
            with self.assertRaises(ValueError):export_vectors(ctx,root/'bad',binding)
            self.assertFalse((root/'bad').exists())

    def test_nonfinite_or_mapping_mismatch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,binding=self.fixture(root);ctx['idmap'][10]=2
            with self.assertRaises(ValueError):export_vectors(ctx,root/'bad',binding)
            ctx,binding=self.fixture(root);ctx['content'][0,0]=float('nan')
            with self.assertRaises(ValueError):export_vectors(ctx,root/'bad',binding)

    def test_audit_targets_and_members_bound_to_exported_query_records(self):
        exported={'query_ids':[7,8],'query_records':[{'query_id':7,'user_id':7,'history':[10],'target':10,'oov':False},{'query_id':8,'user_id':8,'history':[20],'target':20,'oov':False}]}
        queries={'synthetic':True,'query_ids':[7,8],'targets':[10,20],'audit_sha256':'a'*64}
        self.assertEqual(validate_audit_queries(queries,exported,{'query_sha256':'a'*64}),[7,8])
        queries['targets']=[30,30]
        with self.assertRaises(ValueError):validate_audit_queries(queries,exported,{'query_sha256':'a'*64})
        queries['targets']=[10,20];queries['query_ids']=[8,7]
        with self.assertRaises(ValueError):validate_audit_queries(queries,exported,{'query_sha256':'a'*64})

    def test_audit_gate_uses_smallest_passing_and_retains_exact_if_none(self):
        self.assertEqual(select_audit_setting([{'ef_search':64,'overlap@50':.97},{'ef_search':128,'overlap@50':.98},{'ef_search':256,'overlap@50':1.}]),128)
        self.assertIsNone(select_audit_setting([{'ef_search':64,'overlap@50':.97},{'ef_search':128,'overlap@50':.97},{'ef_search':256,'overlap@50':.979}]))
        with self.assertRaises(ValueError):select_audit_setting([{'ef_search':512,'overlap@50':1.}])

    @unittest.skipIf(importlib.util.find_spec('faiss') is not None,'requires missing dependency')
    def test_missing_faiss_fails_explicitly_without_index_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,binding=self.fixture(root);export_vectors(ctx,root/'export',binding)
            with self.assertRaises(DependencyUnavailable):build_index({'path':str(root/'export'/'manifest.json'),'sha256':file_hash(root/'export'/'manifest.json')},root/'index',binding)
            self.assertFalse((root/'index').exists())

    @unittest.skipUnless(importlib.util.find_spec('faiss') is not None,'real existing FAISS unavailable; pending server gate')
    def test_real_hnsw_oracle_mapping_audit_smallest_ef(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx,binding=self.fixture(root);export_vectors(ctx,root/'export',binding)
            build_index({'path':str(root/'export'/'manifest.json'),'sha256':file_hash(root/'export'/'manifest.json')},root/'index',binding)
            queries={'vectors':np.eye(2,64,dtype=np.float32),'query_ids':[7,8],'targets':[10,20],'audit_sha256':binding['query_sha256'],'synthetic':True}
            out=audit_index({'path':str(root/'index'/'manifest.json'),'sha256':file_hash(root/'index'/'manifest.json')},queries,[[10,20,30],[20,10,30]],root/'audit',binding)
            self.assertEqual(out['selected_ef_search'],64)
            self.assertTrue(out['gate_passed'])
            queries['query_ids']=[8,7]
            with self.assertRaises(ValueError):audit_index({'path':str(root/'index'/'manifest.json'),'sha256':file_hash(root/'index'/'manifest.json')},queries,[[10,20,30],[20,10,30]],root/'wrong',binding)

# All trusted parents and dependency bytes below are synthetic. Tests patch only
# the immutable parent trust anchors / observed runtime; production pins stay fixed.
def release_fixture(root):
    import shutil
    from vkgr.retrieval_ann import canonical, file_hash
    repo=Path(__file__).resolve().parents[1]
    names=['.python-version','pyproject.toml','uv.lock','configs/traditional.yaml']
    names += [p.relative_to(repo).as_posix() for p in sorted((repo/'src/vkgr').glob('*.py'))]
    names += ['tests/test_baselines_faiss.py','tests/test_retrieval_ann.py','tests/test_retrieval_infer_run.py']
    base={}
    for name in names+['tests/test_retrieval_fusion.py','tests/test_p4d_v2_verifier_compat.py']:
        out=root/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,out)
        base[name]={'bytes':out.stat().st_size,'sha256':file_hash(out)}
    control=root/'provenance';control.mkdir()
    source={'files':{name:base[name]['sha256'] for name in (
        'src/vkgr/retrieval_ann.py','src/vkgr/retrieval_fusion.py','src/vkgr/retrieval_infer_run.py',
        'tests/test_retrieval_ann.py','tests/test_retrieval_fusion.py','tests/test_retrieval_infer_run.py','tests/test_p4d_v2_verifier_compat.py')},
        'scientific_source_sha256':'573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508',
        'scientific_plan_sha256':'8d8e2a657824743f02cb9f7360dfeda9b120b7b26f81a9deec6ac49040257f3d'}
    src=control/'source.json';src.write_bytes(canonical(source))
    parents={'source':source,'acceptance':{'manifest_sha256':file_hash(src),'status':'accepted_local_real_faiss_and_real_evaluation_pending'},
             'transport':{'members':{n:base[n] for n in names}}}
    refs={}
    for role,value in parents.items():
        out=control/(role+'.json');out.write_bytes(canonical(value))
        refs[role]={'path':out.relative_to(root).as_posix(),'sha256':file_hash(out)}
    (root/'test-parent-trust.json').write_bytes(canonical({role:ref['sha256'] for role,ref in refs.items()}))
    target=root/'dependency-target';target.mkdir()
    dep_names=['packaging/__init__.py','faiss/__init__.py','packaging-26.3.dist-info/METADATA',
               'faiss_cpu-1.15.1.dist-info/METADATA','faiss_cpu.libs/mock.so']
    dep_names += [f'faiss/mock_{i}.py' for i in range(63)]
    members={}
    for name in dep_names:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'mock')
        members[name]={'bytes':4,'sha256':file_hash(p)}
    runtime={'python':'3.11.16','executable_sha256':'a'*64,'dependencies':{}}
    for name,ver in [('numpy','1.26.4'),('torch','2.11.0+cu128'),('pyarrow','19.0.1'),('PyYAML','6.0.2'),('packaging','26.3'),('faiss-cpu','1.15.1')]:
        runtime['dependencies'][name]={'version':ver,'module_version':ver,'origin_sha256':'b'*64,'metadata_sha256':'c'*64,
            'origin':str(target/('faiss' if name=='faiss-cpu' else 'packaging')/'__init__.py') if name in ('packaging','faiss-cpu') else str(root/'fixed-env'/name/'__init__.py'),
            'distribution_root':str(target) if name in ('packaging','faiss-cpu') else str(root/'fixed-env')}
    release={'schema_version':1,'kind':'p4f_consumer_release','parents':refs,
             'declared_build_git':{'commit':'0'*40,'dirty':True},'members':base,
             'dependency_bundle':{'root':str(target),'fixed_root':str(root/'fixed-env'),
                 'wheels':{'packaging':'d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c','faiss-cpu':'f2c31b7f2f6647eb76829a5cfe3c398fb9346df9f26b1d4db35269c91eb58c33'},'members':members},
             'runtime':runtime}
    p=root/'release.json';p.write_bytes(canonical(release))
    return release,{'path':str(p),'sha256':file_hash(p)}


def synthetic_parent_pins(release):
    return {role:ref['sha256'] for role,ref in release['parents'].items()}


class ConsumerReleaseTests(unittest.TestCase):
    def identity(self, root, ref, runtime):
        from unittest.mock import patch
        from vkgr.retrieval_ann import consumer_identity
        self.assertIn('consumer_release',__import__('inspect').signature(consumer_identity).parameters)
        with patch('vkgr.retrieval_ann.__file__',str(root/'src/vkgr/retrieval_ann.py')),patch('vkgr.manifest.ROOT',root),patch('vkgr.retrieval_ann._RELEASE_PARENTS',__import__('json').loads((root/'test-parent-trust.json').read_text())),patch('vkgr.retrieval_ann._runtime_identity',return_value=runtime,create=True):
            return consumer_identity(consumer_release=ref)

    def test_no_git_default_still_refuses(self):
        from unittest.mock import patch
        from vkgr.retrieval_ann import consumer_identity
        with patch('vkgr.manifest.git_context',side_effect=RuntimeError('no Git')):
            with self.assertRaisesRegex(RuntimeError,'no Git'):consumer_identity()

    def test_explicit_release_no_git_and_declared_identity_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root)
            out=self.identity(root,ref,r['runtime'])
            self.assertEqual(out['identity_kind'],'verified_deployment_release')
            self.assertIsNone(out['actual_git_commit']);self.assertIsNone(out['actual_git_dirty'])
            self.assertEqual(out['declared_build_git'],r['declared_build_git'])
            self.assertEqual(out['release_sha256'],ref['sha256'])
            self.assertNotIn('git_commit',out)

    def test_release_missing_hash_schema_parent_and_member_drift(self):
        import copy
        from vkgr.retrieval_ann import canonical,file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root);p=root/'release.json'
            for bad in ({'path':str(p)}, {'path':str(p),'sha256':'0'*64}):
                with self.assertRaises((ValueError,OSError)):self.identity(root,bad,r['runtime'])
            for mutate in (lambda v:v.update(extra=True),lambda v:v.update(schema_version=True),
                           lambda v:v['parents']['acceptance'].update(sha256='0'*64),
                           lambda v:v['members'].pop('uv.lock'),
                           lambda v:v['members'].update({'../outside':{'bytes':0,'sha256':'0'*64}}),
                           lambda v:v['members']['src/vkgr/retrieval.py'].update(sha256='0'*64)):
                v=copy.deepcopy(r);mutate(v);p.write_bytes(canonical(v));bad={'path':str(p),'sha256':file_hash(p)}
                with self.assertRaises((ValueError,OSError)):self.identity(root,bad,r['runtime'])
            p.write_bytes(canonical(r));ref['sha256']=file_hash(p)
            for name in ('uv.lock','configs/traditional.yaml','src/vkgr/retrieval_models.py'):
                orig=(root/name).read_bytes();(root/name).write_bytes(orig+b'\n')
                with self.assertRaises(ValueError):self.identity(root,ref,r['runtime'])
                (root/name).write_bytes(orig)
            (root/'src/vkgr/extra.py').write_text('pass')
            with self.assertRaises(ValueError):self.identity(root,ref,r['runtime'])

    def test_symlink_canonical_json_and_dependency_drift_reject(self):
        import copy
        from vkgr.retrieval_ann import canonical,file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root);p=root/'release.json'
            p.write_bytes(canonical(r)+b'\n');ref['sha256']=file_hash(p)
            with self.assertRaises(ValueError):self.identity(root,ref,r['runtime'])
            p.write_bytes(canonical(r));ref['sha256']=file_hash(p)
            runtime=copy.deepcopy(r['runtime']);runtime['dependencies']['numpy']['version']='2.0'
            with self.assertRaises(ValueError):self.identity(root,ref,runtime)
            dependency=root/'dependency-target/faiss/__init__.py';dependency.write_bytes(b'drift')
            with self.assertRaises(ValueError):self.identity(root,ref,r['runtime'])
            dependency.write_bytes(b'mock')
            member=root/'uv.lock';orig=member.read_bytes();member.unlink();(root/'replacement').write_bytes(orig);member.symlink_to(root/'replacement')
            with self.assertRaises(ValueError):self.identity(root,ref,r['runtime'])

class ReleaseArtifactBindingTests(unittest.TestCase):
    def test_mock_index_audit_require_same_release_before_output(self):
        import contextlib, types, copy
        from unittest.mock import patch
        from vkgr import retrieval_ann as ann
        self.assertIn('consumer_release',__import__('inspect').signature(ann.audit_index).parameters)
        class Index:
            def __init__(self,dimension,*args):self.d=dimension;self.hnsw=types.SimpleNamespace();self.ntotal=0
            def add(self,items):self.items=items;self.ntotal=len(items)
            def search(self,users,k):
                scores=users@self.items.T;positions=np.argsort(-scores,axis=1,kind='stable')[:,:k]
                return np.take_along_axis(scores,positions,axis=1),positions
        indices={}
        def write(index,path):Path(path).write_bytes(b'mock-index');indices[path]=index
        fake=types.SimpleNamespace(IndexHNSWFlat=Index,METRIC_INNER_PRODUCT=0,omp_set_num_threads=lambda n:None,write_index=write,read_index=lambda path:indices[path])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();code=root/'code';code.mkdir();r,ref=release_fixture(code)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(ann,'__file__',str(code/'src/vkgr/retrieval_ann.py')))
                stack.enter_context(patch('vkgr.manifest.ROOT',code))
                stack.enter_context(patch.object(ann,'_RELEASE_PARENTS',synthetic_parent_pins(r)))
                stack.enter_context(patch.object(ann,'_runtime_identity',return_value=r['runtime']))
                stack.enter_context(patch.object(ann,'_faiss',return_value=fake))
                ctx,binding=ExportTests().fixture(root)
                ann.export_vectors(ctx,root/'export',binding,consumer_release=ref)
                exp=root/'export/manifest.json';manifest=ann.read_json(exp);original=exp.read_bytes()
                manifest['consumer']['release_sha256']='0'*64;exp.write_bytes(ann.canonical(manifest))
                with self.assertRaises(ValueError):ann.build_index({'path':str(exp),'sha256':ann.file_hash(exp)},root/'bad-index',binding,consumer_release=ref)
                self.assertFalse((root/'bad-index').exists());exp.write_bytes(original)
                ann.build_index({'path':str(exp),'sha256':ann.file_hash(exp)},root/'index',binding,consumer_release=ref)
                idx=root/'index/manifest.json';original=idx.read_bytes();m=ann.read_json(idx)
                q={'vectors':np.eye(2,64,dtype=np.float32),'query_ids':[7,8],'targets':[10,20],'audit_sha256':binding['query_sha256'],'synthetic':True}
                m['consumer']['release_sha256']='0'*64;idx.write_bytes(ann.canonical(m))
                with self.assertRaises(ValueError):ann.audit_index({'path':str(idx),'sha256':ann.file_hash(idx)},q,[[10,20,30],[20,10,30]],root/'bad-audit',binding,consumer_release=ref)
                self.assertFalse((root/'bad-audit').exists());idx.write_bytes(original)
                out=ann.audit_index({'path':str(idx),'sha256':ann.file_hash(idx)},q,[[10,20,30],[20,10,30]],root/'audit',binding,consumer_release=ref)
                self.assertEqual(out['selected_ef_search'],64)
                self.assertEqual(out['consumer']['release_sha256'],ref['sha256'])
                with patch('vkgr.manifest.git_context',return_value=('0'*40,False)):
                    with self.assertRaises(ValueError):ann.build_index({'path':str(exp),'sha256':ann.file_hash(exp)},root/'native-mixed',binding)

class SourceClosureReviewTests(unittest.TestCase):
    def test_unlisted_bytecode_and_native_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root)
            for name in ('unrecorded.pyc','unrecorded.pyw','unrecorded.data'):
                p=root/'src/vkgr'/name;p.write_bytes(b'executable-or-sidecar')
                with self.assertRaises(ValueError):ConsumerReleaseTests().identity(root,ref,r['runtime'])
                p.unlink()

class RuntimeObservationTests(unittest.TestCase):
    def test_actual_runtime_observer_binds_module_and_distribution_bytes(self):
        import types, importlib.metadata
        from unittest.mock import patch
        from vkgr.retrieval_ann import _runtime_identity, file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();modules={};distributions={}
            for name,module in [('numpy','numpy'),('torch','torch'),('pyarrow','pyarrow'),('PyYAML','yaml'),('packaging','packaging'),('faiss-cpu','faiss')]:
                origin=root/module/'__init__.py';origin.parent.mkdir();origin.write_bytes(b'mock-module')
                meta=root/(name+'.dist-info')/'METADATA';meta.parent.mkdir();meta.write_bytes(b'mock-metadata')
                modules[module]=types.SimpleNamespace(__file__=str(origin),__version__='fixture-version')
                distributions[name]=types.SimpleNamespace(version='fixture-version',files=[meta.relative_to(root)],locate_file=lambda name,base=root:base/name)
            with patch.object(importlib.metadata,'distribution',side_effect=lambda n:distributions[n]),patch.object(importlib,'import_module',side_effect=lambda n:modules[n]):
                actual=_runtime_identity()
                self.assertEqual(actual['dependencies']['faiss-cpu']['origin_sha256'],file_hash(root/'faiss/__init__.py'))
                self.assertEqual(actual['dependencies']['numpy']['metadata_sha256'],file_hash(root/'numpy.dist-info/METADATA'))
                modules['numpy'].__file__=__file__
                with self.assertRaises(ValueError):_runtime_identity()
                modules['numpy'].__file__=str(root/'numpy/__init__.py')
                distributions['numpy'].files=[]
                with self.assertRaises(ValueError):_runtime_identity()

    def test_dependency_schema_and_canonical_roots_fail_closed(self):
        from vkgr.retrieval_ann import canonical, file_hash
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root);p=root/'release.json'
            mutations=(lambda v:v['dependency_bundle'].update(root=1),
                       lambda v:v['dependency_bundle'].update(root=str(root/'dependency-target/../dependency-target')),
                       lambda v:v['dependency_bundle']['members'].__setitem__('7',v['dependency_bundle']['members'].pop('faiss/mock_0.py')),
                       lambda v:v['runtime']['dependencies']['numpy'].update(origin=1))
            for mutate in mutations:
                v=copy.deepcopy(r);mutate(v);p.write_bytes(canonical(v));ref['sha256']=file_hash(p)
                with self.assertRaises(ValueError):ConsumerReleaseTests().identity(root,ref,r['runtime'])

class ProtectedReleaseTests(unittest.TestCase):
    def test_rehashed_protected_source_and_parent_still_refuse(self):
        from vkgr.retrieval_ann import canonical,file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();r,ref=release_fixture(root)
            path=root/'src/vkgr/retrieval_models.py';original=path.read_bytes();path.write_bytes(original+b'\n')
            r['members']['src/vkgr/retrieval_models.py']={'bytes':path.stat().st_size,'sha256':file_hash(path)}
            (root/'release.json').write_bytes(canonical(r));ref['sha256']=file_hash(root/'release.json')
            with self.assertRaisesRegex(ValueError,'protected'):ConsumerReleaseTests().identity(root,ref,r['runtime'])
            path.write_bytes(original);r['members']['src/vkgr/retrieval_models.py']={'bytes':len(original),'sha256':file_hash(path)}
            parent=root/'provenance/source.json';parent.write_bytes(parent.read_bytes()+b'\n')
            r['parents']['source']['sha256']=file_hash(parent)
            (root/'release.json').write_bytes(canonical(r));ref['sha256']=file_hash(root/'release.json')
            with self.assertRaisesRegex(ValueError,'parent'):ConsumerReleaseTests().identity(root,ref,r['runtime'])
