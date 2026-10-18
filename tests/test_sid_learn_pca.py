"""PCA32 is deterministic, train-source bound and has no learned head."""
import importlib,hashlib,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

class FixedPCATests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_pca.py').is_file(),'missing fixed PCA entry')
        self.pca=importlib.import_module('vkgr.sid_learn_pca')

    def diagonal(self):
        x=np.zeros((128,64),dtype=np.float32)
        for j in range(64):x[2*j,j]=64-j;x[2*j+1,j]=-(64-j)
        return x

    def test_hand_top_32_components_projection_and_explained_variance(self):
        x=self.diagonal();state=self.pca.fit_pca(x);want=np.eye(64,dtype=np.float64)[:32]
        np.testing.assert_array_equal(state['mean'],np.zeros(64));np.testing.assert_array_equal(state['components'],want)
        projected=self.pca.project_pca(x,state);np.testing.assert_array_equal(projected,x[:,:32]);self.assertEqual(projected.dtype,np.float32)
        ratio=sum(j*j for j in range(33,65))/sum(j*j for j in range(1,65))
        self.assertAlmostEqual(float(state['explained_variance_ratio'].sum()),ratio,places=12)
        self.assertEqual(state['components_sha256'],hashlib.sha256(want.astype('<f8').tobytes()).hexdigest())
        self.assertEqual(state['seed'],42);self.assertEqual(state['n_train_observed'],128)
        shifted=x+np.arange(64,dtype=np.float32)
        shifted_state=self.pca.fit_pca(shifted)
        np.testing.assert_array_equal(shifted_state['mean'],np.arange(64,dtype=np.float64))
        np.testing.assert_array_equal(self.pca.project_pca(shifted,shifted_state),x[:,:32])

    def test_same_input_same_components_and_no_input_or_global_rng_mutation(self):
        x=np.random.default_rng(12).normal(size=(80,64)).astype('float32');before=x.copy();rng_before=np.random.get_state()
        a=self.pca.fit_pca(x);b=self.pca.fit_pca(x)
        np.testing.assert_array_equal(a['components'],b['components']);self.assertEqual(a['components_sha256'],b['components_sha256'])
        np.testing.assert_array_equal(x,before);rng_after=np.random.get_state();np.testing.assert_array_equal(rng_before[1],rng_after[1]);self.assertEqual(rng_before[2:],rng_after[2:])
        for component in a['components']:self.assertGreaterEqual(component[np.argmax(np.abs(component))],0.)

    def test_frozen_arrays_survive_nonpickle_serialization_roundtrip(self):
        import io
        x=self.diagonal();state=self.pca.fit_pca(x);buffer=io.BytesIO()
        keys=('mean','components','eigenvalues','explained_variance_ratio')
        np.savez(buffer,**{k:state[k] for k in keys});buffer.seek(0)
        with np.load(buffer,allow_pickle=False) as saved:
            restored={**state,**{k:saved[k].copy() for k in keys}}
        self.assertEqual(restored['components_sha256'],state['components_sha256'])
        np.testing.assert_array_equal(self.pca.project_pca(x,restored),self.pca.project_pca(x,state))

    def test_projection_rejects_different_source_or_changed_frozen_components(self):
        x=self.diagonal();state=self.pca.fit_pca(x);changed=x.copy();changed[0,0]+=1
        with self.assertRaises(ValueError):self.pca.project_pca(changed,state)
        state['components'][0,0]=-1
        with self.assertRaises(ValueError):self.pca.project_pca(x,state)

    def test_nonfinite_wrong_dimension_or_zero_variance_stops(self):
        for x in (np.ones((64,63),dtype='float32'),np.ones((64,64),dtype='float64'),np.ones((64,64),dtype='float32')):
            with self.assertRaises(ValueError):self.pca.fit_pca(x)
        x=self.diagonal();x[0,0]=np.nan
        with self.assertRaises(ValueError):self.pca.fit_pca(x)

if __name__=='__main__':unittest.main()
