"""Frozen train-observed C1-prime PCA32 target; no file IO or learned head.

Full covariance eigh, descending eigenvalues, largest-magnitude coordinate of
an eigenvector positive. No RNG is consumed; seed42 is recorded by the recipe.
Targets remain in memory and are never exported by this module.
"""
import hashlib
import numpy as np
from .sid_tok_audit import feature_digest
SCHEMA='sid-learn-pca32-v1'
DIMENSION=64
COMPONENTS=32

def _sha64(value):
    return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()

def _input(vectors):
    if not isinstance(vectors,np.ndarray) or vectors.dtype!=np.float32 or vectors.ndim!=2 or vectors.shape[1]!=DIMENSION or len(vectors)<2 or not np.isfinite(vectors).all():
        raise ValueError('STOP: finite float32 train vectors [N,64] required')
    return vectors

def fit_pca(vectors):
    vectors=_input(vectors)
    values=vectors.astype(np.float64);mean=values.mean(axis=0,dtype=np.float64)
    centered=values-mean
    covariance=(centered.T@centered)/(len(values)-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    total=float(eigenvalues.sum())
    if not np.isfinite(total) or total<=0:
        raise ValueError('STOP: PCA variance is undefined')
    order=np.argsort(eigenvalues,kind='stable')[::-1][:COMPONENTS]
    components=np.ascontiguousarray(eigenvectors[:,order].T)
    pivots=np.argmax(np.abs(components),axis=1)
    signs=np.where(components[np.arange(COMPONENTS),pivots]<0,-1.,1.)
    components*=signs[:,None]
    selected=np.ascontiguousarray(eigenvalues[order])
    ratio=selected/total
    if not all(np.isfinite(x).all() for x in (mean,components,selected,ratio)):
        raise ValueError('STOP: PCA produced NaN or Inf')
    return {'schema':SCHEMA,'seed':42,'n_train_observed':len(vectors),
            'source_array_sha256':feature_digest(vectors),'mean':mean,'components':components,
            'eigenvalues':selected,'explained_variance_ratio':ratio,
            'mean_sha256':_sha64(mean),'components_sha256':_sha64(components)}

def project_pca(vectors,state):
    vectors=_input(vectors)
    if type(state) is not dict or state.get('schema')!=SCHEMA or state.get('seed')!=42 or state.get('n_train_observed')!=len(vectors) or state.get('source_array_sha256')!=feature_digest(vectors):
        raise ValueError('frozen PCA training source differs')
    mean,components=state['mean'],state['components']
    if not isinstance(mean,np.ndarray) or not isinstance(components,np.ndarray) or mean.dtype!=np.float64 or components.dtype!=np.float64 or mean.shape!=(DIMENSION,) or components.shape!=(COMPONENTS,DIMENSION) or not np.isfinite(mean).all() or not np.isfinite(components).all() or _sha64(mean)!=state['mean_sha256'] or _sha64(components)!=state['components_sha256']:
        raise ValueError('frozen PCA component/mean bytes differ')
    target=np.ascontiguousarray((vectors.astype(np.float64)-mean)@components.T,dtype=np.float32)
    if not np.isfinite(target).all():raise ValueError('STOP: PCA target holds NaN or Inf')
    return target
