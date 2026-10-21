"""Real-shaped UR001 physical manifests; metadata only, no dataset access."""
import copy

import pytest


def fixture():
    from analysis.ur001_derive_cli import records_for
    from analysis.ur001_science_entry import A_ROOT
    records,sha,prefix=records_for('full')
    physical={'profile':'full','inventory_sha256':sha,'data_root':str(A_ROOT/prefix),
              'files':[{**r,'rows':1,'row_groups':[1]} for r in records]}
    admitted={'train':[str(A_ROOT/prefix/r['path']) for r in records[:25]],
              'validation':str(A_ROOT/prefix/records[25]['path'])}
    return physical,admitted


def test_original_manifest_relative_paths_map_to_exact_full_whitelist():
    from analysis.p_full_training_backend import physical_records
    physical,admitted=fixture()
    records=physical_records(physical,admitted)
    assert [path for _,path in records]==[*admitted['train'],admitted['validation']]
    assert all('/subsamples/ur0.01/subsamples/' not in path for _,path in records)


@pytest.mark.parametrize('change',['root','double-prefix','profile','inventory','order','missing','admission'])
def test_arbitrary_roots_prefixes_inventory_and_order_fail_closed(change):
    from analysis.p_full_training_backend import physical_records
    physical,admitted=fixture();physical=copy.deepcopy(physical)
    if change=='root':physical['data_root']='/tmp/unapproved'
    elif change=='double-prefix':physical['files'][0]['path']='subsamples/ur0.01/'+physical['files'][0]['path']
    elif change=='profile':physical['profile']='subset'
    elif change=='inventory':physical['inventory_sha256']='0'*64
    elif change=='order':physical['files'][0],physical['files'][1]=physical['files'][1],physical['files'][0]
    elif change=='missing':physical['files'].pop()
    else:admitted['train'][0]=admitted['train'][1]
    with pytest.raises(ValueError):physical_records(physical,admitted)
