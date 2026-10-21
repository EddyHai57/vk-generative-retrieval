"""Read-only JSONL offset views and dense uint8 side storage for frozen P batches.

The original artifacts, scientific batch functions, cursor and optimizer are unchanged.
Callers retain the original input/job/source closure and combined cache/output guard.
"""
from collections.abc import Sequence
import hashlib
import json
import time
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import numpy as np


def checked_root(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('absolute non-symlink storage path required')
    return path


def file_identity(path):
    stat=Path(path).stat()
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def canonical_chunks(value):
    """Original C encoder per ordinary row; only lazy outer sequences stream."""
    if not isinstance(value,RowSequence):
        try:
            raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        except TypeError:
            # Only a necessary outer container containing a lazy sequence
            # reaches this branch; ordinary per-row ints stay in the C encoder.
            if isinstance(value,dict):
                yield b'{'
                for index,key in enumerate(sorted(value)):
                    if type(key)is not str:raise ValueError('string lazy-container keys required')
                    if index:yield b','
                    yield json.dumps(key).encode();yield b':'
                    yield from canonical_chunks(value[key])
                yield b'}';return
            if not isinstance(value,(list,tuple)):raise
        else:
            yield raw;return
    yield b'['
    for index,item in enumerate(value):
        if index:yield b','
        yield from canonical_chunks(item)
    yield b']'


def digest(value):
    result=hashlib.sha256()
    for chunk in canonical_chunks(value):result.update(chunk)
    return result.hexdigest()


class RowSequence(Sequence):
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self))) ]
        if type(index)is not int:raise TypeError('integer row index required')
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError('row index')
        return self.row(index)


class JsonRows(RowSequence):
    """Only line offsets reside in mmap; an access materializes one original row."""
    @classmethod
    def build(cls,source,offsets,count,expected_sha256,guard):
        source=checked_root(source);offsets=checked_root(offsets)
        if type(count)is not int or count<1:raise ValueError('positive producer row count required')
        guard();offsets.parent.mkdir(parents=True,exist_ok=True)
        if hasattr(guard,'projected'):guard.projected((count+1)*8)
        # New-only cache: never truncate an existing file.
        with offsets.open('xb') as output,source.open('rb') as stream:
            value=hashlib.sha256();seen=0;position=0
            output.write(np.uint64(0).tobytes())
            for line in stream:
                guard();value.update(line);position+=len(line);seen+=1
                if seen>count:raise ValueError('producer row count differs')
                output.write(np.uint64(position).tobytes())
            if seen!=count:raise ValueError('producer row count differs')
            if value.hexdigest()!=expected_sha256:raise ValueError('producer JSONL hash differs')
        guard();result=cls();result.source=source;result.offsets=np.memmap(offsets,dtype=np.uint64,mode='r',shape=(count+1,))
        result.guard=guard;result._identity=result.identity();result._offset_identity=file_identity(offsets);return result

    def identity(self):
        return file_identity(self.source)

    def __len__(self):return len(self.offsets)-1

    def __iter__(self):
        self.guard()
        if file_identity(self.offsets.filename)!=self._offset_identity:raise ValueError('offset cache changed')
        if self.identity()!=self._identity:raise ValueError('original JSONL changed during scan')
        with self.source.open('rb') as stream:
            count=0
            for raw in stream:
                if count>=len(self):raise ValueError('original JSONL count changed')
                self.guard();row=json.loads(raw)
                if type(row)is not dict:raise ValueError('original JSONL row must be object')
                count+=1;yield row
            if count!=len(self) or self.identity()!=self._identity:raise ValueError('original JSONL changed during scan')
        self.guard()

    def row(self,index):
        self.guard()
        if file_identity(self.offsets.filename)!=self._offset_identity:raise ValueError('offset cache changed')
        if self.identity()!=self._identity:raise ValueError('original JSONL changed during access')
        start,end=map(int,self.offsets[index:index+2])
        with self.source.open('rb') as stream:
            stream.seek(start);raw=stream.read(end-start)
        if len(raw)!=end-start:raise ValueError('original JSONL truncated')
        row=json.loads(raw)
        if type(row)is not dict:raise ValueError('original JSONL row must be object')
        self.guard();return row


class SideRows(RowSequence):
    def __init__(self,rows,side):self.rows=rows;self.side=side
    def __len__(self):return len(self.rows)
    def __iter__(self):
        for row in self.rows:yield {**row,'side':self.side.lookup(row)}
    def row(self,index):
        row=self.rows[index]
        return {**row,'side':self.side.lookup(row)}


class CutRows(RowSequence):
    def __init__(self,rows,length):
        if type(length)is not int or length<1:raise ValueError('positive history length required')
        self.rows=rows;self.length=length
    def __len__(self):return len(self.rows)
    def __iter__(self):
        for row in self.rows:yield self.cut(row)
    def row(self,index):return self.cut(self.rows[index])
    def cut(self,row):
        length=self.length
        if len(row['side'])!=len(row['history']):raise ValueError('side/history misaligned')
        value={**row,'history':row['history'][-length:],'history_positions':row['history_positions'][-length:],'side':row['side'][-length:]}
        if 'ctx' in row:value['ctx']=row['ctx'][-length:]
        return value


class DenseSide:
    """Wanted/found flags and original side by physical week/ordinal, disk backed."""
    DTYPE=np.dtype([('item','<u4'),('side','u1',(3,)),('state','u1')])

    @classmethod
    def build(cls,root,history,partitions,durations,encode,guard):
        root=checked_root(root);root.mkdir(parents=True,exist_ok=False);guard()
        maxima={}
        for split in ('train','validation'):
            for row in history[split]:
                guard()
                if len(row['history'])!=len(row['history_positions']):raise ValueError('history positions misaligned')
                for position in row['history_positions']:
                    if (type(position)is not list or len(position)!=2 or any(type(v)is not int or v<0 for v in position)
                            or position[0]>24):raise ValueError('invalid physical history position')
                    week,ordinal=position;maxima[week]=max(maxima.get(week,-1),ordinal)
        arrays={}
        for week,maximum in maxima.items():
            guard();path=root/('week-'+str(week)+'.bin')
            if hasattr(guard,'projected'):guard.projected((maximum+1)*cls.DTYPE.itemsize)
            with path.open('xb') as stream:stream.truncate((maximum+1)*cls.DTYPE.itemsize)
            guard();arrays[week]=np.memmap(path,dtype=cls.DTYPE,mode='r+',shape=(maximum+1,))
        for split in ('train','validation'):
            for row in history[split]:
                guard()
                for item,(week,ordinal) in zip(row['history'],row['history_positions']):
                    if type(item)is not int or not 0<=item<2**32:raise ValueError('uint32 history item required')
                    record=arrays[week][ordinal]
                    if record['state'] and int(record['item'])!=item:raise ValueError('history positions disagree on item')
                    record['item']=item;record['state']=1
        previous=-1
        for week,events in partitions:
            if type(week)is not int or not previous<week<=24:raise ValueError('train ordering conflict / non-train week')
            previous=week;array=arrays.get(week)
            for ordinal,row in enumerate(events):
                if ordinal%65536==0:guard()
                if array is None or ordinal>=len(array) or array[ordinal]['state']!=1:continue
                record=array[ordinal];item=int(record['item'])
                if row.get('item_id')!=item:raise ValueError('side/history item mismatch')
                side=encode(row,durations.get(item))
                if type(side)is not list or len(side)!=3 or any(type(v)is not int or not 0<=v<=255 for v in side):raise ValueError('original uint8 side encoding differs')
                record['side']=side;record['state']=2
        for array in arrays.values():
            for start in range(0,len(array),65536):
                guard()
                if np.any(array['state'][start:start+65536]==1):raise ValueError('history position absent from train partitions')
            array.flush()
        result=cls();result.arrays={week:np.memmap(root/('week-'+str(week)+'.bin'),dtype=cls.DTYPE,mode='r',shape=array.shape) for week,array in arrays.items()}
        result.guard=guard;result.identities={week:file_identity(array.filename) for week,array in result.arrays.items()};guard();return result

    def lookup(self,row):
        self.guard();result=[];checked=set()
        if len(row['history'])!=len(row['history_positions']):raise ValueError('history positions misaligned')
        for item,(week,ordinal) in zip(row['history'],row['history_positions']):
            array=self.arrays.get(week)
            if array is None or not 0<=ordinal<len(array):raise ValueError('history position absent from side store')
            if week not in checked:
                if file_identity(array.filename)!=self.identities[week]:raise ValueError('side cache changed')
                checked.add(week)
            record=array[ordinal]
            if record['state']!=2 or int(record['item'])!=item:raise ValueError('side/history item mismatch')
            result.append([int(v) for v in record['side']])
        return result


def view(data,length):return CutRows(data['rows'],length),CutRows(data['queries'],length)


def clone(function,replacements):
    namespace=dict(function.__globals__)
    if not set(replacements)<=set(namespace):raise ValueError('closed original globals required')
    namespace.update(replacements)
    if {key for key in namespace if namespace[key] is not function.__globals__[key]}!=set(replacements):raise ValueError('closed replacement set differs')
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__;return result


def facade(native,guard):
    """Closed storage facade; original numerical functions remain identical objects."""
    for module,pin,names in ((native,'e81c9183c0cda549a149ed46ae56efc2bfd11e9c3e90f22ff125353e0de559b7',('binding_for4','verify_run')),
            (native.fr,'1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd',('_view','binding_for2')),
            (native.b1,'5b8ba48afcd86e86b222455d2105764454f1dbab151a4641b0cdf649c30243d6',('binding_for',))):
        raw=Path(module.__file__).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('frozen storage source differs')
        compiled=compile(raw,module.__file__,'exec',dont_inherit=True)
        for name in names:
            expected=next(code for code in compiled.co_consts if isinstance(code,CodeType) and code.co_name==name)
            if getattr(module,name).__code__!=expected:raise ValueError('original storage function code differs')
    b1=SimpleNamespace(**{**vars(native.b1),'_digest':digest})
    b1.binding_for=clone(native.b1.binding_for,{'_digest':digest})
    def resource_guard(output,cap):
        original=native.b1.resource_guard(output,cap)
        def combined():
            original();guard()
        return combined
    b1.resource_guard=resource_guard
    def binding_for2(data,config,rows,queries,model,device,inputs_binding):
        guard()
        base=b1.binding_for(data,config,rows,queries,model.init_report,device,inputs_binding)
        fr=native.fr
        result={**base,'task':'P-FEAT02','feature':config['feature'],'side_version':fr.SIDE_VERSION,
                'side_sha256':digest(ProjectedRows(rows,queries,'side')),
                'bucket_spec_sha256':fr.bucket_spec_sha256(),'param_report':fr.parameter_report(model)}
        if config['feature']=='f3_f4_recency':result['weeks_sha256']=digest(ProjectedRows(rows,queries,'weeks'))
        if config['feature']=='f3_f5_scene':result['ctx_sha256']=digest(ProjectedRows(rows,queries,'ctx'))
        if config['feature']==fr.F6_FEATURE:result['attr_sha256']=digest([data['author_index'].tolist(),data['author_vocab']])
        guard();return result
    fr=SimpleNamespace(**{**vars(native.fr),'_view':view,'binding_for2':binding_for2})
    binding=clone(native.binding_for4,{'fr':fr})
    verify=clone(native.verify_run,{'fr':fr,'binding_for4':binding})
    return SimpleNamespace(**{**vars(native),'fr':fr,'b1':b1,'binding_for4':binding,'verify_run':verify,'compact_original':native})


class ProjectedRows(RowSequence):
    def __init__(self,rows,queries,field):self.rows=rows;self.queries=queries;self.field=field
    def __len__(self):return len(self.rows)+len(self.queries)
    def __iter__(self):
        for collection in (self.rows,self.queries):
            for row in collection:yield self.project(row)
    def project(self,row):return [p[0] for p in row['history_positions']] if self.field=='weeks' else row[self.field]
    def row(self,index):
        row=self.rows[index] if index<len(self.rows) else self.queries[index-len(self.rows)]
        return self.project(row)


def adapt_rows(native,rows,queries,catalog,supported,partitions,durations,features,*,root,guard,synthetic=False):
    """Same original adapter validation; side values materialize only per row access."""
    import torch
    import importlib
    from vkgr.retrieval_data import catalog_map
    idmap=catalog_map(catalog)
    support=set(supported)
    if supported!=sorted(support) or not support<=set(catalog):raise ValueError('train support/catalog differs')
    if (features.dtype!=np.float32 or features.shape!=(len(catalog)+1,64)
            or not np.isfinite(features).all() or np.any(features[0])):raise ValueError('original content shape/dtype/PAD differs')
    for train,collection in ((True,rows),(False,queries)):
        for ordinal,row in enumerate(collection):
            guard();history=row['history'];positions=row['history_positions'];target=row['target_position']
            if (not 1<=len(history)<=50 or len(history)!=len(positions)
                    or any(i not in idmap for i in history)
                    or any(type(p)is not list or len(p)!=2 or any(type(v)is not int or v<0 for v in p) or p[0]>24 for p in positions)
                    or any(a>=b for a,b in zip(positions,positions[1:]))
                    or type(target)is not list or len(target)!=2
                    or any(type(v)is not int or v<0 for v in target) or positions[-1]>=target
                    or (train and (target[0]>24 or row.get('example_id')!=ordinal or row['target'] not in support))
                    or (not train and (target[0]!=25 or row.get('oov') is not (row['target'] not in idmap)))):
                raise ValueError('UR001 history/target identity differs')
    side_module=importlib.import_module(native.__package__+'.history_side')
    side=DenseSide.build(root,{'train':rows,'validation':queries},partitions,durations,side_module._encode,guard)
    return {'rows':SideRows(rows,side),'queries':SideRows(queries,side),'catalog':catalog,'idmap':idmap,
            'content':torch.from_numpy(features),'support':support,'supported_indices':[idmap[i] for i in supported],
            'synthetic':synthetic}


class CombinedGuard:
    """Original headroom guard plus one unchanged cap for output and both caches."""
    def __init__(self,base,roots,cap):
        self.base=base;self.roots=[checked_root(root) for root in roots];self.cap=cap;self.last=None
        if type(cap)is not int or cap<=0:raise ValueError('positive original disk cap required')
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(self.roots) for b in self.roots[i+1:]):
            raise ValueError('disjoint output/cache roots required')
    def projected(self,additional):
        if type(additional)is not int or additional<0:raise ValueError('nonnegative projected bytes required')
        self.base();total=0
        for root in self.roots:
            if root.is_symlink():raise ValueError('symlink output/cache refused')
            if root.exists():
                for path in root.rglob('*'):
                    if path.is_symlink():raise ValueError('symlink output/cache refused')
                    if path.is_file():total+=path.stat().st_size
        if total+additional>self.cap:raise OSError('combined output/compact/parallel cache exceeds original disk cap')
        self.last=time.monotonic()
    def __call__(self):
        self.base()
        # Same sampled resource discipline as the original 30-second guard;
        # every allocation additionally calls projected without sampling.
        if self.last is None or time.monotonic()-self.last>=30:self.projected(0)


def cache_root(output):
    output=checked_root(output)
    return output.parent/(output.name+'-compact-cache')


def make_guard(native,value):
    output=checked_root(value['output_root']);cap=value['disk_cap_bytes']
    return CombinedGuard(native.b1.resource_guard(output,cap),
            [output,cache_root(output),output.parent/(output.name+'-parallel-eval')],cap)


def load_rows(root,cache,manifest,guard,*,query_count=None):
    root=checked_root(root);cache=checked_root(cache)
    rows=JsonRows.build(root/'UR001-train.jsonl',cache/'train-offsets',manifest['counts']['train_examples'],
            manifest['artifacts']['UR001-train.jsonl']['sha256'],guard)
    queries=JsonRows.build(root/'UR001-validation.jsonl',cache/'validation-offsets',manifest['counts']['eligible'],
            manifest['artifacts']['UR001-validation.jsonl']['sha256'],guard)
    if query_count is not None:
        if type(query_count)is not int or not 1<=query_count<=len(queries):raise ValueError('fixed query prefix unavailable')
        queries=queries[:query_count]
    return rows,queries
