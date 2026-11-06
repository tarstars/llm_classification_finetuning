"""Atomic, contract-bound raw prediction chunks for recoverable inference."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from .experiment import read_json,atomic_json,sha256


def object_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_arrays(ids,original,aligned_swap):
    ids=np.asarray(ids);assert ids.ndim==1 and len(ids)==len(set(map(int,ids)))
    for p in [original,aligned_swap]:
        p=np.asarray(p);assert p.shape==(len(ids),3) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)


class ChunkStore:
    def __init__(self,directory,contract):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.digest=object_digest(contract);self.path=self.root/'manifest.json'
        if self.path.exists():
            self.manifest=read_json(self.path);assert self.manifest['contract_sha256']==self.digest,'Incompatible numerical/source contract'
        else:
            self.manifest={'contract_sha256':self.digest,'contract':contract,'chunks':{}};atomic_json(self.path,self.manifest)

    def read(self,start,ids):
        record=self.manifest['chunks'].get(str(start))
        if record is None:return None
        path=self.root/record['file'];assert sha256(path)==record['sha256'],'Changed committed prediction chunk'
        with np.load(path) as raw:
            np.testing.assert_array_equal(raw['ids'],ids)
            arrays={k:raw[k].copy() for k in ['ids','original','aligned_swap']}
        validate_arrays(**arrays);return arrays

    def write(self,start,ids,original,aligned_swap,**telemetry):
        assert str(start) not in self.manifest['chunks'],'Preserve existing committed chunks'
        validate_arrays(ids,original,aligned_swap)
        path=self.root/f'{start:04d}.npz';assert not path.exists(),'Unreconciled artifact without a manifest entry'
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,ids=np.asarray(ids),original=original,aligned_swap=aligned_swap);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        self.manifest['chunks'][str(start)]={'file':path.name,'rows':len(ids),'sha256':sha256(path),**telemetry}
        atomic_json(self.path,self.manifest)

    def assemble(self,starts,ids):
        arrays=[];offset=0
        for start in starts:
            n=self.manifest['chunks'][str(start)]['rows'];raw=self.read(start,np.asarray(ids)[offset:offset+n]);assert raw is not None
            arrays.append(raw);offset+=n
        assert offset==len(ids),'Incomplete ordered prediction coverage'
        out={k:np.concatenate([r[k] for r in arrays],axis=0) for k in ['ids','original','aligned_swap']}
        np.testing.assert_array_equal(out['ids'],ids);validate_arrays(**out);return out
