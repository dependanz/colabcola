"""Convert a trusted licensed FLAME pickle to numeric arrays without legacy Chumpy."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle


def convert(source, output):
    import numpy as np
    import scipy.sparse
    if output.exists() or source.resolve()==output.resolve():raise ValueError("choose a new output path")
    # Compatibility shims are confined to this conversion process, never the training runtime.
    for name,value in dict(bool=bool,int=int,float=float,complex=complex,object=object,str=str,unicode=str).items():
        if name not in np.__dict__:setattr(np,name,value)
    if not hasattr(inspect,"getargspec"):inspect.getargspec=inspect.getfullargspec
    import chumpy
    with source.open("rb") as f:original=pickle.load(f,encoding="latin1")
    converted={};records={}
    for key,value in original.items():
        if isinstance(value,chumpy.Ch):value=value.r
        if scipy.sparse.issparse(value):value=value.toarray()
        if isinstance(value,np.ndarray):
            if value.dtype.hasobject or not np.isfinite(value).all():raise ValueError("non-numeric model array")
            converted[key]=value.copy()
            records[key]={"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(value.tobytes()).hexdigest()}
        elif isinstance(value,(str,int,float)):converted[key]=value
        else:raise ValueError("unsupported FLAME value: "+key)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as f:pickle.dump(converted,f,protocol=4)
    with output.open("rb") as f:restored=pickle.load(f)
    for key,value in converted.items():
        if isinstance(value,np.ndarray):assert np.array_equal(value,restored[key])
        else:assert value==restored[key]
    report={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "arrays":records,"numeric_values_preserved":True,"license":"Original FLAME license still applies; do not redistribute"}
    output.with_suffix(".conversion.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(json.dumps(convert(a.source,a.output),indent=2))

if __name__=="__main__":main()
