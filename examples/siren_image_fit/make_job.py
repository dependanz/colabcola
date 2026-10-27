"""Build a fixed-image SIREN colajob; no upload or runtime allocation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import numpy as np
from train import atomic,digest


def prepare(args):
    root=Path(__file__).parent.resolve()
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise ValueError('job already exists')
    source=Path(args.source).resolve()
    manifest=json.loads((source/'splits.json').read_text())
    if digest(source/'test.npy')!=manifest['files']['test']['sha256']:
        raise ValueError('source array hash mismatch')
    images=np.load(source/'test.npy',allow_pickle=False)
    image=images[0].transpose(1,2,0).copy()
    data=out.parent/'fixed-image'
    if not data.exists():
        data.mkdir()
        np.save(data/'image.npy',image,allow_pickle=False)
        atomic(data/'manifest.json',{'schema':1,'image_sha256':digest(data/'image.npy'),
               'source_array_sha256':digest(source/'test.npy'),'source_member':'test_batch.bin',
               'source_record':32,'selection':'first image of the retained comparison; fixed before fitting'})
    else:
        assert np.array_equal(np.load(data/'image.npy',allow_pickle=False),image)
    profile=json.loads((root.parent/'hyperplus_image_fit/COLAB_RUNTIME_20261015.json').read_text())
    packages={p:importlib.metadata.version(p) for p in ('torch','numpy','pillow')} if args.local_cpu else profile['packages']
    python=platform.python_version() if args.local_cpu else profile['python']
    payload=[]
    for role,path,target in [('source',root/'train.py','workload/train.py'),('data',data/'image.npy','data/image.npy'),('config',data/'manifest.json','data/manifest.json')]:
        payload.append({'role':role,'source':os.path.relpath(path,out.parent).replace('\\','/'),'target':target})
    job={'schema_version':1,'name':'siren-fixed-image-check' if args.local_cpu else 'siren-fixed-image-colab',
         'payload':payload,'entrypoint':{'kind':'python_script','path':'workload/train.py','cwd':'.',
         'argv':['--data','data','--output','outputs/fit','--device','cpu' if args.local_cpu else 'cuda','--steps',str(args.steps),'--checkpoint-every',str(min(250,args.steps))]},
         'python':{'requires_python':'=='+python,'runtime_requirements':[f'{p}=={v}' for p,v in packages.items()],'install_requirements':[],'embedded_wheels':[]},
         'runtime':{'provider':'google_colab','version':'latest'},
         'resources':{'accelerator':'cpu' if args.local_cpu else 'gpu','gpu_count':0 if args.local_cpu else 1,'cuda_required':not args.local_cpu,'minimum_gpu_memory_bytes':None if args.local_cpu else 4*1024**3},
         'run':{'timeout_seconds':600,'seed':1234,'environment':{'PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'},'capture_stdout':True,'capture_stderr':True},
         'outputs':[{'pattern':'outputs/fit/'+name,'required':True,'maximum_bytes':size} for name,size in [
          ('identity.json',1048576),('metrics.json',1048576),('events.ndjson',16777216),('comparison.png',1048576),('reconstruction.npy',1048576),
          (f'state-{args.steps:08d}.pt',33554432),(f'state-{args.steps:08d}.pt.json',1048576)]],
         'limits':{'maximum_file_count':16},'managed_recovery':{'checkpoint_pattern':'outputs/fit/state-*.pt.json','log_patterns':['outputs/fit/events.ndjson','outputs/fit/metrics.json','outputs/fit/identity.json']}}
    atomic(out,job)
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000); p.add_argument('--local-cpu',action='store_true')
    prepare(p.parse_args())
