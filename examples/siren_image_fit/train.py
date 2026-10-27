"""Fixed-image SIREN integration workload for colabcola; no hypernetwork or search."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch import nn


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic(path, value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


class Sine(nn.Module):
    def forward(self,x):
        return torch.sin(30*x)


def model(width=64, depth=2):
    # SIREN first-layer and sine-layer initialization, with a linear RGB output.
    layers=[]
    for index in range(depth):
        layer=nn.Linear(2 if index==0 else width,width)
        bound=1/layer.in_features if index==0 else math.sqrt(6/layer.in_features)/30
        nn.init.uniform_(layer.weight,-bound,bound)
        layers.extend((layer,Sine()))
    output=nn.Linear(width,3)
    nn.init.uniform_(output.weight,-math.sqrt(6/width)/30,math.sqrt(6/width)/30)
    layers.append(output)
    return nn.Sequential(*layers)


def coordinates(height,width):
    y,x=torch.meshgrid((torch.arange(height)+.5)/height,(torch.arange(width)+.5)/width,indexing='ij')
    return torch.stack((x,y),-1).reshape(-1,2)*2-1


def checkpoint(out,state):
    path=out/f"state-{state['step']:08d}.pt"
    if path.exists() or path.with_suffix('.pt.json').exists():
        raise ValueError('immutable checkpoint already exists')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
    atomic(path.with_suffix('.pt.json'),{'schema':1,'step':state['step'],'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)})


def restore(manifest):
    manifest=Path(manifest)
    record=json.loads(manifest.read_text())
    name=record['file']
    if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name:
        raise ValueError('unsafe checkpoint member')
    path=manifest.parent/name
    if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
        raise ValueError('checkpoint integrity failure')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['schema']!=1 or record['schema']!=1 or state['step']!=record['step']:
        raise ValueError('checkpoint schema/step mismatch')
    return state


def run(args):
    if min(args.width,args.depth,args.steps,args.checkpoint_every)<1:
        raise ValueError('positive architecture and step counts required')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(1234); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    data=Path(args.data)
    meta=json.loads((data/'manifest.json').read_text())
    if digest(data/'image.npy')!=meta['image_sha256']:
        raise ValueError('fixed image integrity failure')
    array=np.load(data/'image.npy',allow_pickle=False)
    if array.dtype!=np.uint8 or array.ndim!=3 or array.shape[-1]!=3:
        raise ValueError('expected uint8 HWC RGB image')
    height,width=array.shape[:2]
    device=torch.device(args.device)
    truth=torch.from_numpy(array.copy()).to(device).float()/255
    grid=coordinates(height,width).to(device)
    target=truth.reshape(-1,3)*2-1
    net=model(args.width,args.depth).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    identity={'schema':1,'source_sha256':digest(__file__),'image_sha256':meta['image_sha256'],
              'data_manifest_sha256':digest(data/'manifest.json'),'width':args.width,'hidden_layers':args.depth,
              'omega':30,'lr':1e-4,'seed':1234,'torch':str(torch.__version__),'device':device.type,
              'coordinate_mapping':'pixel centers [0,1]^2 mapped to [-1,1]^2',
              'target_mapping':'RGB [0,1] mapped to [-1,1]; linear output; no Fourier encoding'}
    start=0; losses=[]
    if args.resume:
        state=restore(args.resume)
        if state['identity']!=identity or state['step']>=args.steps:
            raise ValueError('resume identity/target mismatch')
        net.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        start=state['step']; losses=state['losses']
    atomic(out/'identity.json',{**identity,'restored_step':start})
    started=time.monotonic()
    with (out/'events.ndjson').open('w') as log:
        def emit(event,**values):
            line=json.dumps({'event':event,**values},sort_keys=True,allow_nan=False)
            log.write(line+'\n'); log.flush(); print(line,flush=True)
        emit('started',restored_step=start,target_steps=args.steps,device=str(device),
             gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
        for step in range(start+1,args.steps+1):
            opt.zero_grad(set_to_none=True)
            loss=(net(grid)-target).square().mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward()
            grads=[p.grad for p in net.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in grads): raise ValueError('invalid gradient')
            norm=float(torch.stack([g.norm().square() for g in grads]).sum().sqrt())
            if norm<=0: raise ValueError('zero gradient')
            opt.step(); losses.append(float(loss.detach()))
            emit('train_step',step=step,rgb_mse=losses[-1]/4,gradient_norm=norm)
            if step%args.checkpoint_every==0 or step==args.steps:
                checkpoint(out,{'schema':1,'step':step,'identity':identity,'model':net.state_dict(),
                                'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
                                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'losses':losses})
                emit('checkpoint_committed_local',step=step,manifest=f'state-{step:08d}.pt.json')
        with torch.no_grad(): pred=(net(grid).reshape(height,width,3)+1)/2
        mse=float((pred-truth).square().mean())
        edge=float(sum((pred.diff(dim=d)-truth.diff(dim=d)).square().mean() for d in (0,1))/2)
        np.save(out/'reconstruction.npy',pred.cpu().numpy(),allow_pickle=False)
        panels=np.concatenate((array,(pred.cpu().numpy().clip(0,1)*255).round().astype(np.uint8)),axis=1)
        Image.fromarray(panels).resize((width*8,height*4),Image.Resampling.NEAREST).save(out/'comparison.png')
        atomic(out/'metrics.json',{'step':args.steps,'restored_step':start,'images':1,'height':height,'width':width,
                                  'parameters':sum(p.numel() for p in net.parameters()),'rgb_mse_unclipped':mse,
                                  'psnr_db_unclipped':-10*math.log10(max(mse,1e-12)),'edge_mse_unclipped':edge,
                                  'elapsed_seconds':time.monotonic()-started,'scope':'direct fitting of one fixed image; not held-out generalization'})
        emit('completed_local',step=args.steps)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--steps',type=int,default=2000); p.add_argument('--checkpoint-every',type=int,default=250)
    p.add_argument('--width',type=int,default=64); p.add_argument('--depth',type=int,default=2)
    p.add_argument('--resume',default=os.environ.get('COLABCOLA_RESUME_MANIFEST'))
    return p

if __name__=='__main__': run(parser().parse_args())
