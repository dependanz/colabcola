"""Ordinary stdlib workload for a bounded dated-image/Drive-renewal acceptance run."""
import argparse,hashlib,json,os,platform,random,time
from pathlib import Path

def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(temporary,path)

def run(seconds=3900,interval=60,*,clock=time.monotonic,sleep=time.sleep):
    if not 0<=seconds<=3900 or not 0<interval<=60:raise ValueError('probe bounds: 0-3900 seconds, interval <=60')
    if os.environ.get('COLABCOLA_RESUME_MANIFEST'):raise RuntimeError('this one-runtime probe does not resume')
    root=Path('outputs/probe');root.mkdir(parents=True,exist_ok=True)
    started=clock();step=0
    while True:
        step+=1;body=random.Random(step).randbytes(1024*1024)
        filename=f'state-{step:04d}.bin';(root/filename).write_bytes(body)
        elapsed=clock()-started
        atomic(root/(filename+'.json'),{'schema':1,'step':step,'file':filename,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        event={'step':step,'elapsed_seconds':elapsed,'python':platform.python_version()}
        with (root/'events.ndjson').open('a',encoding='utf-8') as log:log.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        if elapsed>=seconds:break
        sleep(min(interval,seconds-elapsed))
    atomic(root/'summary.json',{'purpose':'dated-runtime and Drive-renewal acceptance','requested_seconds':seconds,
        'elapsed_seconds':clock()-started,'checkpoints':step,'bytes_per_checkpoint':1024*1024,'python':platform.python_version()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=3900);parser.add_argument('--interval',type=int,default=60)
    args=parser.parse_args();run(args.seconds,args.interval)
