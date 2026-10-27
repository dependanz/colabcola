"""Tiny compiled Kodak acceptance using only the public general packaging CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from colabcola.cli import main
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final


def verify(destination):
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=False)
    data=root/'fixture';data.mkdir()
    image=data/'kodim01.png'
    Image.fromarray(np.random.default_rng(7).integers(0,256,(8,8,3),dtype=np.uint8)).save(image)
    (data/'manifest.json').write_text(json.dumps({'schema':1,'images':[{'name':image.name,'shape':[8,8,3],
        'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}]}),encoding='utf-8')
    job=root/'kodak.colajob'
    script=Path(__file__).resolve().parents[1]/'examples/kodak_grid/run.py'
    command=['package','--output',str(job),'--data',str(data)+'=data',
        '--runtime-requirement','torch','--runtime-requirement','numpy','--runtime-requirement','pillow',
        '--artifact','outputs/grid/fits/*.zip','--artifact','outputs/grid/summary.json',
        '--artifact','outputs/grid/identity.json','--artifact','outputs/grid/progress.json','--artifact','outputs/grid/events.ndjson',
        '--checkpoint','outputs/grid/state-*.pt.json','--resume-arg=--resume',
        str(script),'--data','data','--device','cpu','--steps','4','--pixels','16','--checkpoint-every','2']
    assert main(command)==0
    first=compile_invocation(load_job_invocation(job),root/'first.ipynb')
    second=compile_invocation(load_job_invocation(job),root/'second.ipynb')
    assert first.notebook_bytes==second.notebook_bytes
    namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(root/'runtime')}
    for cell in json.loads(first.notebook_bytes)['cells']:
        if cell['cell_type']=='code':exec(cell['source'],namespace)
    results=root/'runtime/colabcola/results'/first.job_id
    result=verify_final(results,first.job_id)
    assert result['terminal_state']=='success' and len(result['outputs'])==16
    work=root/'runtime/colabcola/jobs'/first.job_id
    summary=json.loads((work/'outputs/grid/summary.json').read_text())
    assert summary['completed_fits']==24
    assert all(p['initial_sha256'][0]==p['initial_sha256'][1] for p in summary['pairs'])
    proof={'scope':'Local compiled 8x8 synthetic Kodak-shaped fixture; not a new Colab run',
        'entrypoint_source':str(script),'package_command':command,'job_id':first.job_id,
        'notebook_sha256':first.notebook_sha256,'deterministic_notebook':True,
        'completed_fits':24,'steps_per_fit':4,'verified_outputs':16,
        'artifact_sha256':hashlib.sha256((results/'artifacts.zip').read_bytes()).hexdigest(),
        'terminal_state':result['terminal_state']}
    (root/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination');verify(p.parse_args().destination)
