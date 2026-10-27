"""Four-step compiled-job and exact resume check, not a reconstruction experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
import train


def verify(job,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent.resolve()
    sys.path.insert(0,str(root.parents[1]/'src'))
    from colabcola.invocation import load_job_invocation
    from colabcola.compiler import compile_invocation
    invocation=load_job_invocation(Path(job))
    first=compile_invocation(invocation,output/'job.ipynb')
    second=compile_invocation(invocation,output/'repeat.ipynb')
    assert first.notebook_bytes==second.notebook_bytes
    namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(output/'runtime')}
    for cell in json.loads(first.notebook_bytes)['cells']:
        if cell['cell_type']=='code': exec(cell['source'],namespace)
    results=output/'runtime/colabcola/results'/first.job_id
    from colabcola.lifecycle import verify_final
    result=verify_final(results,first.job_id)
    assert result['terminal_state']=='success'
    full=output/'runtime/colabcola/jobs'/first.job_id/'outputs/fit'
    state=train.restore(full/'state-00000004.pt.json')
    data=Path(job).parent/'fixed-image'
    for name,steps,resume in [('first',2,None),('continued',4,output/'first/state-00000002.pt.json')]:
        command=[sys.executable,str(root/'train.py'),'--data',str(data),'--output',str(output/name),'--device','cpu','--steps',str(steps),'--checkpoint-every',str(steps)]
        if resume: command.extend(['--resume',str(resume)])
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    continued=train.restore(output/'continued/state-00000004.pt.json')
    assert state['losses']==continued['losses']
    assert all(torch.equal(value,continued['model'][key]) for key,value in state['model'].items())
    assert all(int(s['step'])==4 for s in continued['optimizer']['state'].values())
    evidence={'scope':'local packaging/correctness only; not Colab execution','job_id':first.job_id,'notebook_sha256':first.notebook_sha256,
              'deterministic_notebook':True,'verified_outputs':len(result['outputs']),'fresh_process_resume_exact':True,
              'terminal_state':result['terminal_state'],'siren_parameters':sum(p.numel() for p in train.model().parameters()),
              'artifact_sha256':train.digest(results/'artifacts.zip')}
    train.atomic(output/'verification.json',evidence); print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.job,a.output)
