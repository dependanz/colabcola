"""Verify the probe locally; optional explicit cold Drive/API checks never allocate."""
import argparse,json,sys,zipfile
from pathlib import Path
from colabcola.lifecycle import verify_final,get_token
from colabcola.lifecycle_io import DriveStore,recover,atomic
from colabcola.lifecycle_auth import TokenLease

def check(directory):
    root=Path(directory).resolve();plan=json.loads((root/'plan.json').read_text());state=json.loads((root/'state.json').read_text())
    backend=plan.get('provider_backend','legacy')
    assert backend in ('legacy','rest') and plan['gpu']=='NONE'
    assert plan['runtime_version']==('latest' if backend=='legacy' else '2026.07')
    assert plan['max_attempts']==1 and plan['max_seconds']==4500
    assert state['phase']=='completed' and len(state['attempts'])==1 and state['attempts'][0].get('released')
    final=Path(state['final']['local_directory']);result=verify_final(final,plan['job_id'])
    assert result['terminal_state']=='success'
    with zipfile.ZipFile(final/'artifacts.zip') as archive:summary=json.loads(archive.read('outputs/probe/summary.json'))
    assert summary['requested_seconds']==3900 and summary['elapsed_seconds']>=3900
    if backend=='rest':
        assert summary['python'].startswith('3.12.')
    else:
        major,minor,*_=map(int,summary['python'].split('.'))
        assert major==3 and minor>=11
    evidence=json.loads((final/'authorization.json').read_text())
    assert evidence['token_changes']>=1 and evidence['latest_expires_at']>evidence['initial_expires_at']
    assert evidence['verified_after_initial_expiry_at']>=evidence['initial_expires_at']+120
    assert evidence['first_verified_rotated_publication_at']<=evidence['verified_after_initial_expiry_at']
    return root,plan,state,{'state':'verified','scope':'local recovered evidence','job_id':plan['job_id'],'provider_backend':backend,'runtime_version':plan['runtime_version'],'dated_image_verified':backend=='rest','summary':summary,'authorization':evidence}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir',type=Path);parser.add_argument('--drive-readback',action='store_true');args=parser.parse_args()
    root,plan,state,report=check(args.run_dir)
    if args.drive_readback:
        token=TokenLease(lambda:get_token(plan['token_command']))
        store=DriveStore(plan['parent'],plan['run_id'],lambda:token()['access_token'])
        reference=state['final'];commit=json.loads(store.get(reference));assert commit==reference['commit']
        destination=root/'cold-verification'/reference['sha256']
        assert not destination.exists(),'choose a fresh verification directory for cold readback'
        recovered=recover(store,reference,destination);verify_final(recovered,plan['job_id'])
        assert json.loads((recovered/'authorization.json').read_text())==report['authorization']
        from colabcola.lifecycle_provider import provider_for_plan
        assert provider_for_plan(plan).lookup(state['attempts'][0]['endpoint']) is None
        report['scope']='cold Drive bytes and exact owned-runtime absence verified'
    atomic(root/'probe-verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
