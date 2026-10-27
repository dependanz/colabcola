"""Offline real-size ZIP64/transport acceptance; no credentials, network or cloud."""
import argparse,contextlib,hashlib,json,os,resource,shutil,sys,tempfile,time,zipfile
from pathlib import Path
from colabcola.packager import parser,package
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final
from colabcola.lifecycle_io import ARTIFACT_BYTES,CHUNK_BYTES,digest,encoded,publish_files,recover,LifecycleError

class DiskStore:
    def __init__(self,root):self.root=root;root.mkdir();self.calls=[];self.interrupt=False;self.maximum_body=0
    def put(self,kind,name,body):
        assert len(body)<=CHUNK_BYTES
        self.maximum_body=max(self.maximum_body,len(body));sha=digest(body);p=self.root/sha
        if not p.exists():p.write_bytes(body)
        assert p.read_bytes()==body
        return {'id':sha,'sha256':sha,'bytes':len(body)}
    def get(self,ref):
        self.calls.append(ref['id'])
        if self.interrupt and len(self.calls)==2:raise LifecycleError('injected interrupted download')
        body=(self.root/ref['id']).read_bytes()
        assert len(body)==ref['bytes'] and digest(body)==ref['sha256']
        return body

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(out).free>20*1024**3
    # This would reject the former whole-file read, independently of sampled RSS.
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='large-proof-',dir=out) as temporary:
        root=Path(temporary).resolve();root.relative_to(out)
        source=root/'work.py';source.write_text("from pathlib import Path\np=Path('outputs/large.bin');p.parent.mkdir()\nwith p.open('wb') as f:f.truncate(5*1024**3)\n")
        job=root/'large.colajob'
        package(parser().parse_args(['-o',str(job),'--artifact','outputs/large.bin','--artifact-max-bytes',str(ARTIFACT_BYTES),str(source)]))
        compiled=compile_invocation(load_job_invocation(job,()),root/'notebook.ipynb')
        namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(root/'content')}
        with (out/'compiled-runtime.log').open('w') as log,contextlib.redirect_stdout(log):
            for cell in json.loads((root/'notebook.ipynb').read_text())['cells']:
                if cell['cell_type']=='code':exec(cell['source'],namespace)
        result_root=root/'content/colabcola/results'/compiled.job_id
        result=verify_final(result_root,compiled.job_id)
        assert result['terminal_state']=='success' and result['outputs'][0]['size']==ARTIFACT_BYTES
        # Use stored ZIP64 to exercise a real >5 GiB transfer, even though zeros compress well.
        transfer=root/'transfer';transfer.mkdir();raw=(result_root/'result.json').read_bytes();(transfer/'result.json').write_bytes(raw)
        data=root/'content/colabcola/jobs'/compiled.job_id/'outputs/large.bin'
        with zipfile.ZipFile(transfer/'artifacts.zip','w',allowZip64=True) as archive:
            archive.writestr('result.json',raw)
            info=zipfile.ZipInfo('outputs/large.bin');info.file_size=ARTIFACT_BYTES
            with archive.open(info,'w',force_zip64=True) as target,data.open('rb') as src:shutil.copyfileobj(src,target,CHUNK_BYTES)
        store=DiskStore(root/'store')
        snap=publish_files(store,{'run_id':'a'*32,'job_id':compiled.job_id},'final',{'result.json':transfer/'result.json','artifacts.zip':transfer/'artifacts.zip'})
        store.interrupt=True
        try:recover(store,snap,root/'recovered');raise AssertionError('expected interruption')
        except LifecycleError as exc:assert 'injected interrupted' in str(exc)
        assert not (root/'recovered').exists()
        store.interrupt=False;recovered=recover(store,snap,root/'recovered')
        assert verify_final(recovered,compiled.job_id)==result
        first=snap['commit']['members']['artifacts.zip']['chunks'][0]['id'];assert store.calls.count(first)==1
        with zipfile.ZipFile(recovered/'artifacts.zip') as archive:
            info=archive.getinfo('outputs/large.bin');assert info.extract_version>=45 and info.file_size==ARTIFACT_BYTES
        report={'scope':'offline real-size test, no cloud','output_bytes':ARTIFACT_BYTES,
            'stored_zip_bytes':(transfer/'artifacts.zip').stat().st_size,'max_transfer_body_bytes':store.maximum_body,
            'zip_chunk_count':len(snap['commit']['members']['artifacts.zip']['chunks']),
            'memory_address_space_limit_bytes':768*1024**2,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'interrupted_chunk_reused':True,'compiled_runtime_streaming_passed':True,'zip64_and_final_hashes_verified':True,
            'elapsed_seconds':time.monotonic()-started,'large_temporary_files_retained':False}
    (out/'large-artifact-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
