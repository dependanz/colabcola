"""Compiled execution acceptance for ordinary scripts/modules/sequences; no cloud calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from colabcola.cli import main
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final
from colabcola.bundle import build_bundle
from colabcola.packager import package, parser
from colabcola.spec import JobSpec
from colabcola.errors import SpecError


class PackagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project=self.root/'project';self.project.mkdir()

    def make(self,*args,name='test'):
        output=self.root/(name+'.colajob')
        result=package(parser().parse_args(['--output',str(output),*args]))
        self.assertEqual(result['state'],'packaged')
        return output

    def execute(self,job,expected='success'):
        first=compile_invocation(load_job_invocation(job),job.with_suffix('.ipynb'))
        second=compile_invocation(load_job_invocation(job),job.with_suffix('.repeat.ipynb'))
        self.assertEqual(first.notebook_bytes,second.notebook_bytes)
        runtime=self.root/(job.stem+'-runtime')
        namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(runtime)}
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            for cell in json.loads(first.notebook_bytes)['cells']:
                if cell['cell_type']=='code':
                    try: exec(cell['source'],namespace)
                    except RuntimeError:
                        if expected=='success': raise
        result_dir=runtime/'colabcola/results'/first.job_id
        result=verify_final(result_dir,first.job_id)
        self.assertEqual(result['terminal_state'],expected,result.get('failure'))
        return runtime/'colabcola/jobs'/first.job_id,result

    def test_standalone_is_frozen_and_forwards_literal_arguments(self):
        script=self.project/'run.py'
        script.write_text('import json,sys\nfrom pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/args.json").write_text(json.dumps(sys.argv[1:]))\n',encoding='utf-8')
        args=['space here','"quoted"','$HOME','; echo injected','--output','literal']
        job=self.make('--artifact','outputs/args.json',str(script),*args)
        before=build_bundle(load_job_invocation(job)).job_id
        script.write_text('raise RuntimeError("the source changed")',encoding='utf-8')
        self.assertEqual(before,build_bundle(load_job_invocation(job)).job_id)
        work,result=self.execute(job)
        self.assertEqual(json.loads((work/'outputs/args.json').read_text()),args)
        self.assertEqual(len(result['outputs']),1)

    def test_module_with_project_data_relative_import_and_exclusions(self):
        pkg=self.project/'model';pkg.mkdir()
        (pkg/'__init__.py').write_text('')
        (pkg/'helper.py').write_text('VALUE = 17')
        (pkg/'train.py').write_text('from .helper import VALUE\nfrom pathlib import Path\np=Path("outputs");p.mkdir()\n(p/"model.txt").write_text(str(VALUE)+Path("data/input.txt").read_text())')
        (self.project/'.env').write_text('DO_NOT_PACKAGE=secret')
        (self.project/'.venv').mkdir();(self.project/'.venv/ignored.py').write_text('ignore')
        (self.project/'omit.txt').write_text('ignore')
        data=self.root/'input.txt';data.write_text(' data')
        job=self.make('--project',str(self.project),'--exclude','omit.txt','--data',str(data)+'=data/input.txt','--artifact','outputs/model.txt','--','python','-m','model.train')
        names={f.path for f in build_bundle(load_job_invocation(job)).files}
        self.assertNotIn('.env',names);self.assertNotIn('.venv/ignored.py',names);self.assertNotIn('omit.txt',names)
        work,_=self.execute(job);self.assertEqual((work/'outputs/model.txt').read_text(),'17 data')

    def test_train_evaluate_sequence_uses_fresh_processes_and_stops_on_failure(self):
        pkg=self.project/'model';pkg.mkdir();(pkg/'__init__.py').write_text('')
        (pkg/'train.py').write_text('from pathlib import Path\nimport os\nPath("outputs").mkdir()\nPath("outputs/model.txt").write_text("42")\nos.environ["ONLY_TRAIN"]="yes"')
        (pkg/'evaluate.py').write_text('from pathlib import Path\nimport os\nassert "ONLY_TRAIN" not in os.environ\nassert Path("outputs/model.txt").read_text()=="42"\nPath("outputs/score.txt").write_text("passed")')
        commands=self.root/'commands.json';commands.write_text(json.dumps([['python','-m','model.train'],['python','-m','model.evaluate']]))
        job=self.make('--project',str(self.project),'--commands',str(commands),'--artifact','outputs/score.txt')
        work,_=self.execute(job);self.assertEqual((work/'outputs/score.txt').read_text(),'passed')
        (pkg/'train.py').write_text('raise SystemExit(7)')
        failed=self.make('--project',str(self.project),'--commands',str(commands),name='failed')
        work,result=self.execute(failed,'job_failed')
        self.assertFalse((work/'outputs/score.txt').exists())
        self.assertEqual(result['process_exit_code'],7)

    def test_dependencies_parse_ordinary_and_hashed_without_installing_locally(self):
        script=self.project/'run.py';script.write_text('pass')
        req=self.root/'requirements.txt';req.write_text('numpy>=2.1\npillow==11.3.0 # comment\nfoo==1 --hash=sha256:'+'a'*64+'\n')
        with mock.patch('subprocess.run',side_effect=AssertionError('packaging cannot run commands')):
            job=self.make('--requirements',str(req),str(script))
        spec=load_job_invocation(job).spec
        self.assertEqual(spec.python.pip_requirements,('numpy>=2.1','pillow==11.3.0'))
        self.assertEqual(len(spec.python.install_requirements),1)
        manifest=json.loads(build_bundle(load_job_invocation(job)).canonical_manifest_bytes)
        self.assertEqual(manifest['python']['pip_requirements'],['numpy>=2.1','pillow==11.3.0'])
        req.write_text('-r private.txt')
        with self.assertRaises(SpecError):self.make('--requirements',str(req),str(script),name='invalid')
        self.assertFalse((self.root/'invalid.colajob').exists())

    def test_embedded_wheel_and_remote_manifest_are_packaged_offline(self):
        script=self.project/'run.py';script.write_text('pass')
        wheel=self.root/'example-1.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel,'w') as archive:
            archive.writestr('example.py','VALUE=42')
        records=[{'target':f'weights/{i}.bin','url':f'https://example.com/{i}.bin',
                  'bytes':7,'sha256':'a'*64} for i in range(110)]
        manifest=self.root/'remote.json';manifest.write_text(json.dumps(records))
        with mock.patch('subprocess.run',side_effect=AssertionError('no host execution')), \
                mock.patch('urllib.request.urlopen',side_effect=AssertionError('no remote fetch')):
            job=self.make('--wheel',str(wheel),'--remote-inputs',str(manifest),str(script))
            compiled=compile_invocation(load_job_invocation(job),self.root/'preview.ipynb')
        spec=load_job_invocation(job).spec
        bundle=build_bundle(load_job_invocation(job))
        frozen=json.loads(bundle.canonical_manifest_bytes)
        self.assertEqual(spec.python.embedded_wheels,('wheels/'+wheel.name,))
        self.assertEqual(frozen['remote_inputs'],sorted(records,key=lambda item:item['target']))
        self.assertEqual(next(x.role for x in bundle.files if x.path=='wheels/'+wheel.name),'wheel')
        self.assertGreaterEqual(spec.limits.maximum_file_count,len(bundle.files)+len(records))
        self.assertFalse(any(x.path.startswith('weights/') for x in bundle.files))
        original=bundle.job_id;wheel.write_bytes(b'changed source')
        self.assertEqual(build_bundle(load_job_invocation(job)).job_id,original)

    def test_remote_manifest_validation_and_collisions_leave_no_job(self):
        script=self.project/'run.py';script.write_text('pass')
        good={'target':'weights/a.bin','url':'https://example.com/a.bin','bytes':7,'sha256':'a'*64}
        for i,records in enumerate([{},[{**good,'sha256':'bad'}],[{**good,'target':'../escape'}],
                                    [{**good,'target':'run.py'}],[good,good]]):
            manifest=self.root/'remote.json';manifest.write_text(json.dumps(records))
            with self.assertRaises(SpecError):
                self.make('--remote-inputs',str(manifest),str(script),name='bad'+str(i))
            self.assertFalse((self.root/('bad'+str(i)+'.colajob')).exists())
            self.assertFalse((self.root/('bad'+str(i)+'.files')).exists())

    def test_wheel_requires_file_and_rejects_duplicate_payload_target(self):
        script=self.project/'run.py';script.write_text('pass')
        wheel=self.root/'sample.whl';wheel.write_bytes(b'fixture')
        with self.assertRaises(SpecError):self.make('--wheel',str(script),str(script))
        with self.assertRaises(SpecError):self.make('--wheel',str(self.project),str(script))
        with self.assertRaises(SpecError):
            self.make('--data',str(wheel)+'=wheels/sample.whl','--wheel',str(wheel),str(script))

    def test_dependency_installation_is_remote_and_records_resolved_versions(self):
        script=self.project/'run.py';script.write_text('print("ok")')
        requirements=self.root/'requirements.txt';requirements.write_text('packaging>=23')
        job=self.make('--requirements',str(requirements),str(script))
        real_run=subprocess.run;calls=[]
        def pip_only(command,**kwargs):
            if command[1:4]==['-m','pip','install']:
                calls.append(command)
                return subprocess.CompletedProcess(command,0,stdout='',stderr='')
            return real_run(command,**kwargs)
        with mock.patch('subprocess.run',side_effect=pip_only):
            _,result=self.execute(job)
        self.assertEqual(len(calls),1)
        self.assertIn('packaging>=23',calls[0])
        self.assertIn('packaging>=23',result['environment']['resolved_packages'])

    def test_resume_option_is_generic_and_sequences_reject_extra_arguments(self):
        script=self.project/'run.py';script.write_text('import sys,json\nfrom pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/args.json").write_text(json.dumps(sys.argv[1:]))')
        job=self.make('--checkpoint','outputs/state-*.json','--resume-arg=--resume',str(script))
        with mock.patch.dict(os.environ,{'COLABCOLA_RESUME_MANIFEST':'/restored/state.json'}):
            work,_=self.execute(job)
        self.assertEqual(json.loads((work/'outputs/args.json').read_text()),['--resume','/restored/state.json'])
        commands=self.root/'commands.json';commands.write_text(json.dumps([['python','run.py'],['python','run.py']]))
        sequence=self.make('--project',str(self.project),'--commands',str(commands),name='sequence')
        body=json.loads(sequence.read_text());body['entrypoint']['argv']=['ambiguous'];sequence.write_text(json.dumps(body))
        work,_=self.execute(sequence,'job_failed');self.assertFalse((work/'outputs').exists())

    def test_cpu_configuration_and_checkpoint_are_optional(self):
        script=self.project/'run.py';script.write_text('print("ordinary script")')
        helper=self.root/'helper.json';helper.write_text('["python","helper.py"]')
        with mock.patch('subprocess.run',side_effect=AssertionError('offline only')):
            job=self.make('--drive-parent','fixture','--token-command',str(helper),str(script))
        config=json.loads(job.with_suffix('.run.json').read_text())
        self.assertEqual(config['gpu'],'NONE');self.assertEqual(config['max_attempts'],1)
        self.assertIsNone(json.loads((job.with_suffix('.files')/'_colabcola/commands.json').read_text())['resume_arg'])
        self.assertNotIn('_colabcola/drive-helper.json',{f.path for f in build_bundle(load_job_invocation(job)).files})
        work,_=self.execute(job)
        self.assertTrue((work/'run.py').exists())

    def test_cli_dispatch_and_invalid_inputs_leave_no_job(self):
        script=self.project/'run.py';script.write_text('pass')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['package','-o',str(self.root/'cli.colajob'),str(script)]),0)
        for arguments in [ ['--data',str(script)+'=../escape',str(script)],
                           ['--project',str(self.project),'--exclude','run.py','run.py'],
                           ['--','python','-m','missing'], ['--','bash','run.py'] ]:
            with self.subTest(arguments=arguments),self.assertRaises(SpecError):self.make(*arguments,name='invalid')
            self.assertFalse((self.root/'invalid.colajob').exists())
        with self.assertRaises(SpecError):self.make(str(script),name='cli')

    def test_collision_and_failed_publication_leave_no_partial_package(self):
        script=self.project/'run.py';script.write_text('pass')
        with self.assertRaises(SpecError):
            self.make('--data',str(script)+'=RUN.py',str(script))
        self.assertFalse((self.root/'test.colajob').exists())
        self.assertFalse((self.root/'test.files').exists())
        helper=self.root/'helper.json';helper.write_text('["python","helper.py"]')
        original=Path.open
        def fail_config(path,*args,**kwargs):
            if path.name=='test.run.json':raise OSError('simulated publication failure')
            return original(path,*args,**kwargs)
        with mock.patch.object(Path,'open',fail_config),self.assertRaises(OSError):
            self.make('--drive-parent','fixture','--token-command',str(helper),str(script))
        self.assertFalse((self.root/'test.files').exists())
        self.assertFalse((self.root/'test.colajob').exists())
        self.assertFalse(list(self.root.glob('.colabcola-package-*')))

    @unittest.skipIf(os.name=='nt','POSIX process-group timeout is verified on WSL')
    def test_timeout_kills_sequence_children(self):
        script=self.project/'slow.py'
        script.write_text('import time\nfrom pathlib import Path\ntime.sleep(5)\nPath("escaped.txt").write_text("bad")')
        job=self.make('--timeout','1',str(script))
        work,_=self.execute(job,'job_failed')
        import time;time.sleep(5)
        self.assertFalse((work/'escaped.txt').exists())


if __name__=='__main__':unittest.main()
