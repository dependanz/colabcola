"""Acceptance of the public command-first interactive and TOML answer workflow."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest import mock

from colabcola.cli import main
from colabcola.compiler import compile_invocation
from colabcola.invocation import load_job_invocation
from colabcola.lifecycle import verify_final
from colabcola.package_guide import save_answers, _split_command


class GuidedPackagingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.project=self.root/'project';self.project.mkdir()
        before=Path.cwd();os.chdir(self.project);self.addCleanup(os.chdir,before)

    def call(self,*args,answers=()):
        out=io.StringIO();err=io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err),mock.patch('builtins.input',side_effect=answers):
            status=main(['package',*args])
        return status,out.getvalue(),err.getvalue()

    def config(self,commands,**overrides):
        data={'commands':commands,'project':str(self.project),'inputs':[],
              'environment':{'dependencies':[]},'compute':{'gpu':False},
              'outputs':{'include':['outputs/**'],'checkpoints':[]},'limits':{'timeout_seconds':60}}
        data.update(overrides)
        path=self.root/'answers.colaml';save_answers(path,data)
        return path

    def execute(self,job):
        first=compile_invocation(load_job_invocation(job),self.root/'first.ipynb')
        second=compile_invocation(load_job_invocation(job),self.root/'second.ipynb')
        self.assertEqual(first.notebook_bytes,second.notebook_bytes)
        namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(self.root/'runtime')}
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            for cell in json.loads(first.notebook_bytes)['cells']:
                if cell['cell_type']=='code': exec(cell['source'],namespace)
        result=verify_final(self.root/'runtime/colabcola/results'/first.job_id,first.job_id)
        self.assertEqual(result['terminal_state'],'success')
        return self.root/'runtime/colabcola/jobs'/first.job_id,result

    def test_interactive_command_inspects_local_imports_maps_data_and_saves_answers(self):
        (self.project/'helper.py').write_text('VALUE=42')
        (self.project/'experiment.py').write_text('from helper import VALUE\nfrom pathlib import Path\nimport json,sys\nPath("outputs").mkdir()\nPath("outputs/result.json").write_text(json.dumps([VALUE,Path(sys.argv[1]).read_text(),sys.argv[2:]]))')
        data=self.root/'source data.txt';data.write_text('real input')
        secret=self.project/'.env';secret.write_text('NEVER_INCLUDE=private')
        saved=self.root/'saved.colaml'
        argv=['--','python','experiment.py',str(data),'space here','$HOME','; echo literal']
        status,out,err=self.call(*argv,answers=[str(data)+'=data/input.txt','NONE','outputs/**','none','1','y',str(saved)])
        self.assertEqual(status,0,err)
        result=json.loads(out);self.assertEqual(len(result['questions_asked']),7)
        self.assertNotIn('helper',err.split('External imports:')[-1] if 'External imports:' in err else '')
        job=self.project/'experiment.colajob';spec=json.loads(job.read_text())
        self.assertNotIn('.env',{p['target'] for p in spec['payload']})
        original=hashlib.sha256(job.read_bytes()).hexdigest()
        # Packaging has not executed the user's script or created its outputs.
        self.assertFalse((self.project/'outputs').exists())
        (self.project/'experiment.py').write_text('raise RuntimeError("changed")')
        work,report=self.execute(job)
        self.assertEqual(json.loads((work/'outputs/result.json').read_text()),[42,'real input',['space here','$HOME','; echo literal']])
        self.assertEqual(hashlib.sha256(job.read_bytes()).hexdigest(),original)
        answers=tomllib.loads(saved.read_text());self.assertEqual(answers['commands'][0][2],'data/input.txt')
        self.assertEqual(answers['environment']['dependencies'],[])

    def test_colaml_and_legacy_answers_alias_produce_the_same_job(self):
        (self.project/'train.py').write_text('pass')
        cfg=self.config([['python','train.py']])
        ids=[]
        for flag,save_flag,stem in [('--colaml','--save-colaml','new'),('--answers','--save-answers','legacy')]:
            saved=self.root/(stem+'.colaml')
            status,out,err=self.call(flag,str(cfg),'--non-interactive','--name','alias-test',
                '--output',stem+'.colajob',save_flag,str(saved))
            self.assertEqual(status,0,err)
            ids.append(json.loads(out)['job_id'])
            self.assertTrue(saved.is_file())
        self.assertEqual(ids[0],ids[1])
        resolved=[tomllib.loads((self.root/(stem+'.colaml')).read_text()) for stem in ('new','legacy')]
        for value in resolved: value.pop('output')  # Each invocation requested its own destination.
        self.assertEqual(resolved[0],resolved[1])
        from colabcola.packager import parser
        self.assertIn('--colaml COLAML',parser().format_help())

    def test_answer_paths_relative_to_file_and_cli_overrides(self):
        (self.project/'train.py').write_text('pass')
        (self.root/'input.txt').write_text('payload')
        (self.root/'requirements.lock').write_text('example==1 --hash=sha256:'+'a'*64+'\n')
        answers=self.config([['python','train.py']],project='project',
            inputs=[{'source':'input.txt','target':'data/input.txt'}],
            environment={'requirements':'requirements.lock'},compute={'gpu':'A100','high_mem':True},
            outputs={'include':['outputs/**'],'checkpoints':[],'maximum_bytes':5*1024**3})
        status,out,err=self.call('--colaml',str(answers),'--non-interactive','--output','override.colajob','--gpu','T4','--timeout','120')
        self.assertEqual(status,0,err);self.assertEqual(json.loads(out)['questions_asked'],[])
        spec=json.loads((self.project/'override.colajob').read_text())
        self.assertEqual(spec['run']['timeout_seconds'],120)
        self.assertEqual(len(spec['python']['install_requirements']),1)
        self.assertEqual(spec['outputs'][0]['maximum_bytes'],5*1024**3)
        self.assertIn('data/input.txt',{p['target'] for p in spec['payload']})
        self.assertNotIn('answers.colaml',{p['target'] for p in spec['payload']})
        self.assertEqual(json.loads(out)['requested_gpu'],'T4')

    def test_module_src_layout_and_ordered_command_strings_execute(self):
        pkg=self.project/'src/model';pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('')
        (pkg/'train.py').write_text('from pathlib import Path\nPath("outputs").mkdir()\nPath("outputs/model.txt").write_text("42")')
        (pkg/'evaluate.py').write_text('from pathlib import Path\nassert Path("outputs/model.txt").read_text()=="42"\nPath("outputs/score.txt").write_text("passed")')
        cfg=self.config([['python','unused.py']])
        status,out,err=self.call('--colaml',str(cfg),'--non-interactive','--output','pipeline.colajob',
            '--command','python -m model.train','--command','python -m model.evaluate')
        self.assertEqual(status,0,err)
        job=self.project/'pipeline.colajob';spec=json.loads(job.read_text())
        self.assertEqual(spec['run']['environment']['PYTHONPATH'],'src')
        work,_=self.execute(job);self.assertEqual((work/'outputs/score.txt').read_text(),'passed')

    def test_console_entrypoint_keeps_argv_and_runs_in_workload_environment(self):
        # pip is a real console script installed alongside the test interpreter.
        cfg=self.config([['pip','--version']],outputs={'include':[],'checkpoints':[]})
        with mock.patch('subprocess.run',side_effect=AssertionError('packaging executed code')),mock.patch('urllib.request.urlopen',side_effect=AssertionError('network')):
            status,out,err=self.call('--colaml',str(cfg),'--non-interactive','-o','console.colajob')
        self.assertEqual(status,0,err)
        command=json.loads((self.project/'console.files/_colabcola/commands.json').read_text())
        self.assertEqual(command['console_commands'],[True]);self.assertEqual(command['commands'],[['pip','--version']])
        self.execute(self.project/'console.colajob')

    def test_static_dependencies_and_pyproject_metadata_without_importing_workload(self):
        (self.project/'train.py').write_text('import PIL, numpy\nraise RuntimeError("do not import")')
        cfg=self.config([['python','train.py']],environment={})
        status,out,err=self.call('--colaml',str(cfg),'--output','inspect.colajob',answers=['pillow==11.3.0, numpy>=2','y','none'])
        self.assertEqual(status,0,err)
        spec=json.loads((self.project/'inspect.colajob').read_text())
        self.assertEqual(spec['python']['pip_requirements'],['pillow==11.3.0','numpy>=2'])
        (self.project/'pyproject.toml').write_text('[project]\nname="fixture"\ndependencies=["pillow>=11"]\nrequires-python=">=3.12"')
        status,out,err=self.call('--colaml',str(cfg),'--non-interactive','-o','metadata.colajob')
        self.assertEqual(status,0,err)
        spec=json.loads((self.project/'metadata.colajob').read_text())
        self.assertEqual(spec['python']['requires_python'],'>=3.12');self.assertEqual(spec['python']['pip_requirements'],['pillow>=11'])

    def test_incomplete_noninteractive_and_invalid_toml_do_not_create_job(self):
        (self.project/'train.py').write_text('pass')
        status,out,err=self.call('--non-interactive','--','python','train.py')
        self.assertEqual(status,2);self.assertIn('missing answer',err)
        for i,text in enumerate(['[compute]\ngup="T4"','inputs="oops"','commands=["python","train.py"]','[limits]\ntimeout_minutes=true','[compute]\ngpu=4','[environment]\nenv={ X=1 }','malformed=[']):
            cfg=self.root/f'bad{i}.colaml';cfg.write_text(text)
            status,out,err=self.call('--colaml',str(cfg),'--non-interactive','-o','bad.colajob','--','python','train.py')
            self.assertEqual(status,2,err);self.assertFalse((self.project/'bad.colajob').exists());self.assertFalse((self.project/'bad.files').exists())

    def test_saved_answers_replay_from_another_cwd_without_questions(self):
        (self.project/'train.py').write_text('pass')
        cfg=self.config([['python','train.py']])
        saved=self.root/'resolved.colaml'
        status,out,err=self.call('--colaml',str(cfg),'--non-interactive','--save-colaml',str(saved),'-o','first.colajob')
        self.assertEqual(status,0,err)
        os.chdir(self.root)
        status,out,err=self.call('--colaml',str(saved),'--non-interactive','-o','second.colajob')
        self.assertEqual(status,0,err)
        first=load_job_invocation(self.project/'first.colajob');second=load_job_invocation(self.root/'second.colajob')
        from colabcola.bundle import build_bundle
        self.assertEqual(build_bundle(first).job_id,build_bundle(second).job_id)

    def test_windows_command_string_preserves_backslashes_quotes_and_no_shell(self):
        self.assertEqual(_split_command('python "scripts\\my train.py" --name "space here"'),['python','scripts\\my train.py','--name','space here'])
        with self.assertRaises(ValueError): _split_command('python "unterminated')
        from colabcola.errors import SpecError
        with self.assertRaises(SpecError): _split_command('python train.py && python evaluate.py')

    def test_cancellation_and_existing_answer_destination_leave_no_package(self):
        (self.project/'train.py').write_text('pass');cfg=self.config([['python','train.py']])
        status,out,err=self.call('--colaml',str(cfg),'-o','cancelled.colajob',answers=['n'])
        self.assertEqual(status,2);self.assertFalse((self.project/'cancelled.colajob').exists())
        status,out,err=self.call('--colaml',str(cfg),'--non-interactive','--save-colaml',str(cfg),'-o','exists.colajob')
        self.assertEqual(status,2);self.assertIn('destination exists',err);self.assertFalse((self.project/'exists.colajob').exists())

if __name__=='__main__': unittest.main()
