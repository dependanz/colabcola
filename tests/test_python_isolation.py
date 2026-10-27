import copy,hashlib,io,json,os,platform,tarfile,tempfile,unittest,zipfile
from pathlib import Path
from unittest import mock
from colabcola.spec import JobSpec
from colabcola.errors import SpecError
from colabcola.bundle import build_bundle
from colabcola.invocation import bind_job_arguments
from colabcola.python_profiles import PYTHON_312
from colabcola.isolation_runtime import _colabcola_extract_python, _colabcola_isolated_env, _colabcola_prepare_python
from test_notebook import notebook_job

class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'run.py';self.source.write_text('print("ok")')
        self.mapping=notebook_job(self.source);self.mapping['python']['isolation']=dict(PYTHON_312)

    def test_schema_identity_and_compatibility(self):
        valid=JobSpec.from_mapping(self.mapping)
        first=build_bundle(bind_job_arguments(valid))
        self.assertEqual(json.loads(first.canonical_manifest_bytes)['python']['isolation'],PYTHON_312)
        alternate=copy.deepcopy(self.mapping);alternate['python']['isolation']['sha256']='a'*64
        self.assertNotEqual(first.job_id,build_bundle(bind_job_arguments(JobSpec.from_mapping(alternate))).job_id)
        for key,value in [('bytes',True),('bytes',268435457),('sha256','broken'),('version','3.13.1'),('url','http://example.org/python.tar.gz')]:
            invalid=copy.deepcopy(self.mapping);invalid['python']['isolation'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(SpecError):JobSpec.from_mapping(invalid)
        invalid=copy.deepcopy(self.mapping);invalid['python']['requires_python']='>=3.13'
        with self.assertRaisesRegex(SpecError,'does not satisfy'):JobSpec.from_mapping(invalid)
        invalid=copy.deepcopy(self.mapping);invalid['run']['environment']['PYTHONPATH']='/host/site-packages'
        with self.assertRaisesRegex(SpecError,'environment overrides'):JobSpec.from_mapping(invalid)

    def test_compiler_pair_selection_and_ambient_override_removal(self):
        from colabcola.isolation_runtime import _colabcola_build_environment
        for available, family in [({'gcc':'/usr/bin/gcc','g++':'/usr/bin/g++','clang':'/usr/bin/clang','clang++':'/usr/bin/clang++'},'gcc'),
                                  ({'gcc':'/usr/bin/gcc','clang':'/usr/bin/clang','clang++':'/usr/bin/clang++'},'clang'),
                                  ({'gcc':'/usr/bin/gcc'},None)]:
            with self.subTest(family=family), mock.patch.dict(os.environ,{'CC':'missing-cc','CXX':'missing-cxx','LDSHARED':'missing-linker'}), mock.patch('colabcola.isolation_runtime.shutil.which',side_effect=lambda name,**kw:available.get(name)):
                env,facts=_colabcola_build_environment('/job/venv/bin/python')
            if family:
                self.assertEqual(facts['family'],family)
                self.assertEqual(env['CC'],facts['cc'])
                self.assertEqual(env['CXX'],facts['cxx'])
                self.assertEqual(env['LDSHARED'],facts['cc']+' -shared')
            else:
                self.assertEqual(facts['state'],'unavailable')
                self.assertNotIn('CC',env);self.assertNotIn('CXX',env);self.assertNotIn('LDSHARED',env)

    def test_environment_cannot_leak_host_packages_or_pip_destination(self):
        with mock.patch.dict(os.environ,{'PYTHONPATH':'/host','PYTHONHOME':'/host','PIP_TARGET':'/host','PIP_USER':'1','VIRTUAL_ENV':'/host'}):
            env=_colabcola_isolated_env('/job/venv/bin/python',{'PYTHONUNBUFFERED':'1'})
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('PYTHONHOME',env);self.assertNotIn('PIP_TARGET',env)
        self.assertNotIn('PIP_USER',env);self.assertEqual(env['VIRTUAL_ENV'],str(Path('/job/venv')))
        self.assertTrue(env['PATH'].startswith(str(Path('/job/venv/bin'))+os.pathsep))
        with self.assertRaisesRegex(RuntimeError,'forbids'):_colabcola_isolated_env(declared={'PIP_TARGET':'/escape'})

    def test_tar_rejects_traversal_special_files_duplicates_and_escaping_links(self):
        for name,kind,link in [('../escaped','file',''),('/absolute','file',''),('python/a','link','../../escape'),('python/a','fifo',''),('python/a','hard','python/b'),('python/a','duplicate','')]:
            with self.subTest(name=name,kind=kind):
                archive=self.root/'python.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    item=tarfile.TarInfo(name)
                    if kind=='link':item.type=tarfile.SYMTYPE;item.linkname=link
                    if kind=='hard':item.type=tarfile.LNKTYPE;item.linkname=link
                    if kind=='fifo':item.type=tarfile.FIFOTYPE
                    tar.addfile(item)
                    if kind=='duplicate':tar.addfile(item)
                with self.assertRaises(RuntimeError):_colabcola_extract_python(archive,self.root/'extract')
        self.assertFalse((self.root.parent/'escaped').exists())

    @unittest.skipUnless(platform.system()=='Linux','standalone target is Linux')
    def test_hash_rejection_precedes_extraction_and_execution(self):
        archive=self.root/'invalid.tar.gz';archive.write_bytes(b'bad')
        config={**PYTHON_312,'bytes':3}
        with mock.patch('colabcola.isolation_runtime.subprocess.run',side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(RuntimeError,'SHA-256'):
                _colabcola_prepare_python(config,self.root/'envs',archive_source=archive)
        self.assertEqual(list((self.root/'envs').iterdir()),[])


@unittest.skipUnless(platform.system()=='Linux' and os.environ.get('COLABCOLA_ISOLATION_ARCHIVE'), 'opt-in real Linux standalone archive rehearsal')
class CompiledIsolationTests(unittest.TestCase):
    def test_compiled_sequence_uses_isolated_install_probe_and_children(self):
        from colabcola.packager import package,parser
        from colabcola.compiler import compile_invocation
        from colabcola.invocation import load_job_invocation
        from colabcola.lifecycle import verify_final
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);project=root/'project';project.mkdir();(project/'ordinary').mkdir()
            (project/'ordinary/__init__.py').write_text('')
            code='''import json,sys,importlib.util
from pathlib import Path
import isolation_fixture
assert isolation_fixture.VALUE==42
assert sys.version_info[:2]==(3,12)
assert importlib.util.find_spec("host_only_poison") is None
Path("outputs").mkdir(exist_ok=True)
Path("outputs/train.json").write_text(json.dumps({"python":sys.version,"prefix":sys.prefix,"executable":sys.executable}))
'''
            (project/'ordinary/train.py').write_text(code)
            (project/'ordinary/evaluate.py').write_text('import json,sys\nfrom pathlib import Path\nx=json.loads(Path("outputs/train.json").read_text());assert x["executable"]==sys.executable\nPath("outputs/eval.json").write_text(json.dumps(x))\n')
            commands=root/'commands.json';commands.write_text(json.dumps([['python','-m','ordinary.train'],['python','-m','ordinary.evaluate']]))
            output=root/'sequence.colajob'
            package(parser().parse_args(['--output',str(output),'--project',str(project),'--commands',str(commands),'--isolated-python','3.12','--python','>=3.12,<3.13','--artifact','outputs/**']))
            wheel=root/'isolation_fixture-1.0-py3-none-any.whl'
            with zipfile.ZipFile(wheel,'w') as z:
                z.writestr('isolation_fixture.py','VALUE=42\n')
                z.writestr('isolation_fixture-1.0.dist-info/METADATA','Metadata-Version: 2.1\nName: isolation-fixture\nVersion: 1.0\nRequires-Python: >=3.12\n')
                z.writestr('isolation_fixture-1.0.dist-info/WHEEL','Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
                z.writestr('isolation_fixture-1.0.dist-info/RECORD','')
            job=json.loads(output.read_text());job['payload'].append({'role':'wheel','source':str(wheel),'target':'wheels/'+wheel.name})
            job['python']['embedded_wheels']=['wheels/'+wheel.name];job['python']['pip_requirements']=['isolation-fixture==1.0']
            output.write_text(json.dumps(job));compiled=compile_invocation(load_job_invocation(output),root/'sequence.ipynb')
            second=compile_invocation(load_job_invocation(output),root/'again.ipynb');self.assertEqual(compiled.notebook_bytes,second.notebook_bytes)
            poison=root/'poison';poison.mkdir();(poison/'host_only_poison.py').write_text('raise RuntimeError("host leaked")')
            namespace={'COLABCOLA_CONTENT_ROOT_OVERRIDE':str(root/'content')}
            archive=os.environ['COLABCOLA_ISOLATION_ARCHIVE']
            with mock.patch('urllib.request.urlopen',side_effect=lambda *a,**k:open(archive,'rb')), mock.patch.dict(os.environ,{'PYTHONPATH':str(poison),'PIP_TARGET':str(poison)}), mock.patch('platform.python_version',return_value='3.13.16'):
                for cell in json.loads(compiled.notebook_bytes)['cells']:
                    if cell['cell_type']=='code':exec(cell['source'],namespace)
            result=verify_final(root/'content/colabcola/results'/compiled.job_id,compiled.job_id)
            self.assertEqual(result['terminal_state'],'success',result.get('failure'))
            self.assertEqual(result['runtime']['python'],'3.13.16')
            self.assertEqual(result['environment']['isolation']['python'],'3.12.13')
            self.assertEqual(result['environment']['resolved_packages']['isolation-fixture==1.0'],'1.0')
            self.assertEqual(len(result['outputs']),2)

if __name__=='__main__':unittest.main()
