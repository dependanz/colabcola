import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
from colabcola import lifecycle_io as io
from colabcola.lifecycle import verify_final
from colabcola.lifecycle_worker import AuthorizationEvidence
from test_lifecycle import MemoryStore

class LargeArtifactTests(unittest.TestCase):
    def test_retry_reuses_verified_chunks_and_never_exposes_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'file';source.write_bytes(b'abcdefghijkl')
            store=MemoryStore();identity={'run_id':'a'*32,'job_id':'b'*64}
            with mock.patch.object(io,'CHUNK_BYTES',4):
                snap=io.publish_files(store,identity,'final',{'artifact':source})
                original=store.get;calls=[]
                def flaky(ref):
                    calls.append(ref['id'])
                    if len(calls)==2:raise io.LifecycleError('interrupted')
                    return original(ref)
                store.get=flaky
                with self.assertRaisesRegex(io.LifecycleError,'interrupted'):
                    io.recover(store,snap,root/'result')
                self.assertFalse((root/'result').exists())
                self.assertFalse(list(root.glob('result.*.partial')))
                io.recover(store,snap,root/'result')
                first=snap['commit']['members']['artifact']['chunks'][0]['id']
                self.assertEqual(calls.count(first),1)
                self.assertEqual((root/'result/artifact').read_bytes(),source.read_bytes())

    def test_growing_file_cannot_escape_publication_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'file';source.write_bytes(b'abcdefgh');store=MemoryStore()
            original=store.put
            def grow(kind,name,body):
                if kind=='member':
                    with source.open('ab') as stream:stream.write(b'more')
                return original(kind,name,body)
            store.put=grow
            with mock.patch.object(io,'CHUNK_BYTES',4),self.assertRaisesRegex(io.LifecycleError,'grew'):
                io.publish_files(store,{'run_id':'a'*32,'job_id':'b'*64},'final',{'a':source})
            self.assertFalse(store.list('final'))

    def test_probe_emits_valid_immutable_checkpoints_with_simulated_time(self):
        import importlib.util,os
        spec=importlib.util.spec_from_file_location('renewal_probe',Path(__file__).resolve().parents[1]/'examples/runtime_renewal/run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            cwd=Path.cwd();now=[0]
            try:
                os.chdir(tmp)
                with mock.patch('builtins.print'):
                    module.run(120,60,clock=lambda:now[0],sleep=lambda value:now.__setitem__(0,now[0]+value))
                summary=json.loads(Path('outputs/probe/summary.json').read_text())
                self.assertEqual(summary['checkpoints'],3)
                for path in Path('outputs/probe').glob('*.bin.json'):
                    manifest=json.loads(path.read_text());body=(path.parent/manifest['file']).read_bytes()
                    self.assertEqual(len(body),manifest['bytes']);self.assertEqual(io.digest(body),manifest['sha256'])
            finally:os.chdir(cwd)

    def test_corrupt_cached_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'file';source.write_bytes(b'abcdefgh')
            store=MemoryStore()
            with mock.patch.object(io,'CHUNK_BYTES',4):
                snap=io.publish_files(store,{'run_id':'a'*32,'job_id':'b'*64},'final',{'a':source})
                first=snap['commit']['members']['a']['chunks'][0]
                cache=root/'.objects/.chunks';cache.mkdir(parents=True)
                (cache/first['sha256']).write_bytes(b'evil')
                with self.assertRaisesRegex(io.LifecycleError,'integrity'):
                    io.recover(store,snap,root/'result')
                self.assertFalse((root/'result').exists())

    def test_oversized_snapshot_and_disk_shortfall_fail_before_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'file';source.write_bytes(b'abcd');store=MemoryStore()
            snap=io.publish_files(store,{'run_id':'a'*32,'job_id':'b'*64},'final',{'a':source})
            store.get=mock.Mock(side_effect=AssertionError('must not transfer'))
            with mock.patch.object(io,'SNAPSHOT_BYTES',3),self.assertRaises(io.LifecycleError):
                io.recover(store,snap,root/'too-large')
            with mock.patch('shutil.disk_usage',return_value=mock.Mock(free=0)),self.assertRaisesRegex(io.LifecycleError,'disk'):
                io.recover(store,snap,root/'no-space')
            store.get.assert_not_called()

    def test_artifact_allowance_excludes_result_metadata_but_rejects_larger_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body=b'abcd'
            result={'job_id':'id','outputs':[{'path':'a','size':4,'sha256':io.digest(body)}]}
            raw=io.encoded(result);(root/'result.json').write_bytes(raw)
            with zipfile.ZipFile(root/'artifacts.zip','w') as archive:
                archive.writestr('result.json',raw);archive.writestr('a',body)
            with mock.patch('colabcola.lifecycle.ARTIFACT_BYTES',4):self.assertEqual(verify_final(root,'id'),result)
            with mock.patch('colabcola.lifecycle.ARTIFACT_BYTES',3),self.assertRaises(io.LifecycleError):verify_final(root,'id')

    def test_renewal_evidence_requires_real_token_change_and_post_expiry_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);now=[100];e=AuthorizationEvidence(clock=lambda:now[0])
            e.used({'access_token':'private-first','expires_at':200});e.published(root,'checkpoint')
            now[0]=400;e.used({'access_token':'private-first','expires_at':500});e.published(root,'logs')
            self.assertNotIn('first_verified_rotated_publication_at',e.value)
            e.used({'access_token':'private-second','expires_at':600});e.published(root,'checkpoint')
            public=(root/'authorization.json').read_text()
            self.assertNotIn('private',public)
            self.assertEqual(e.value['token_changes'],1)
            self.assertEqual(e.value['verified_after_initial_expiry_at'],400)

if __name__=='__main__':unittest.main()
