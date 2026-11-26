"""Offline, frozen private input declarations (no live account or provider)."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from types import SimpleNamespace
from unittest import mock
import zipfile

from colabcola.bundle import build_bundle
from colabcola.cli import main
from colabcola.errors import BundleError, SpecError
from colabcola.invocation import bind_job_arguments, load_job_invocation
from colabcola.notebook import build_notebook
from colabcola.package_guide import save_answers
from colabcola.private_inputs import freeze_source, PRIVATE_INPUT_TOTAL_BYTES
from colabcola import private_inputs
from colabcola.spec import JobSpec
from colabcola.transport import build_staged_bundle


class PrivatePackagingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.project = self.root / 'project'
        self.project.mkdir()
        (self.project / 'check.py').write_text('pass\n')
        self.asset = self.project / 'model.bin'
        self.asset.write_bytes(b'private-test-data-329x')
        self.target = 'weights/model.bin'
        self.recipe = self.root / 'recipe.colaml'
        self.answers = {
            'name': 'private-test', 'project': 'project', 'inputs': [],
            'commands': [['python', 'check.py', str(self.asset)]],
            'private_inputs': [{'source': 'project/model.bin', 'target': self.target}],
            'environment': {'dependencies': []}, 'compute': {'gpu': 'NONE'},
            'outputs': {'include': ['outputs/**'], 'checkpoints': []},
            'limits': {'timeout_seconds': 60, 'max_seconds': 900, 'max_runtimes': 1},
        }
        save_answers(self.recipe, self.answers)

    def package(self, name, *extra, recipe=None, expected=0):
        output = self.root / (name + '.colajob')
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch('urllib.request.urlopen', side_effect=AssertionError('network')), \
                mock.patch('subprocess.run', side_effect=AssertionError('execution')):
            code = main(['package', '--colaml', str(recipe or self.recipe), '--output', str(output),
                         '--non-interactive', *extra])
        self.assertEqual(code, expected, err.getvalue() + out.getvalue())
        if expected == 0:
            self.assertEqual(json.loads(out.getvalue())['private_input_bytes'], self.asset.stat().st_size)
            return output
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.files').exists())
        return err.getvalue()

    def manifest(self, job):
        return build_bundle(load_job_invocation(job))

    def test_private_content_excluded_from_bundle_source_binding_local_and_replay(self):
        saved = self.root / 'saved.colaml'
        job = self.package('first', '--save-colaml', str(saved))
        bundle = self.manifest(job)
        m = json.loads(bundle.canonical_manifest_bytes)
        self.assertEqual(m['private_inputs'], [{'target': self.target, **freeze_source(self.asset)}])
        self.assertNotIn(str(self.asset).encode(), bundle.canonical_manifest_bytes)
        self.assertEqual(json.loads(job.read_text())['private_inputs'][0]['source'], str(self.asset))
        with zipfile.ZipFile(io.BytesIO(bundle.archive_bytes)) as archive:
            self.assertNotIn('model.bin', archive.namelist())
            self.assertNotIn(self.target, archive.namelist())
            for name in archive.namelist():
                self.assertNotIn(self.asset.read_bytes(), archive.read(name))
            commands = json.loads(archive.read('_colabcola/commands.json'))
            self.assertEqual(commands['commands'][0][-1], self.target)
        self.assertEqual(tomllib.loads(saved.read_text())['private_inputs'][0]['source'], str(self.asset))
        self.assertEqual(self.manifest(self.package('second', recipe=saved)).job_id, bundle.job_id)
        self.asset.write_bytes(b'new content')
        self.assertEqual(self.manifest(job).job_id, bundle.job_id)
        self.assertNotEqual(self.manifest(self.package('third')).job_id, bundle.job_id)
        self.asset.unlink()
        self.assertEqual(self.manifest(job).job_id, bundle.job_id)
        for compile in (build_notebook, build_staged_bundle):
            with self.assertRaisesRegex(BundleError, 'managed acquisition'):
                compile(bundle)

    def test_windows_path_and_handle_ctime_can_differ(self):
        real = self.asset.stat()
        fields = {name: getattr(real, name) for name in (
            'st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}
        fields['st_birthtime_ns'] = 100
        before = SimpleNamespace(**fields, st_ctime_ns=100)
        opened = SimpleNamespace(**fields, st_ctime_ns=200)
        expected = {'bytes': real.st_size, 'sha256': hashlib.sha256(self.asset.read_bytes()).hexdigest()}
        with mock.patch.object(private_inputs, '_WINDOWS', True), \
                mock.patch.object(private_inputs, 'inspect_source', return_value=before), \
                mock.patch.object(private_inputs.os, 'fstat', return_value=opened):
            self.assertEqual(freeze_source(self.asset), expected)
        # POSIX must still compare change time across APIs.
        with mock.patch.object(private_inputs, '_WINDOWS', False), \
                mock.patch.object(private_inputs, 'inspect_source', return_value=before), \
                mock.patch.object(private_inputs.os, 'fstat', return_value=opened), \
                self.assertRaisesRegex(SpecError, 'changed before hashing'):
            freeze_source(self.asset)

    def test_windows_source_mutations_are_still_rejected(self):
        real = self.asset.stat()
        fields = {name: getattr(real, name) for name in (
            'st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}
        fields['st_birthtime_ns'] = 100
        before = SimpleNamespace(**fields, st_ctime_ns=100)
        opened = SimpleNamespace(**fields, st_ctime_ns=200)
        # All cross-API identity fields remain mandatory.
        for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_birthtime_ns'):
            different = SimpleNamespace(**{**vars(opened), field: getattr(opened, field) + 1})
            with self.subTest(field=field), \
                    mock.patch.object(private_inputs, '_WINDOWS', True), \
                    mock.patch.object(private_inputs, 'inspect_source', return_value=before), \
                    mock.patch.object(private_inputs.os, 'fstat', return_value=different), \
                    self.assertRaisesRegex(SpecError, 'changed before hashing'):
                freeze_source(self.asset)
        # Same-API comparisons still detect metadata-only changes to the
        # open descriptor, changes to contents/size, and path replacement.
        for api, field in (('handle', 'st_ctime_ns'), ('handle', 'st_mtime_ns'),
                           ('handle', 'st_size'), ('path', 'st_ino'), ('path', 'st_ctime_ns')):
            last_handle = SimpleNamespace(**vars(opened))
            last_path = SimpleNamespace(**vars(before))
            target = last_handle if api == 'handle' else last_path
            setattr(target, field, getattr(target, field) + 1)
            with self.subTest(api=api, field=field), \
                    mock.patch.object(private_inputs, '_WINDOWS', True), \
                    mock.patch.object(private_inputs, 'inspect_source', side_effect=[before, last_path]), \
                    mock.patch.object(private_inputs.os, 'fstat', side_effect=[opened, last_handle]), \
                    self.assertRaisesRegex(SpecError, 'changed while hashing'):
                freeze_source(self.asset)

    def test_declared_source_aliases_omitted_and_explicit_embed_rejected(self):
        alias = self.project / 'alias.bin'
        try:
            os.link(self.asset, alias)
        except OSError:
            self.skipTest('hardlinks unavailable')
        job = self.package('alias')
        self.assertNotIn('alias.bin', {x.path for x in self.manifest(job).files})
        self.package('explicit', '--data', str(alias)+'=duplicated.bin', expected=2)

    def test_collisions_and_reserved_targets(self):
        job = self.package('valid')
        value = json.loads(job.read_text())
        for target in ('check.py', 'CHECK.PY/x', '_colabcola/x', 'outputs/a', 'x/../z', 'x/CON.txt', 'x//z', 'x/foo.'):
            with self.subTest(target=target), self.assertRaises(SpecError):
                record = json.loads(json.dumps(value))
                record['private_inputs'][0]['target'] = target
                build_bundle(bind_job_arguments(JobSpec.from_mapping(record, base_dir=self.root)))
        for target in (self.target, 'weights', self.target + '/child'):
            with self.subTest(remote=target), self.assertRaises(SpecError):
                record = json.loads(json.dumps(value))
                record['remote_inputs'] = [{'target': target, 'url': 'https://example.org/x', 'bytes': 1, 'sha256': 'a'*64}]
                build_bundle(bind_job_arguments(JobSpec.from_mapping(record, base_dir=self.root)))
        value['payload'].append({'source': str(self.asset), 'target': 'another.bin', 'role': 'data'})
        with self.assertRaisesRegex(SpecError, 'embedded payload'):
            build_bundle(bind_job_arguments(JobSpec.from_mapping(value, base_dir=self.root)))

    def test_identity_bounds_and_unknown_fields(self):
        job = self.package('valid')
        value = json.loads(job.read_text())
        for patch in ({'bytes': True}, {'bytes': 0}, {'bytes': 4*1024**3+1}, {'sha256': 'invalid'}, {'url': 'https://example.org/'}):
            with self.subTest(patch=patch), self.assertRaises(SpecError):
                record = json.loads(json.dumps(value))
                record['private_inputs'][0].update(patch)
                JobSpec.from_mapping(record, base_dir=self.root)
        value['private_inputs'] = [{'source': str(self.root / f'absent-{i}'), 'target': f'assets/{i}',
                                    'bytes': PRIVATE_INPUT_TOTAL_BYTES//2, 'sha256': 'a'*64} for i in range(3)]
        with self.assertRaisesRegex(SpecError, 'total exceeds'):
            JobSpec.from_mapping(value, base_dir=self.root)

    def test_source_path_is_not_part_of_job_identity(self):
        job = self.package('before')
        a = self.manifest(job)
        v = json.loads(job.read_text())
        other = self.root / 'moved.bin'
        other.write_bytes(self.asset.read_bytes())
        v['private_inputs'][0]['source'] = str(other)
        b = build_bundle(bind_job_arguments(JobSpec.from_mapping(v, base_dir=self.root)))
        self.assertEqual(a.job_id, b.job_id)

    def test_source_safety_and_no_control_file_reuse(self):
        other = self.root / 'other.colaml'
        self.answers['private_inputs'][0]['source'] = 'project'
        save_answers(other, self.answers)
        self.package('directory', recipe=other, expected=2)
        self.package('requirements', '--requirements', str(self.asset), expected=2)
        linked = self.root / 'link.bin'
        try:
            linked.symlink_to(self.asset)
        except OSError:
            return
        with self.assertRaisesRegex(SpecError, 'link or reparse'):
            freeze_source(linked)

    def test_named_step_and_flags_map_private_source(self):
        other = self.root / 'named.colaml'
        del self.answers['commands']
        del self.answers['environment']
        self.answers['environments'] = {'train': {'python': '3.12'}}
        self.answers['steps'] = [{'name': 'check', 'environment': 'train', 'command': ['python', 'check.py', '--input='+str(self.asset)]}]
        save_answers(other, self.answers)
        m = json.loads(self.manifest(self.package('named', recipe=other)).canonical_manifest_bytes)
        self.assertEqual(m['steps'][0]['command'][-1], '--input='+self.target)
