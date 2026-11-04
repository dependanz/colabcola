"""Named CLI selection and private-profile packaging, using synthetic local files."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from colabcola import cli, packager, workflow
from colabcola.errors import SpecError
from colabcola.profiles import config_home, named_home


class ProfileCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / 'home'
        self.home.mkdir()
        self.project = self.root / 'project'
        self.project.mkdir()
        self.script = self.project / 'work.py'
        self.script.write_text('print("fixture")\n', encoding='utf-8')
        home = mock.patch('pathlib.Path.home', return_value=self.home)
        home.start()
        self.addCleanup(home.stop)
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('COLABCOLA_CONFIG_HOME', None)
        os.environ.pop('COLABCOLA_PROFILE', None)

    def call(self, arguments, main=cli.main):
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return main(arguments)

    def test_global_and_submission_option_select_profile_only_for_execution(self):
        old = str(self.root / 'explicit-old-home')
        os.environ['COLABCOLA_CONFIG_HOME'] = old
        os.environ['COLABCOLA_PROFILE'] = 'environment'
        for argv in (
            ['--profile', 'work', 'job.colajob', '--profile', 'workload'],
            ['--transport', 'embedded', '--profile=work', 'job.colajob', '--profile', 'workload'],
            ['--profile=work', 'compile', 'job.colajob', '--profile', 'workload'],
        ):
            with self.subTest(argv=argv):
                seen = []
                def execute(command):
                    seen.append((config_home(), command))
                    return 0
                with mock.patch.object(cli, '_execute', side_effect=execute):
                    self.assertEqual(self.call(argv), 0, self.stderr.getvalue())
                self.assertEqual(seen[0][0], named_home('work'))
                self.assertEqual(seen[0][1].profile, 'work')
                self.assertEqual(seen[0][1].job_args, ('--profile', 'workload'))
                self.assertEqual(os.environ['COLABCOLA_CONFIG_HOME'], old)
                self.assertEqual(os.environ['COLABCOLA_PROFILE'], 'environment')
        self.assertFalse((self.home / '.colabcola').exists())

    def test_job_argument_profile_without_selector_is_not_consumed(self):
        command = cli._parse_command(['job.colajob', '--profile', '../literal-workload-value'])
        self.assertIsNone(command.profile)
        self.assertEqual(command.job_args, ('--profile', '../literal-workload-value'))

    def test_package_prefix_and_direct_selection_keep_workload_arguments(self):
        for index, (entry, prefix) in enumerate((
            (cli.main, ['--profile', 'work', 'package']),
            (cli.main, ['package', '--profile=work']),
            (packager.main, ['--profile', 'work']),
        )):
            with self.subTest(prefix=prefix):
                output = self.root / f'package-{index}.colajob'
                with mock.patch.object(packager, 'package', wraps=packager.package) as operation:
                    self.assertEqual(self.call(prefix + ['--output', str(output), '--', 'python',
                        str(self.script), '--profile', '../literal-workload-value'], entry), 0, self.stderr.getvalue())
                self.assertEqual(operation.call_args.args[0].profile, 'work')
                commands = json.loads((output.with_suffix('.files') / '_colabcola/commands.json').read_text())
                self.assertEqual(commands['commands'], [['work.py', '--profile', '../literal-workload-value']])
                self.assertEqual(json.loads(output.with_suffix('.run.json').read_text())['profile'], 'default')
                self.assertNotIn('COLABCOLA_CONFIG_HOME', os.environ)
        self.assertFalse((self.home / '.colabcola').exists())

    def test_run_selector_is_scoped_for_global_subcommand_and_direct_calls(self):
        for entry, prefix in (
            (cli.main, ['--profile', 'work', 'run']),
            (cli.main, ['run', '--profile', 'work']),
            (workflow.main, ['--profile', 'work']),
        ):
            with self.subTest(prefix=prefix):
                seen = []
                def execute(args):
                    seen.append((config_home(), args.profile))
                    return 0
                with mock.patch.object(workflow, '_run', side_effect=execute):
                    self.assertEqual(self.call(prefix + ['--colaml', 'recipe.colaml', '--run-dir', 'run'], entry), 0)
                self.assertEqual(seen, [(named_home('work'), 'work')])
                self.assertNotIn('COLABCOLA_CONFIG_HOME', os.environ)

    def test_prefix_dispatches_login_and_scopes_legacy_lifecycle(self):
        with mock.patch('colabcola.onboarding.main', return_value=0) as login:
            self.assertEqual(self.call(['--profile', 'work', 'login', '--status']), 0)
        login.assert_called_once_with(['--profile', 'work', '--status'])
        seen = []
        def legacy(arguments):
            seen.append((arguments, config_home()))
            return 0
        with mock.patch('colabcola.lifecycle.main', side_effect=legacy):
            self.assertEqual(self.call(['--profile', 'work', 'lifecycle', 'status']), 0)
        self.assertEqual(seen, [(['status'], named_home('work'))])
        self.assertNotIn('COLABCOLA_CONFIG_HOME', os.environ)

    def test_repeated_and_conflicting_selectors_fail_before_any_action(self):
        commands = (
            ['--profile', 'work', '--profile', 'work', 'job.colajob'],
            ['--profile', 'work', 'compile', '--profile', 'other', 'job.colajob'],
            ['--profile', 'work', 'package', '--profile', 'work', '--output', 'unused.colajob', str(self.script)],
            ['package', '--profile', 'work', '--profile', 'other', '--output', 'unused.colajob', str(self.script)],
            ['--profile', 'work', 'run', '--profile', 'other', '--colaml', 'recipe.colaml', '--run-dir', 'unused'],
        )
        for argv in commands:
            with self.subTest(argv=argv), mock.patch.object(cli, '_execute') as submit, \
                 mock.patch.object(packager, 'package') as package, mock.patch.object(workflow, '_run') as run:
                with self.assertRaises(SystemExit) as failure:
                    self.call(argv)
                self.assertEqual(failure.exception.code, 2)
                self.assertIn('--profile may be specified only once', self.stderr.getvalue())
                submit.assert_not_called(); package.assert_not_called(); run.assert_not_called()
        self.assertFalse((self.home / '.colabcola').exists())

    def test_invalid_selection_does_not_start_execution_or_create_workspace(self):
        commands = (
            ['--profile', '../bad', 'job.colajob'],
            ['package', '--profile', '../bad', '--output', str(self.root / 'unused.colajob'), str(self.script)],
            ['run', '--profile', '../bad', '--colaml', 'recipe.colaml', '--run-dir', str(self.root / 'unused')],
        )
        for argv in commands:
            with self.subTest(argv=argv), mock.patch.object(cli, '_execute') as submit, \
                 mock.patch.object(packager, 'package') as package, mock.patch.object(workflow, '_run') as run:
                self.assertEqual(self.call(argv), 2)
                self.assertIn('Invalid profile name', self.stderr.getvalue())
                submit.assert_not_called(); package.assert_not_called(); run.assert_not_called()
        self.assertFalse((self.root / 'unused').exists())
        self.assertFalse((self.home / '.colabcola').exists())

    def test_selection_is_restored_after_execution_error(self):
        os.environ['COLABCOLA_CONFIG_HOME'] = str(self.root / 'old')
        with mock.patch.object(cli, '_execute', side_effect=RuntimeError('fixture failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                self.call(['--profile', 'work', 'job.colajob'])
        self.assertEqual(os.environ['COLABCOLA_CONFIG_HOME'], str(self.root / 'old'))

    def test_help_documents_profile_and_internal_token_rejects_global_selector(self):
        for argv in (['--help'], ['package', '--help'], ['run', '--help']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as result:
                self.call(argv)
            self.assertEqual(result.exception.code, 0)
            self.assertIn('--profile NAME', self.stdout.getvalue())
        with mock.patch('colabcola.onboarding.token_main') as token, self.assertRaises(SystemExit) as result:
            self.call(['--profile', 'work', '_drive-token', '--config-home', 'elsewhere'])
        self.assertEqual(result.exception.code, 2)
        token.assert_not_called()

    def private_files(self):
        paths = [
            self.home / '.colabcola/configs/work/ordinary.txt',
            self.home / '.colabcola/configs/inactive/ordinary.txt',
            self.home / '.colabcola/other-private/ordinary.txt',
            self.home / '.config/colabcola/ordinary.txt',
            self.home / 'custom-account/ordinary.txt',
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('SYNTHETIC-PRIVATE-CREDENTIAL', encoding='utf-8')
        os.environ['COLABCOLA_CONFIG_HOME'] = str(self.home / 'custom-account')
        return paths

    def test_home_tree_package_excludes_every_named_legacy_and_active_config(self):
        self.private_files()
        source = self.home / 'work.py'
        source.write_text('pass\n', encoding='utf-8')
        output = self.root / 'home.colajob'
        self.assertEqual(self.call(['package', '--project', str(self.home), '--output', str(output),
            '--', 'python', 'work.py']), 0, self.stderr.getvalue())
        job = json.loads(output.read_text())
        self.assertEqual({item['target'] for item in job['payload']},
            {'work.py', '_colabcola/launch.py', '_colabcola/commands.json'})
        for path in output.with_suffix('.files').rglob('*'):
            if path.is_file():
                self.assertNotIn(b'SYNTHETIC-PRIVATE-CREDENTIAL', path.read_bytes())

    def test_named_selection_keeps_shadowed_custom_config_out_of_home_package(self):
        self.private_files()
        source = self.home / 'work.py'
        source.write_text('pass\n', encoding='utf-8')
        output = self.root / 'named-home.colajob'
        self.assertEqual(self.call(['--profile', 'work', 'package', '--project', str(self.home),
            '--output', str(output), '--', 'python', 'work.py']), 0, self.stderr.getvalue())
        job = json.loads(output.read_text())
        self.assertEqual({item['target'] for item in job['payload']},
            {'work.py', '_colabcola/launch.py', '_colabcola/commands.json'})
        self.assertEqual(os.environ['COLABCOLA_CONFIG_HOME'], str(self.home / 'custom-account'))

    def test_private_configuration_inputs_are_rejected_before_reading(self):
        paths = self.private_files()
        for option in ('--requirements', '--remote-inputs', '--commands', '--answers'):
            with self.subTest(option=option):
                source = paths[-1].with_name('private-input.colaml' if option == '--answers' else 'private-input.json')
                source.write_text('SYNTHETIC-PRIVATE-CREDENTIAL', encoding='utf-8')
                args = ['--profile', 'work', 'package', '--output', str(self.root / 'rejected.colajob'), option, str(source)]
                if option != '--commands':
                    args.append(str(self.script))
                self.assertEqual(self.call(args), 2)
                self.assertIn('account configuration cannot be packaged', self.stderr.getvalue())
                self.assertNotIn('SYNTHETIC-PRIVATE-CREDENTIAL', self.stderr.getvalue() + self.stdout.getvalue())
                self.assertFalse((self.root / 'rejected.colajob').exists())

    def test_explicit_private_data_files_and_directories_are_rejected(self):
        paths = self.private_files()
        for index, source in enumerate([*paths, *(path.parent for path in paths)]):
            with self.subTest(source=source):
                output = self.root / f'rejected-{index}.colajob'
                self.assertEqual(self.call(['package', '--output', str(output), '--data', str(source) + '=data',
                    str(self.script)]), 2)
                self.assertIn('account configuration cannot be packaged', self.stderr.getvalue())
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix('.files').exists())

    def test_explicit_host_token_helper_remains_outside_portable_payload(self):
        self.private_files()
        helper = self.home / 'custom-account/drive-helper.json'
        argv = ['python', '-m', 'colabcola', '_drive-token', '--config-home', str(helper.parent)]
        helper.write_text(json.dumps(argv), encoding='utf-8')
        output = self.root / 'legacy-helper.colajob'
        self.assertEqual(self.call(['package', '--output', str(output), '--drive-parent', 'fixture',
            '--token-command', str(helper), str(self.script)]), 0, self.stderr.getvalue())
        job = json.loads(output.read_text())
        self.assertNotIn('_colabcola/drive-helper.json', {item['target'] for item in job['payload']})
        copied = output.with_suffix('.files') / '_colabcola/drive-helper.json'
        self.assertEqual(json.loads(copied.read_text()), argv)
        for item in job['payload']:
            self.assertNotIn(b'SYNTHETIC-PRIVATE-CREDENTIAL', (output.parent / item['source']).read_bytes())

    def test_private_script_and_wheel_paths_cannot_bypass_data_guard(self):
        paths = self.private_files()
        script = paths[1].with_name('entry.py')
        script.write_text('pass\n', encoding='utf-8')
        wheel = paths[1].with_name('private.whl')
        wheel.write_bytes(b'SYNTHETIC-PRIVATE-CREDENTIAL')
        for options in ([str(script)], ['--wheel', str(wheel), str(self.script)]):
            with self.subTest(options=options):
                args = packager.parser().parse_args(['--output', str(self.root / 'rejected.colajob'), *options])
                with self.assertRaisesRegex(SpecError, 'account configuration cannot be packaged'):
                    packager.package(args)


if __name__ == '__main__':
    unittest.main()
