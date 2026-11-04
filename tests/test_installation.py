"""Offline install diagnostics must never construct an authenticated provider."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from colabcola import lifecycle
from colabcola.lifecycle_io import LifecycleError
from colabcola.lifecycle_provider import dependency_report, verify_provider


class InstallationTests(unittest.TestCase):
    def test_missing_provider_has_install_guidance(self):
        with mock.patch('colabcola.lifecycle_provider.importlib.resources.files', side_effect=ModuleNotFoundError):
            with self.assertRaisesRegex(LifecycleError, 'reinstall the colabcola wheel'):
                verify_provider()

    def test_different_provider_rejected_before_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'client.py').write_text('unrelated implementation')
            with mock.patch('colabcola.lifecycle_provider.importlib.resources.files', return_value=root):
                with self.assertRaisesRegex(LifecycleError, 'differs from inspected commit'):
                    verify_provider()

    def test_doctor_never_constructs_provider_or_checks_credentials(self):
        with mock.patch('colabcola.lifecycle_provider.dependency_report', return_value={'provider_source_verified':True}), \
             mock.patch('colabcola.lifecycle_provider.credentials', side_effect=AssertionError('credentials accessed')), \
             mock.patch('colabcola.lifecycle_provider.GoogleProvider', side_effect=AssertionError('provider constructed')):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(lifecycle.main(['doctor']),0)
            result=json.loads(output.getvalue())
            self.assertFalse(result['credentials_checked'])
            self.assertFalse(result['network_accessed'])
            self.assertFalse(result['live_provider_verified'])

    def test_installed_provider_identity(self):
        import importlib.util
        with mock.patch('colabcola.lifecycle_provider.credentials', side_effect=AssertionError('credentials accessed')):
            self.assertTrue(dependency_report()['provider_source_verified'])
