"""Non-REST selection and proof checks; no credentials, networking or allocation."""
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from colabcola import cli
from colabcola.lifecycle_io import LifecycleError, atomic
from colabcola.lifecycle_provider import GoogleProvider, provider_for_plan
from colabcola.packager import parser, package


class NonRestTests(unittest.TestCase):
    def test_legacy_and_historical_default_never_import_rest(self):
        real_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if 'lifecycle_rest' in name: raise AssertionError('beta REST must not be imported')
            return real_import(name,*args,**kwargs)
        for plan in ({}, {'provider_backend':'legacy','runtime_version':'latest','high_mem':True}):
            with mock.patch('builtins.__import__',side_effect=guarded), mock.patch('colabcola.lifecycle_provider.GoogleProvider') as provider:
                self.assertIs(provider_for_plan(plan),provider.return_value)
                self.assertEqual(provider.return_value.high_mem,plan.get('high_mem',False))

    def test_invalid_backend_or_legacy_date_fails_before_credentials(self):
        with mock.patch('colabcola.lifecycle_provider.GoogleProvider') as provider:
            for plan in ({'provider_backend':'typo'},{'provider_backend':'legacy','runtime_version':'2026.07'}):
                with self.assertRaises(LifecycleError):provider_for_plan(plan)
            provider.assert_not_called()

    def test_release_waits_for_exact_owned_runtime_absence(self):
        provider=GoogleProvider.__new__(GoogleProvider)
        provider.client=mock.Mock()
        provider.lookup=mock.Mock(return_value=object())
        with self.assertRaisesRegex(LifecycleError,'not yet confirmed'):provider.release('owned')
        provider.client.unassign.assert_called_once_with('owned')
        self.assertEqual(provider.lookup.call_args_list,[mock.call('owned')])
        provider.lookup=mock.Mock(return_value=None)
        provider.release('owned')

    def test_release_missing_listing_still_unassigns_exact_owned_endpoint(self):
        provider=GoogleProvider.__new__(GoogleProvider);provider.client=mock.Mock()
        provider.lookup=mock.Mock(return_value=None);provider.release('owned')
        provider.client.unassign.assert_called_once_with('owned')

    def test_public_packager_prepares_long_legacy_job_without_helper_or_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=root/'run.py';script.write_text('print("ordinary script")',encoding='utf-8')
            helper=root/'helper.json';atomic(helper,['must-not-run'])
            job=root/'probe.colajob'
            with mock.patch('colabcola.lifecycle.get_token',side_effect=AssertionError('no credentials')), mock.patch('colabcola.lifecycle_provider.provider_for_plan',side_effect=AssertionError('no provider')), contextlib.redirect_stdout(io.StringIO()):
                package(parser().parse_args(['-o',str(job),'--gpu','NONE','--max-runtimes','1','--max-seconds','4500','--timeout','4200','--drive-parent','placeholder','--token-command',str(helper),str(script)]))
                self.assertEqual(cli.main(['--prepare-only',str(job)]),0)
            plan=json.loads((root/'probe.run/plan.json').read_text(encoding='utf-8'))
            state=json.loads((root/'probe.run/state.json').read_text(encoding='utf-8'))
            self.assertEqual((plan['provider_backend'],plan['runtime_version']),('legacy','latest'))
            self.assertEqual(plan['authorization_policy'],'renewable-v1')
            self.assertEqual(plan['max_seconds'],4500);self.assertEqual(plan['max_attempts'],1)
            self.assertEqual(state,{'attempts':[],'events':[],'phase':'prepared'})


class RenewalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        path=Path(__file__).resolve().parents[1]/'examples/runtime_renewal/verify.py'
        spec=importlib.util.spec_from_file_location('verify_renewal',path)
        self.verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.verifier)
        self.plan={'gpu':'NONE','provider_backend':'legacy','runtime_version':'latest','max_attempts':1,'max_seconds':4500,'job_id':'id'}
        atomic(self.root/'plan.json',self.plan)
        atomic(self.root/'state.json',{'phase':'completed','attempts':[{'released':True}],'final':{'local_directory':str(self.root)}})
        self.summary={'requested_seconds':3900,'elapsed_seconds':3900.5,'python':'3.13.15'}
        with zipfile.ZipFile(self.root/'artifacts.zip','w') as z:z.writestr('outputs/probe/summary.json',json.dumps(self.summary))
        self.auth={'token_changes':1,'initial_expires_at':100,'latest_expires_at':4000,'verified_after_initial_expiry_at':225,'first_verified_rotated_publication_at':180}
        atomic(self.root/'authorization.json',self.auth)
        patcher=mock.patch.object(self.verifier,'verify_final',return_value={'terminal_state':'success'})
        patcher.start();self.addCleanup(patcher.stop)

    def test_legacy_acceptance_records_observed_python_without_dated_claim(self):
        report=self.verifier.check(self.root)[3]
        self.assertEqual(report['provider_backend'],'legacy');self.assertFalse(report['dated_image_verified'])
        self.assertEqual(report['summary']['python'],'3.13.15')

    def test_renewal_cannot_pass_with_same_token_or_early_publication(self):
        for changes in ({'token_changes':0},{'verified_after_initial_expiry_at':219}):
            atomic(self.root/'authorization.json',{**self.auth,**changes})
            with self.assertRaises(AssertionError):self.verifier.check(self.root)

    def test_rest_proof_retains_python312_requirement(self):
        atomic(self.root/'plan.json',{**self.plan,'provider_backend':'rest','runtime_version':'2026.07'})
        with self.assertRaises(AssertionError):self.verifier.check(self.root)


if __name__=='__main__':unittest.main()
