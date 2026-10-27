# Phase 3, step 4: full-LJ transport measurement

This run measures the production Drive path with the complete pinned LJ Speech corpus. It uses a
CPU-only probe and does **not** train HiFi-GAN. Do not run the similarly named
`hifigan-ljspeech-full-transport-2500.colajob` during this measurement: that job starts a 2,500-step
full-corpus training segment.

## Bound local inputs

| Item | Verified value |
|---|---:|
| JobSpec | `jobs/hifigan-ljspeech-full-transport-probe.colajob` |
| Staged bundle | `dist/hifigan-ljspeech-full-transport-probe.colabcola` |
| Job ID | `4f4992d525300d61e67cdaeafec39f4297f7451c1a76a37c1a3e7a99167c51f9` |
| Staged bytes | `3,213,481,440` |
| Staged SHA-256 | `d7d06f7fa6e9f57220b52ccb4cfc5c64d72c83208614e0e5718a4d66427ba2b3` |
| Archive bytes | `3,211,120,106` |
| Archive SHA-256 | `d1ae192d8e715c5e01bbf2b1064add7674aa1910dfe378b73279888ea38413eb` |
| File records | `13,116` |
| Raw input bytes | `3,802,808,156` |

The bundle is above the 1,024 MiB warning threshold. The primary Drive transport remains allowed.
Do not use or reveal the embedded-base64 transport for this job: it would contain approximately
4,281,493,476 base64 bytes and can make the Colab frontend unusable.

## Operator procedure

1. Open PowerShell in `C:\dev\colabcola` and activate the project virtual environment if it is not
   already active.

   ```powershell
   Set-Location C:\dev\colabcola
   .\.venv\Scripts\Activate.ps1
   ```

2. Submit the probe once. This recompiles and verifies the deterministic bundle, uploads it to the
   authorized Drive account, and writes a bound controller and lifecycle receipt. Preserve the one
   JSON line printed at the end.

   ```powershell
   colabcola --output .\dist\hifigan-ljspeech-full-transport-probe.colabcola .\jobs\hifigan-ljspeech-full-transport-probe.colajob
   ```

   The JSON must report the bound job ID and SHA-256 above, `reused:false`, and these timing fields:
   `compile_seconds`, `drive_staging_seconds`, and `controller_seconds`. The first
   `drive_staging_seconds` value is the measured full upload/staging time.

3. Run the identical submission command a second time and preserve its JSON. It must bind the same
   Drive file ID and report `reused:true`. Its `drive_staging_seconds` value measures the exact-object
   lookup/reuse path without another 3.21 GB upload.

4. Open the generated
   `dist\hifigan-ljspeech-full-transport-probe-controller.ipynb` in Google Colab. Select runtime
   `2026.07` with **CPU** hardware, grant this notebook access only to
   `COLABCOLA_GOOGLE_DRIVE_OAUTH`, and choose **Run all**. Complete Google's device authorization
   when the acquisition cell asks for it.

5. Confirm that the execution cell ends with `transport_probe_complete`. The final output also
   prints one line beginning `colabcola artifacts evidence:`. Copy that entire JSON value; it is the
   independent expected byte count and SHA-256 for the download.

6. Download
   `/content/colabcola/results/4f4992d525300d61e67cdaeafec39f4297f7451c1a76a37c1a3e7a99167c51f9/artifacts.zip`
   to `C:\Users\Danze\.colabcola\full-transport-probe-artifacts.zip`. Time the download from the
   click until the local file finishes writing; record the elapsed seconds.

7. Read the result locally without extracting it. Its `transport` object contains exact
   `authorization_and_metadata_seconds`, `download_seconds`, `staged_reconstruction_seconds`,
   `acquisition_seconds`, and `extraction_seconds` measurements from the Colab runtime.

   ```powershell
   python -c "import json,zipfile; p=r'C:\Users\Danze\.colabcola\full-transport-probe-artifacts.zip'; z=zipfile.ZipFile(p); r=json.loads(z.read('result.json')); print(json.dumps(r,indent=2,sort_keys=True))"
   ```

8. Replace the two placeholders below with the `size` and `sha256` copied from the Colab artifact
   evidence line. This verifies the local download and advances the receipt to `results_verified`.

   ```powershell
   $receipt='C:\dev\colabcola\dist\hifigan-ljspeech-full-transport-probe-drive-receipt.json'
   $artifact='C:\Users\Danze\.colabcola\full-transport-probe-artifacts.zip'
   $expectedSize=REPLACE_WITH_COLAB_SIZE
   $expectedSha='REPLACE_WITH_COLAB_SHA256'
   python -c "from pathlib import Path; from colabcola.drive import load_drive_receipt,record_verified_result,write_drive_receipt; p=Path(r'$receipt'); r=record_verified_result(load_drive_receipt(p),Path(r'$artifact'),expected_size=$expectedSize,expected_sha256='$expectedSha'); write_drive_receipt(r,p); print(r.state)"
   ```

9. Only after step 8 prints `results_verified`, clean the receipt-scoped staged payload and time the
   operation. This moves exactly the recorded payload ID to Drive trash; it does not touch any
   previous failed job.

   ```powershell
   $timer=[Diagnostics.Stopwatch]::StartNew()
   python -c "from pathlib import Path; from colabcola.drive import cleanup_drive_staging,load_drive_receipt,write_drive_receipt; from colabcola.google_drive import create_google_drive_backend,load_google_drive_settings; p=Path(r'$receipt'); r=load_drive_receipt(p); c=cleanup_drive_staging(r,create_google_drive_backend(load_google_drive_settings())); write_drive_receipt(c,p); print(c.state)"
   $timer.Stop()
   "cleanup_seconds=$($timer.Elapsed.TotalSeconds)"
   ```

10. Send the two submission JSON lines, the downloaded `result.json` display, the artifact evidence
    line, `download_seconds`, `cleanup_seconds`, and the final receipt display back for reconciliation.

## Acceptance checks

1. The first submission reports `reused:false`; the second reports `reused:true` for one unchanged
   Drive file ID.
2. Colab reports 12,950 training clips, 150 validation clips, 13,100 WAVs, 3,798,339,464 WAV bytes,
   and 3,798,483,564 dataset payload bytes.
3. The result has terminal state `success`, and its bundle/job identities equal the bound values.
4. The downloaded artifact matches Colab's independently printed size and SHA-256.
5. The final receipt has state `cleaned`, with its one cleanup file ID also present in
   `cleaned_file_ids`.

## Bounded generated-audio diagnostic preparation

This follow-up uses only the verified step-5,000 generator and three pinned validation WAVs. It
returns each exact reference WAV, one generated WAV per reference, and a canonical JSON report.
The report is operational evidence that the checkpoint loads and emits finite audio; it is not a
convergence, perceptual-quality, MOS, or paper-reproduction claim.

1. Extract only `outputs/cp_hifigan/g_00005000` from the already verified result archive. The
   helper refuses the wrong member, size, SHA-256, symlinks, changed existing output, or duplicate
   archive member. It is idempotent and does not extract the approximately 961 MB state checkpoint.

   ```powershell
   Set-Location C:\dev\colabcola
   .\.venv\Scripts\Activate.ps1
   python -m colabcola._hifigan.extract_audio_diagnostic_input `
     'C:\Users\Danze\.colabcola\artifacts (5).zip' `
     'C:\dev\colabcola\dist\hifigan-ljspeech-audio-diagnostic-input\g_00005000'
   ```

   This should take seconds and prints one JSON line only when complete. It must report 55,825,339
   bytes, SHA-256 `836627cc247ec688589593d2324aae2cfadc8958eb17304c9426c44de56b46e3`,
   and `reused:false` on the first run (`reused:true` is valid if the exact output already exists).

2. Validate the now-complete 12-entry JobSpec. This should finish in well under a second.

   ```powershell
   colabcola validate .\jobs\hifigan-ljspeech-audio-diagnostic.colajob
   ```

3. Build and independently verify the Drive bundle locally without uploading it. The command is
   currently quiet until its final JSON line. It handles approximately 57 MB of inputs and should
   take well under a minute on the measured workstation, not the several minutes required by the
   full corpus.

   ```powershell
   python -c "import json,time; from colabcola.compiler import prepare_drive_bundle; from colabcola.spec import load_job_spec; t=time.perf_counter(); r=prepare_drive_bundle(load_job_spec(r'.\jobs\hifigan-ljspeech-audio-diagnostic.colajob'),r'.\dist\hifigan-ljspeech-audio-diagnostic.colabcola'); print(json.dumps({'archive_bytes':len(r.artifact.bundle.archive_bytes),'bundle_bytes':r.staged_bytes,'bundle_sha256':r.staged_sha256,'compile_seconds':time.perf_counter()-t,'file_count':len(r.artifact.bundle.files),'job_id':r.job_id,'warnings':list(r.warnings)},sort_keys=True,separators=(',',':')))"
   ```

4. Stop after the local JSON line. Record its exact job ID, byte count, SHA-256, duration, and
   warnings before separately authorizing or performing Drive upload, controller launch, result
   recovery, or cleanup.
