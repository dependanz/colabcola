"""Recipe-to-run command: preserve inputs, freeze one job, reuse one journal."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from ._version import __version__
from .errors import BundleError, SpecError
from .lifecycle_io import LifecycleError, atomic, digest, encoded, read_regular
from .profiles import add_profile_argument, selected


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redact(text):
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", text)
    return re.sub(r"(?i)((?:access_token|refresh_token|client_secret|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+",r"\1[redacted]",text)


class _Tee:
    """Keep console streams usable; timestamp/redact automatic log lines."""
    def __init__(self, stream, log, channel, lock):
        self.stream, self.log, self.channel, self.lock = stream, log, channel, lock
        self.pending = ""
    @property
    def encoding(self):
        return getattr(self.stream,"encoding","utf-8")
    def isatty(self):
        return self.stream.isatty()
    def write(self, text):
        with self.lock:
            self.stream.write(text)
            self.pending += text
            while "\n" in self.pending:
                line,self.pending = self.pending.split("\n",1)
                self.log.write(f"[{_utc()} {self.channel}] " + _redact(line.rstrip("\r")) + "\n")
            if len(self.pending) > 65536:
                self.log.write(f"[{_utc()} {self.channel}] " + _redact(self.pending) + "\n")
                self.pending = ""
            self.log.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def finish(self):
        with self.lock:
            if self.pending:
                self.log.write(f"[{_utc()} {self.channel}] " + _redact(self.pending) + "\n")
                self.pending = ""
            self.flush()


def parser():
    p = argparse.ArgumentParser(prog="colabcola run", description="Package a .colaml recipe, prepare, preflight and run it in one command. Repeat the same workspace to reconnect without repackaging.")
    p.add_argument("--colaml", required=True, type=Path, help="recipe; read only, with paths relative to its original location")
    p.add_argument("--run-dir", required=True, type=Path, help="new workflow folder, or this command's existing folder to reconnect; never overwrites another job")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--prepare-only",action="store_true",help="package and prepare offline; do not check credentials or allocate")
    group.add_argument("--preflight-only",action="store_true",help="package/prepare and run online read-only admission without uploading or allocating")
    p.add_argument("--non-interactive",action="store_true",help="require the .colaml to answer all packaging questions")
    p.add_argument("--approve",help="optional exact approval for this workspace's prepared plan")
    p.add_argument("--expect-job-id",help="optional portable job SHA256 guard, retained on reconnect")
    p.add_argument("--allow-active-assignment",action="append",default=[],metavar="ENDPOINT")
    add_profile_argument(p)
    return p


def _destination(path):
    value = Path(os.path.abspath(path))
    for parent in (value,*value.parents):
        if parent.is_symlink() or (hasattr(parent,"is_junction") and parent.is_junction()):
            raise LifecycleError("workflow links/junctions are not supported")
    return value


def _identity(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def _open_workspace(args):
    workspace = _destination(args.run_dir)
    recipe = Path(os.path.abspath(args.colaml))
    if recipe.suffix != ".colaml":
        raise SpecError("--colaml must name a .colaml recipe")
    if args.expect_job_id is not None and not _identity(args.expect_job_id):
        raise SpecError("--expect-job-id must be a SHA256")
    if workspace.exists():
        # Reject unrelated directories BEFORE creating locks, logs or exit files.
        record = json.loads(read_regular(workspace,"workflow.json",65536))
        if (record.get("schema") != 1 or record.get("kind") != "colabcola-recipe-run"
                or record.get("directory") != str(workspace) or record.get("recipe") != str(recipe)
                or record.get("phase") != "ready" or not _identity(record.get("job_id"))):
            raise LifecycleError("not a completed recipe preparation for this command; use a NEW --run-dir")
        name = record.get("job_file")
        if not isinstance(name,str) or Path(name).name != name or not re.fullmatch(r"[A-Za-z0-9_-]+\.colajob",name):
            raise LifecycleError("invalid workflow job path")
        expected = record.get("expected_job_id")
        if expected and args.expect_job_id and expected != args.expect_job_id:
            raise LifecycleError("expected job identity changed; use a new workspace")
        return workspace,record,False
    from .packager import _safe_input
    _safe_input(recipe)
    # A sibling is safe; creating inside a previous job folder is not preservation.
    for parent in workspace.parents:
        if parent.is_dir() and any(parent.glob("*.run/plan.json")):
            raise LifecycleError("fresh workflow must be outside existing managed job folders")
    workspace.mkdir(parents=True,exist_ok=False)
    stem = re.sub(r"[^A-Za-z0-9_-]+","-",recipe.stem).strip("-") or "job"
    record = {"schema":1,"kind":"colabcola-recipe-run","phase":"packaging",
              "directory":str(workspace),"recipe":str(recipe),"recipe_sha256":digest(recipe.read_bytes()),
              "job_file":stem+".colajob","expected_job_id":args.expect_job_id,
              "created_at":_utc(),"controller_version":__version__}
    atomic(workspace/"workflow.json",record)
    return workspace,record,True


def _package(recipe, destination, non_interactive):
    from .packager import parser as package_parser, package
    from .package_guide import guide
    arguments = ["--colaml",str(recipe),"--output",str(destination)]
    if non_interactive:
        arguments.append("--non-interactive")
    cli = package_parser()
    args = cli.parse_args(arguments)
    args,answers,save_path,questions = guide(args,arguments,cli)
    result = package(args)
    if not result["execution_config"]:
        raise SpecError("packager did not produce execution settings")
    result["questions_asked"] = questions
    return result


def _execute(args, workspace, record, fresh):
    from .bundle import build_bundle
    from .invocation import load_job_invocation
    from .cli import SubmissionCommand
    from .managed import run_managed
    job = workspace / record["job_file"]
    if fresh:
        print("colabcola: Packaging recipe into " + str(workspace),file=sys.stderr,flush=True)
        result = _package(Path(record["recipe"]),job,args.non_interactive)
        atomic(workspace/"packaging.json",result)
        record.update(phase="ready",job_id=result["job_id"])
        atomic(workspace/"workflow.json",record)
    else:
        print("colabcola: Reusing frozen job and journal in " + str(workspace),file=sys.stderr,flush=True)
        recipe = Path(record["recipe"])
        if not recipe.is_file() or digest(recipe.read_bytes()) != record["recipe_sha256"]:
            print("colabcola: Recipe changed or is absent; reconnecting with this workspace's frozen inputs",file=sys.stderr,flush=True)
    invocation = load_job_invocation(job)
    actual = build_bundle(invocation).job_id
    expected = record.get("expected_job_id") or args.expect_job_id
    if actual != record["job_id"] or (expected and actual != expected):
        raise LifecycleError("packaged job differs from the expected identity; inspect this workspace before running")
    if expected and not record.get("expected_job_id"):
        record["expected_job_id"] = expected
        atomic(workspace/"workflow.json",record)
    print("colabcola: Job " + actual,file=sys.stderr,flush=True)
    print("colabcola: Automatic command log: " + str(workspace/"launch.log"),file=sys.stderr,flush=True)
    command = SubmissionCommand(job_file=job,job_args=(),output=None,drive_config=None,
        prepare_only=args.prepare_only,preflight_only=args.preflight_only,approve=args.approve,
        allow_active_assignments=tuple(args.allow_active_assignment),
        profile=getattr(args, "profile", None))
    return run_managed(command,invocation)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        with selected(args.profile):
            return _run(args)
    except LifecycleError as exc:
        print("colabcola run: " + _redact(str(exc)),file=sys.stderr)
        return 2


def _run(args):
    from .lifecycle import run_lock
    try:
        workspace,record,fresh = _open_workspace(args)
        # Lock the whole workflow, including packaging. The managed journal also
        # retains its own lock for direct .colajob reconnection/cleanup.
        with run_lock(workspace):
            code,started = 2,_utc()
            with (workspace/"launch.log").open("a",encoding="utf-8",newline="\n") as log:
                lock = threading.RLock()
                out,err = _Tee(sys.stdout,log,"stdout",lock),_Tee(sys.stderr,log,"stderr",lock)
                try:
                    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                        try:
                            code = _execute(args,workspace,record,fresh)
                        except KeyboardInterrupt:
                            print("colabcola run: interrupted; repeat the same workspace to inspect/reconnect",file=sys.stderr)
                            code = 130
                        except (OSError,ValueError,SpecError,BundleError,LifecycleError) as exc:
                            print("colabcola run: " + _redact(str(exc)),file=sys.stderr)
                        except Exception as exc:
                            print("colabcola run: " + type(exc).__name__ + "; inspect this workspace before reconnecting",file=sys.stderr)
                finally:
                    out.finish();err.finish()
                    journal = workspace/Path(record["job_file"]).with_suffix(".run")
                    report = {"exit_code":code,"started_at_utc":started,"observed_finished_utc":_utc(),
                              "controller_version":__version__,"job_id":record.get("job_id"),
                              "journal":str(journal),"prepare_only":args.prepare_only,"preflight_only":args.preflight_only}
                    if (journal/"state.json").is_file():
                        try:
                            state=json.loads(read_regular(journal,"state.json"))
                            report["phase"]=state.get("phase")
                            report["attempts"]=len(state.get("attempts",[]))
                            plan=json.loads(read_regular(journal,"plan.json"))
                            report["approval_sha256"]=digest(encoded(plan))
                        except (OSError,ValueError,LifecycleError):
                            report["journal_attention"]="unreadable"
                    atomic(workspace/"cli-exit.json",report)
            return code
    except (OSError,ValueError,SpecError,BundleError,LifecycleError) as exc:
        print("colabcola run: " + _redact(str(exc)),file=sys.stderr)
        return 2
