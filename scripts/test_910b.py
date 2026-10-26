#!/usr/bin/env python3
"""One-command Linux acceptance suite for 910B/CANN 8.5.0; stdlib bootstrap."""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paretopilot.config import load_config
from paretopilot.official import enabled as official_enabled, batch_enabled, default_params, COMMIT
from paretopilot.search import confirmation_roles
from paretopilot.workload import baseline_only, shapes
from paretopilot.routing import same_operator
from paretopilot.util import TuningError, read_json, write_json


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive(value):
    value = nonnegative(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cann-path", required=True, type=Path, help="CANN root, e.g. /usr/local/Ascend/cann or ascend-toolkit/latest")
    p.add_argument("--cann-env", type=Path, help="Explicit set_env.sh when it is outside the CANN root")
    p.add_argument("--soc", required=True, help="Exact target SoC, e.g. Ascend910B1; check npu-smi")
    p.add_argument("--device", type=nonnegative, default=0)
    p.add_argument("--operator", choices=("all", "matmul", "add"), default="all", help="Select hardware acceptance stages")
    p.add_argument("--matmul-implementation", choices=("batch-v3", "legacy"), default="batch-v3", help="Default: BatchMatMulV3 source baseline and Host tiling; legacy retains earlier MatMulV3/baseline-only cases")
    p.add_argument("--matmul-case", choices=("2d", "shared_rhs", "batched_rhs", "3d", "all"), default="2d", help="Choose 2D/3D workloads; batch-v3 tunes all cases, legacy 3D cases are baseline-only")
    p.add_argument("--scope", choices=("smoke", "full"), default="full", help="Both run preflight, unit tests, simulation, baselines and smoke; full also searches both operators")
    p.add_argument("--deps", choices=("install", "offline", "check"), default="install", help="install/offline use an isolated venv; check uses this interpreter without installing dependencies")
    p.add_argument("--wheelhouse", type=Path, help="Local wheels for --deps offline")
    p.add_argument("--venv", type=Path, default=ROOT/".venv-910b", help="Reusable venv for install/offline; must match this Python version")
    p.add_argument("--output", type=Path, help="New or empty directory; default creates runs/910b-<unique-id>")
    p.add_argument("--config-dir", type=Path, default=ROOT/"examples", help="Directory with ascend910b_add.json and ascend910b_matmul.json")
    p.add_argument("--timeout", type=positive, default=1800, help="Per compiler/ATC/runtime subprocess and doctor dependency-import timeout in seconds")
    return p


def resolve_cann(path, env_script=None):
    supplied = path.expanduser().resolve()
    if not supplied.is_dir():
        raise TuningError(f"CANN directory not found: {supplied}")
    root = (supplied/"latest").resolve() if (supplied/"latest").is_dir() else supplied
    candidates = [supplied/"set_env.sh", root/"set_env.sh"]
    if supplied.name == "latest" or root.parent.name == "ascend-toolkit":
        candidates.append(root.parent/"set_env.sh")
    script = env_script.expanduser().resolve() if env_script else next((p for p in candidates if p.is_file()), None)
    if script is None or not script.is_file():
        raise TuningError("Cannot find set_env.sh; pass --cann-env /actual/path/set_env.sh")
    files = [root/"version.info", *root.glob("*/ascend_toolkit_install.info")]
    versions = {str(p): p.read_text(errors="replace") for p in files if p.is_file()}
    if not any(re.search(r"(?im)^\s*version\s*[:=]\s*8\.5\.0\s*$", text) for text in versions.values()):
        raise TuningError(f"CANN root must contain an installed 8.5.0 version record: {root}")
    return root, script, versions


def source_environment(root, script, log):
    # The path is a positional argument, never interpolated into shell code.
    with log.open("wb") as stream:
        result = subprocess.run(["bash", "-c", 'source "$1" >&2 || exit $?; env -0', "paretopilot-env", str(script)],
                                stdout=subprocess.PIPE, stderr=stream, timeout=60, check=True)
    env = {}
    for item in result.stdout.split(b"\0"):
        if item:
            key, value = item.split(b"=", 1)
            env[os.fsdecode(key)] = os.fsdecode(value)
    env["ASCEND_TOOLKIT_HOME"] = str(root)
    env["ASCEND_HOME_PATH"] = str(root)
    env["PATH"] = str(root/"python/site-packages/bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"], env["PYTHONIOENCODING"] = "1", "utf-8"
    # No full environment dump: credentials in the caller's environment stay private.
    return env


def create_output(path=None):
    if path is None:
        (ROOT/"runs").mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="910b-", dir=ROOT/"runs"))
    root = path.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise TuningError("--output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


class Session:
    def __init__(self, root, args):
        self.root, self.env = root, os.environ.copy()
        self.data = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                     "scope": args.scope, "matmul_case": args.matmul_case, "matmul_implementation": args.matmul_implementation, "soc": args.soc, "device": args.device, "stages": [], "results": {}}
        self.save()

    def save(self):
        write_json(self.root/"test-summary.json", self.data)

    def run(self, name, argv):
        log = self.root/(name+".log")
        stage = {"name": name, "argv": list(map(str, argv)), "log": str(log), "status": "running"}
        self.data["stages"].append(stage)
        self.save()
        print(f"\n[{name}] Log: {log}", flush=True)
        started = time.monotonic()
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(stage["argv"], cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
                try:
                    tail = deque(maxlen=40)
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        decoded = line.decode("utf-8", errors="replace")
                        tail.append(decoded)
                        print(decoded, end="", flush=True)
                    code = process.wait()
                except KeyboardInterrupt:
                    # Let the controller interrupt its own compiler/runtime process groups.
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait()
                    raise
                finally:
                    process.stdout.close()
            stage["exit_code"] = code
            if code != 0:
                raise TuningError(f"Stage {name} exited {code}; see {log}")
            if name == "unit-tests" and any(re.search(r"OK \(skipped=[1-9]", line) for line in tail):
                raise TuningError(f"Unit tests skipped required coverage; see {log}")
            stage["status"] = "passed"
        except BaseException as exc:
            stage.update(status="failed", error=str(exc) or type(exc).__name__)
            raise
        finally:
            stage["seconds"] = time.monotonic()-started
            self.save()


def prepare_python(session, args):
    if args.deps == "check":
        return sys.executable
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise TuningError("Pinned installation requires Python 3.10-3.12; use a matching interpreter")
    venv = args.venv.expanduser().resolve()
    python = venv/"bin/python"
    if venv.exists():
        if not python.is_file():
            raise TuningError(f"Existing --venv is not a Linux Python environment: {venv}")
    else:
        session.run("create-venv", [sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
    session.run("venv-version", [str(python), "-c", f"import sys; assert sys.version_info[:2] == {sys.version_info[:2]!r}, 'venv Python version mismatch'"])
    session.env["VIRTUAL_ENV"] = str(venv)
    session.env["PATH"] = str(venv/"bin") + os.pathsep + session.env.get("PATH", "")
    pip = [str(python), "-m", "pip", "--disable-pip-version-check", "install"]
    if args.deps == "offline":
        pip += ["--no-index", "--find-links", str(args.wheelhouse.expanduser().resolve())]
        session.run("install-offline-deps", pip + ["-r", str(ROOT/"requirements/910b.txt"), "torch==2.7.1"])
    else:
        session.run("install-deps", pip + ["-r", str(ROOT/"requirements/910b.txt")])
        session.run("install-torch", pip + ["torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
    return str(python)


def prepare_configs(args, output):
    configs = {}
    for kind, filename in workload_cases(args):
        config = load_config(args.config_dir/filename)
        if config["backend"] != "ascend" or config["operator"]["kind"] != kind.split("-")[0]:
            raise TuningError(f"{kind} input config must describe the matching Ascend operator")
        config["ascend"].update(soc_version=args.soc, device=args.device)
        config["agent"] = {"kind": "dummy"}
        config["measurement"]["timeout_s"] = args.timeout
        if baseline_only(config):
            path = output/"config"/(kind+".json")
            write_json(path, config)
            load_config(path)
            configs[kind] = path
            continue
        smoke = deepcopy(config)
        smoke["space"] = {name: [values[0]] for name, values in config["space"].items()}
        if official_enabled(config):
            smoke["space"] = {k: [v] for k, v in default_params(config).items()}
        else:
            smoke["space"]["block_dim"] = [1]
        for suffix, value in ((kind, config), (kind+"-smoke", smoke)):
            path = output/"config"/(suffix+".json")
            write_json(path, value)
            load_config(path)
            configs[suffix] = path
    return configs


def operator_kinds(args):
    return ("add", "matmul") if args.operator == "all" else (args.operator,)


def workload_cases(args):
    cases = []
    matmul_stem = "batchmatmul" if args.matmul_implementation == "batch-v3" else "matmul"
    for kind in operator_kinds(args):
        if kind == "add":
            cases.append((kind, "ascend910b_add.json"))
            continue
        if args.matmul_case in ("2d", "all"):
            cases.append((kind, f"ascend910b_{matmul_stem}.json"))
        for mode in ("shared_rhs", "batched_rhs"):
            if args.matmul_case in (mode, "3d", "all"):
                cases.append(("matmul-"+mode.replace("_", "-"), "ascend910b_"+matmul_stem+"_"+mode+".json"))
    return cases


def verify_baseline_result(directory):
    result = read_json(directory/"baseline_summary.json")
    config = read_json(directory/"manifest.json")["config"]
    inputs, output = shapes(config["operator"])
    if (result.get("status") != "ok" or result.get("mode") != "ascend"
            or result.get("correctness", {}).get("passed") is not True
            or result.get("correctness", {}).get("simulated", False)
            or result.get("input_shapes") != inputs or result.get("output_shape") != output
            or len(result.get("samples_us", [])) != config["measurement"]["repeats"]
            or not result.get("om_sha256") or not result.get("csv_files")
            or not result.get("roofline") or not (directory/"report.html").is_file()):
        raise TuningError(f"{directory.name}: invalid real baseline or missing shape/sample evidence")
    return {"status": "passed", "kind": "baseline_only", "tuning_supported": False,
            "operator_identity": result.get("operator_identity", {}),
            "selected": "baseline", "input_shapes": inputs, "output_shape": output,
            "task_metrics": {"baseline_us": result["duration_us"]},
            "roofline": result["roofline"], "report": str(directory/"report.html")}


def verify_result(directory, smoke):
    result = read_json(directory/"summary.json")
    def valid(trial):
        return (trial.get("status") == "ok" and trial.get("mode") == "ascend"
                and trial.get("correctness", {}).get("passed") is True
                and not trial.get("correctness", {}).get("simulated", False))
    history = result.get("history", [])
    confirmation = result.get("confirmation", {}).get("trials", [])
    config = read_json(directory/"manifest.json")["config"]
    if result.get("routing", {}).get("tuning_supported") is False:
        verified = verify_baseline_result(directory)
        verified.update(kind="tuning_unavailable", reason=result["routing"]["reason"],
                        operator_identity=result["routing"]["operator_identity"])
        return verified
    rounds = config["measurement"]["confirmation_rounds"]
    roles = [role for i in range(rounds) for role in confirmation_roles(config, i)]
    expected = len(roles)
    if result.get("mode") != "ascend" or not valid(result.get("baseline", {})):
        raise TuningError(f"{directory.name}: invalid real baseline")
    if official_enabled(config) and not valid(result.get("source_baseline", {})):
        raise TuningError(f"{directory.name}: invalid official source baseline")
    good = [t for t in history if valid(t)]
    if not good or (smoke and (len(history) != 1 or len(good) != 1 or rounds != 2)):
        raise TuningError(f"{directory.name}: candidate execution/correctness failed")
    if len(confirmation) != expected or not all(valid(t) for t in confirmation) or [t.get("role") for t in confirmation] != roles:
        raise TuningError(f"{directory.name}: incomplete or failed confirmation; inspect confirmation/")
    if official_enabled(config):
        source_trials = [result["source_baseline"], *good, *(t for t in confirmation if t["role"] != "baseline")]
        if any(t.get("selection", {}).get("commit") != COMMIT or not t.get("selection", {}).get("effective_tiling") for t in source_trials):
            raise TuningError(f"{directory.name}: official source identity/effective tiling evidence missing")
    if batch_enabled(config):
        if result.get("primary_baseline") != "source_baseline" or result.get("selected") not in ("candidate", "source_baseline"):
            raise TuningError(f"{directory.name}: BatchMatMulV3 primary-baseline policy violated")
        all_trials = [result["baseline"], result["source_baseline"], *good, *confirmation]
        for trial in all_trials:
            expected_op = "BatchMatMulV2" if trial.get("role") == "baseline" else "BatchMatMulV3"
            inputs, output = shapes(config["operator"])
            if (trial.get("operator_identity", {}).get("Op Type") != expected_op
                    or trial.get("input_shapes") != inputs or trial.get("output_shape") != output
                    or len(trial.get("samples_us", [])) != config["measurement"]["repeats"]):
                raise TuningError(f"{directory.name}: BatchMatMulV3 shape/operator/sample evidence invalid")
            if trial.get("role") != "baseline" and trial["selection"].get("operator") != "BatchMatMulV3":
                raise TuningError(f"{directory.name}: wrong official source operator")
    return {"status": "passed", "successful_candidates": len(good), "failed_candidates": len(history)-len(good),
            "selected": result["selected"], "selected_params": result["selected_params"],
            "reason": result["confirmation"]["reason"], "report": str(directory/"report.html"),
            "task_metrics": {k: result["confirmation"][k] for k in ("baseline_us", "source_baseline_us", "candidate_us", "speedup", "source_speedup", "primary_baseline_us", "primary_speedup", "stock_reference_us", "stock_speedup", "effective_tiling_changed", "tuning_accepted") if k in result["confirmation"]}}


def execute_suite(session, args, python, configs):
    cli = [python, "-m", "paretopilot"]
    kinds = [name for name, _ in workload_cases(args)]
    doctor_config = configs[kinds[-1]]
    preflight = read_json(doctor_config)
    if same_operator(preflight):
        preflight["implementation"] = {"kind": "baseline_only"}
        preflight["space"] = {}
        doctor_config = session.root/"config/preflight.json"
        write_json(doctor_config, preflight)
    session.run("doctor", cli+["doctor", "--config", str(doctor_config), "--output", str(session.root/"environment.json")])
    # Required imports prevent optional dependency tests from silently skipping.
    session.run("python-dependencies", [python, "-c", "import acl,numpy,torch,onnx; print('PyACL:',acl.__file__); print('numpy/torch/onnx:',numpy.__version__,torch.__version__,onnx.__version__)"])
    session.run("python-packages", [python, "-c", "from importlib.metadata import distributions; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in distributions() if d.metadata['Name'])))"])
    for kind in kinds:
        session.run("validate-"+kind, cli+["validate", "--config", str(configs[kind])])
    session.run("unit-tests", [python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    for kind in operator_kinds(args):
        session.run("simulation-"+kind, cli+["run", "--config", str(ROOT/f"examples/simulation_{kind}.json"), "--output", str(session.root/("simulation-"+kind))])
    unavailable = set()
    for kind in kinds:
        if batch_enabled(read_json(configs[kind])):
            continue  # The source baseline is built inside each run before any candidate.
        session.run(kind+"-baseline", cli+["baseline", "--config", str(configs[kind]), "--output", str(session.root/(kind+"-baseline"))])
        routing_path = session.root/(kind+"-baseline")/"routing.json"
        routing = read_json(routing_path) if routing_path.is_file() else {}
        if baseline_only(read_json(configs[kind])) or routing.get("tuning_supported") is False:
            session.data["current_check"] = "verify-"+kind+"-baseline"
            session.save()
            session.data["results"][kind+"-baseline"] = verify_baseline_result(session.root/(kind+"-baseline"))
            if routing.get("tuning_supported") is False:
                unavailable.add(kind)
                session.data["results"][kind+"-baseline"].update(
                    kind="tuning_unavailable", reason=routing["reason"], operator_identity=routing["operator_identity"])
            session.data.pop("current_check", None)
            session.save()
    phases = ("smoke", "full") if args.scope == "full" else ("smoke",)
    for phase in phases:
        for kind in kinds:
            if baseline_only(read_json(configs[kind])) or kind in unavailable:
                continue
            name = kind+"-"+phase
            directory = session.root/name
            command = cli+["run", "--config", str(configs[name if phase == "smoke" else kind]), "--output", str(directory)]
            session.run(name, command+(["--smoke"] if phase == "smoke" else []))
            session.data["current_check"] = "verify-"+name
            session.save()
            session.data["results"][name] = verify_result(directory, phase == "smoke")
            if session.data["results"][name].get("kind") == "tuning_unavailable":
                unavailable.add(kind)
            session.data.pop("current_check", None)
            session.save()


def diagnostics(root, excluded=()):
    archive = root/"diagnostics.zip"
    excluded = {p.resolve() for p in excluded}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for directory, dirs, files in os.walk(root):
            directory = Path(directory)
            dirs[:] = sorted(d for d in dirs if d not in {"project", "opp", "__pycache__"} and not d.startswith(".venv")
                             and not (directory/d).is_symlink() and (directory/d).resolve() not in excluded)
            for name in sorted(files):
                path = directory/name
                if not path.is_symlink() and path.suffix.lower() in {".log", ".json", ".jsonl", ".csv", ".md", ".html", ".svg", ".txt"}:
                    out.write(path, path.relative_to(root).as_posix())
    return archive


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not re.fullmatch(r"Ascend910B[A-Za-z0-9]*", args.soc):
        p.error("--soc must be the exact Ascend910B variant")
    if args.operator == "add" and args.matmul_case != "2d":
        p.error("--matmul-case requires --operator matmul or all")
    if args.deps == "offline" and (args.wheelhouse is None or not args.wheelhouse.is_dir()):
        p.error("--deps offline requires an existing --wheelhouse directory")
    if args.deps != "offline" and args.wheelhouse is not None:
        p.error("--wheelhouse requires --deps offline")
    if not sys.platform.startswith("linux") or sys.version_info < (3, 10):
        p.error("Run this script on the Linux test machine with Python 3.10+")
    try:
        root = create_output(args.output)
    except (OSError, TuningError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    session = Session(root, args)
    code = 0
    print(f"Test output: {root}", flush=True)
    try:
        cann, script, versions = resolve_cann(args.cann_path, args.cann_env)
        write_json(root/"cann.json", {"root": str(cann), "env_script": str(script), "version_records": versions})
        session.env = source_environment(cann, script, root/"cann-env.log")
        configs = prepare_configs(args, root)
        selected_configs = [read_json(configs[name]) for name, _ in workload_cases(args)]
        commands = {c["ascend"].get(tool, tool) for c in selected_configs for tool in ("atc", "msprof")}
        if any(not baseline_only(c) and not same_operator(c) for c in selected_configs):
            commands.update(("cmake", "g++", "make"))
        commands.update(c["ascend"].get("msopgen", "msopgen") for c in selected_configs
                        if not official_enabled(c) and not baseline_only(c))
        for name in sorted(commands | {"npu-smi"}):
            if not shutil.which(name, path=session.env.get("PATH")):
                raise TuningError(f"Required executable missing: {name}; check CANN Toolkit/ops and build tools")
        session.run("npu-smi", ["npu-smi", "info"])
        if shutil.which("git", path=session.env.get("PATH")) and (ROOT/".git").exists():
            session.run("git-commit", ["git", "rev-parse", "HEAD"])
        python = prepare_python(session, args)
        session.data["python"] = python
        execute_suite(session, args, python, configs)
        session.data["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
        session.data.update(status="failed", error=str(exc) or "Interrupted")
        print(f"\nFAILED: {session.data['error']}", file=sys.stderr)
    finally:
        session.data["exit_code"] = code
        session.save()
        try:
            print(f"Diagnostics: {diagnostics(root, excluded=(args.venv,))}", flush=True)
        except Exception as exc:
            print(f"Could not create diagnostics ZIP: {exc}; raw files remain at {root}", file=sys.stderr)
    print(f"{session.data['status'].upper()}: {root/'test-summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
