#!/usr/bin/env python3
"""Inspect installed BatchMatMulV2 registration and replay a baseline ATC build."""
import argparse
import ast
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paretopilot.util import TuningError, read_json, run_command, write_json
from scripts.test_910b import resolve_cann, source_environment, positive

OPERATOR = "BatchMatMulV2"
MATCH = re.compile(r"BatchMatMulV2|batch_?mat_?mul_?v2|oppath|op_compile_stats|"
                   r"kernel.?name|tiling.?key|opFile|opInterface|reuse (?:binary|disk|task)|dispatch online", re.I)


def installed_files(roots, warnings):
    """Follow CANN symlinks, visiting each real directory/file only once."""
    directories, files = set(), set()
    for root in roots:
        for current, dirs, names in os.walk(root, followlinks=True, onerror=lambda e: warnings.append(str(e))):
            directory = Path(current).resolve()
            if directory in directories:
                dirs[:] = []
                continue
            directories.add(directory)
            for name in names:
                path = (directory/name).resolve()
                if path not in files and path.is_file():
                    files.add(path)
                    yield path


def registration(path):
    if path.stat().st_size > 32*1024*1024:
        raise TuningError(f"Registration file exceeds 32 MiB: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix == ".json":
        value = json.loads(text)
        return value.get(OPERATOR) if isinstance(value, dict) else None
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.optionxform = str
    ini.read_string(text)
    return dict(ini[OPERATOR]) if ini.has_section(OPERATOR) else None


def file_record(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return {"path": str(path), "size": path.stat().st_size, "sha256": hasher.hexdigest()}


def python_details(path):
    """Read AST without importing/executing installed compiler modules."""
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    return {"imports": [{"line": n.lineno, "statement": ast.get_source_segment(source, n)}
                        for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))],
            "functions": [{"line": n.lineno, "name": n.name, "arguments": ast.unparse(n.args)}
                          for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
            "compile_and_tiling_lines": [{"line": i, "text": line.strip()} for i, line in enumerate(source.splitlines(), 1)
                                         if re.search(r"compile_op|auto_schedule|build_code|cce_build|tiling|impl_mode|register_operator", line)][:120]}


def inspect_installation(roots, warnings):
    files = list(installed_files(roots, warnings))
    registrations, priority = [], []
    stems = {"batch_matmul_v2", "batch_mat_mul_v2", "batch_matmul"}
    for path in files:
        if path.name == "config.ini" and path.parent.name == "vendors":
            priority.append({"path": str(path), "text": path.read_text(errors="replace")[:16384]})
        if "ops-info" not in path.name or path.suffix not in (".json", ".ini"):
            continue
        try:
            entry = registration(path)
            if entry is None:
                continue
            registrations.append({**file_record(path), "entry": entry})
            value = entry.get("opFile", entry.get("opFile.value", {}))
            if isinstance(value, dict):
                value = value.get("value")
            if isinstance(value, str):
                stems.add(Path(value).stem)
        except (OSError, ValueError, configparser.Error) as exc:
            warnings.append(f"{path}: {exc}")
    candidates = []
    for path in files:
        # These are candidates only. The trace/logs determine actual use.
        if not any(path.name == stem or path.name.startswith(stem+".") for stem in stems):
            continue
        try:
            item = file_record(path)
            if path.suffix == ".py" and path.stat().st_size <= 4*1024*1024:
                try:
                    item["python"] = python_details(path)
                except (SyntaxError, ValueError) as exc:
                    item["parse_error"] = str(exc)
            candidates.append(item)
        except OSError as exc:
            warnings.append(str(exc))
    return {"operator": OPERATOR, "registrations": registrations,
            "candidate_files": candidates, "vendor_priority": priority}


def replay_command(baseline, output):
    record = read_json(baseline/"atc.command.json")
    argv = record.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise TuningError("Invalid baseline atc.command.json argv")
    if Path(argv[0]).name not in ("atc", "atc.exe") or "--framework=5" not in argv:
        raise TuningError("Expected a recorded ONNX ATC command")
    models = [a.split("=", 1)[1] for a in argv if a.startswith("--model=")]
    if len(models) != 1:
        raise TuningError("Expected exactly one recorded --model= argument")
    model = Path(models[0])
    if not model.is_absolute():
        model = Path(record.get("cwd", baseline))/model
    model = model.resolve()
    if not model.is_file():
        raise TuningError(f"Baseline ONNX no longer exists: {model}")
    argv = [a for a in argv if not a.startswith(("--model=", "--output=", "--log=", "--debug_dir="))]
    return argv+["--model="+str(model), "--output="+str(output/"model"), "--log=debug"], file_record(model)


def diagnostic_environment(env, output):
    env = env.copy()
    env.pop("ASCEND_CUSTOM_OPP_PATH", None)  # Same isolation as the stock baseline.
    env.pop("ASCEND_MODULE_LOG_LEVEL", None)
    env["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    env["DUMP_GE_GRAPH"] = "2"
    for key, folder in (("ASCEND_PROCESS_LOG_PATH", "logs"), ("DUMP_GRAPH_PATH", "graphs"),
                        ("ASCEND_WORK_PATH", "work"), ("ASCEND_CACHE_PATH", "cache"),
                        ("NPU_COLLECT_PATH", "collect")):
        path = output/folder
        path.mkdir()
        env[key] = str(path)
    return env


def traced_files(path):
    """Only successful file opens/execs; failed searches are not dependencies."""
    found = set()
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if re.search(r"=\s*-1\b", line):
                continue
            match = re.search(r"=\s*\d+<(/[^>]+)>", line)
            if match:
                found.add(match.group(1))
            elif re.search(r"execve\(.*=\s*0\s*$", line):
                match = re.search(r'execve\("(/[^"\\]+)"', line)
                if match:
                    found.add(match.group(1))
    return sorted(found)


def collect_evidence(output, opened, warnings):
    # A loaded library is a process dependency, not proof it implements this op.
    libraries = [p for p in opened if ".so" in Path(p).name]
    write_json(output/"opened-files.json", {"files": opened, "process_libraries": libraries})
    readelf = shutil.which("readelf")
    with (output/"library-dependencies.txt").open("w", encoding="utf-8") as out:
        if not readelf:
            out.write("readelf unavailable; inspect opened-files.json and trace-files.log.\n")
        for name in libraries:
            if not readelf or not re.search(r"optiling|ophost|opsproto|op_proto|tbe", name, re.I):
                continue
            try:
                result = subprocess.run([readelf, "-d", name], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=15)
                out.write(f"\n{name}\n"+"\n".join(line for line in result.stdout.splitlines()
                                                   if any(k in line for k in ("NEEDED", "RPATH", "RUNPATH")))+"\n")
            except (OSError, subprocess.TimeoutExpired) as exc:
                warnings.append(str(exc))
    with (output/"log-matches.txt").open("w", encoding="utf-8") as out:
        count = 0
        for path in sorted(output.rglob("*")):
            if (not path.is_file() or path.suffix not in (".log", ".txt", ".pbtxt")
                    or path.name in ("log-matches.txt", "trace-files.log", "report.txt", "library-dependencies.txt")):
                continue
            with path.open(encoding="utf-8", errors="replace") as stream:
                for i, line in enumerate(stream, 1):
                    if MATCH.search(line):
                        out.write(f"{path.relative_to(output)}:{i}: {line[:3000].rstrip()}\n")
                        count += 1
                        if count >= 1000:
                            out.write("Excerpt limited to 1000 matches; full logs remain in this directory.\n")
                            return


def summarize(output):
    """Read independent evidence files without one unreadable log hiding the rest."""
    output = Path(output).expanduser().resolve()
    for name in ("report.json", "atc.log", "registration-and-sources.json"):
        print(f"\n=== {name} ===")
        try:
            text = (output/name).read_text(encoding="utf-8", errors="replace")
            if name == "atc.log":
                print("\n".join(text.splitlines()[-120:]))
            elif name == "registration-and-sources.json":
                data = json.loads(text)
                selected = {"registrations": [r for r in data["registrations"] if "/ascend910b/" in r["path"]],
                            "candidate_files": [r for r in data["candidate_files"]
                                                if "/impl/ops_legacy/" in r["path"]
                                                and r["path"].endswith("/batch_matmul_v2.py")]}
                print(json.dumps(selected, ensure_ascii=False, indent=2))
            else:
                print(text)
        except (OSError, ValueError, KeyError) as exc:
            print(f"Cannot read this evidence file: {exc}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-path", type=Path)
    parser.add_argument("--cann-env", type=Path)
    parser.add_argument("--baseline-dir", type=Path, help="Existing stock baseline/ with atc.command.json")
    parser.add_argument("--summarize", type=Path, metavar="DIAGNOSTIC_DIR", help="Read an existing report without rerunning ATC")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-only", action="store_true", help="Inspect installed files without replaying ATC")
    parser.add_argument("--no-strace", action="store_true")
    parser.add_argument("--timeout", type=positive, default=3600)
    args = parser.parse_args(argv)
    if args.summarize:
        return summarize(args.summarize)
    if not args.cann_path or not args.baseline_dir:
        parser.error("--cann-path and --baseline-dir are required unless --summarize is used")
    if not sys.platform.startswith("linux"):
        parser.error("Run on the Linux CANN test machine")
    if getattr(os, "geteuid", lambda: None)() == 0 and os.environ.get("SUDO_USER"):
        parser.error("Run without sudo in the activated Conda environment. sudo changes Python/dependency resolution and output ownership; this inspection does not require root.")
    (ROOT/"runs").mkdir(exist_ok=True)
    output = args.output.expanduser().resolve() if args.output else Path(tempfile.mkdtemp(prefix="inspect-bmmv2-", dir=ROOT/"runs"))
    if output.is_relative_to(args.cann_path.expanduser().resolve()):
        parser.error("Diagnostic output must be outside the CANN installation")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    report = {"operator": OPERATOR, "output": str(output), "warnings": [], "atc_status": "not_run", "trace_status": "not_run",
              "python": {"executable": sys.executable, "prefix": sys.prefix,
                         "effective_uid": getattr(os, "geteuid", lambda: None)()}}
    code = 0
    try:
        cann, script, versions = resolve_cann(args.cann_path, args.cann_env)
        env = source_environment(cann, script, output/"cann-env.log")
        report["python"]["path_python3"] = shutil.which("python3", path=env.get("PATH"))
        report["cann"] = {"root": str(cann), "versions": versions}
        report["original_custom_opp"] = env.get("ASCEND_CUSTOM_OPP_PATH")
        roots = [cann]
        if env.get("ASCEND_OPP_PATH"):
            roots.append(Path(env["ASCEND_OPP_PATH"]))
        print("Inspecting installed operator registration and Python entry points...", flush=True)
        inventory = inspect_installation(roots, report["warnings"])
        write_json(output/"registration-and-sources.json", inventory)
        report["registration_files"] = [v["path"] for v in inventory["registrations"]]
        report["candidate_files"] = [v["path"] for v in inventory["candidate_files"]]
        baseline = args.baseline_dir.expanduser().resolve()
        if (baseline/"trial.json").is_file():
            trial = read_json(baseline/"trial.json")
            if trial.get("role") != "baseline":
                raise TuningError("--baseline-dir must be the stock baseline, not source_baseline or a candidate")
            report["previous_baseline"] = {k: trial.get(k) for k in ("role", "status", "duration_us", "om_sha256")}
        command, model = replay_command(baseline, output)
        report["model"] = model
        atc = shutil.which(command[0], path=env.get("PATH"))
        if not atc:
            raise TuningError(f"Recorded ATC executable is unavailable: {command[0]}")
        command[0] = atc
        report["atc_executable"] = file_record(Path(atc).resolve())
        env = diagnostic_environment(env, output)
        report["environment"] = {k: env[k] for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
                                  "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH", "DUMP_GE_GRAPH", "DUMP_GRAPH_PATH", "ASCEND_PROCESS_LOG_PATH") if k in env}
        if not args.scan_only:
            tracer = None if args.no_strace else shutil.which("strace", path=env.get("PATH"))
            report["trace_status"] = "disabled" if args.no_strace else "unavailable"
            if tracer:
                try:
                    run_command([tracer, "-f", "-s", "4096", "-yy", "-e", "trace=open,openat,execve",
                                 "-o", str(output/"trace-probe.txt"), sys.executable, "-c", "pass"],
                                output, output/"strace-probe.log", 30, env)
                    tracer = [tracer, "-f", "-s", "4096", "-yy", "-e", "trace=open,openat,execve",
                              "-o", str(output/"trace-files.log")]
                    report["trace_status"] = "enabled"
                except (TuningError, OSError, subprocess.SubprocessError) as exc:
                    report["warnings"].append("strace probe failed; using ATC logs only: "+str(exc))
                    report["trace_status"], tracer = "probe_failed", None
            write_json(output/"replay-atc.json", {"argv": command, "cwd": str(output)})
            print(f"Replaying baseline ATC with diagnostic logs (strace: {report['trace_status']})...", flush=True)
            report["atc_status"] = "failed"
            run_command((tracer or [])+command, output, output/"atc.log", args.timeout, env)
            report["atc_status"] = "succeeded"
        opened = traced_files(output/"trace-files.log")
        collect_evidence(output, opened, report["warnings"])
        report["opened_candidate_files"] = [p for p in report["candidate_files"] if p in opened]
    except (OSError, TuningError, subprocess.SubprocessError, ValueError) as exc:
        report["error"], code = str(exc), 2
        collect_evidence(output, traced_files(output/"trace-files.log"), report["warnings"])
    finally:
        report["exit_code"] = code
        write_json(output/"report.json", report)
        text = json.dumps(report, ensure_ascii=False, indent=2)+"\n\n"
        text += ("Static registration/source matches are candidates, not proof of use.\n"
                 "Successful opens show process file access, not execution of every function in that file.\n"
                 "Correlate log-matches.txt with registration-and-sources.json and opened-files.json.\n"
                 "An imported .py wrapper does not establish that tiling/kernel source is available.\n"
                 "A new diagnostic cache does not prevent reuse of installed precompiled binaries.\n"
                 "No inference benchmark was run. Diagnostic timings are not performance results.\n")
        (output/"report.txt").write_text(text, encoding="utf-8")
        print(f"Report: {output/'report.txt'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
