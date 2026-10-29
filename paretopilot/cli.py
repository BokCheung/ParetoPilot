import argparse
import json
import sys
import subprocess
from . import __version__
from .config import load_config
from .doctor import inspect_environment
from .package import create_package
from .profiling import inspect_profiles, read_profile
from .roofline import analyze
from .official import enabled as official_enabled, batch_enabled, primary_source, add_enabled, default_params
from .search import run
from .space import enumerate_space
from .workload import baseline_only
from .util import TuningError, write_json


def parser():
    p = argparse.ArgumentParser(description="Offline Ascend kernel tuning; no Agent service required")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("validate", "doctor", "roofline", "baseline", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        if name in ("baseline", "run"):
            cmd.add_argument("--output", required=True, help="New empty experiment directory")
            cmd.add_argument("--execution", choices=("om", "kernel"), help="Kernel: direct BatchMatMulV3 search, fresh OM confirmation")
            cmd.add_argument("--smoke", action="store_true", help="BatchMatMulV3: source baseline and one explicit-tile check when available; other adapters: one candidate and two confirmation rounds")
            cmd.add_argument("--fresh-search", action="store_true", help="Ignore historical tuning parameters and perform a fresh search")
        if name == "doctor":
            cmd.add_argument("--output", help="Optional JSON environment record")
    cmd = sub.add_parser("inspect-profile", help="List exact task identifiers and counts in a CSV")
    cmd.add_argument("csv", nargs="+")
    cmd = sub.add_parser("import-profile", help="Validate and analyze an existing target-op CSV")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--role", choices=("baseline", "source_baseline", "candidate"), required=True)
    cmd.add_argument("csv", nargs="+")
    cmd = sub.add_parser("package", help="Create a source ZIP to transfer to the test machine")
    cmd.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "package":
            result = create_package(args.output)
        elif args.command == "inspect-profile":
            result = inspect_profiles(args.csv)
        else:
            c = load_config(args.config)
            if args.command == "validate":
                count = len(enumerate_space(c))
                if not count and not baseline_only(c):
                    raise TuningError("No legal template configurations")
                result = {"valid": True, "backend": c["backend"], "legal_candidates": count, "agent": c["agent"]["kind"]}
                if c["search"].get("domain_mode") == "shape":
                    result.update(domain_mode="shape", candidate_count_basis="configured preview; regenerated after verified platform query")
            elif args.command == "doctor":
                result = inspect_environment(c)
                if args.output:
                    write_json(args.output, result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["ready"] else 2
            elif args.command == "roofline":
                result = analyze(c)
            elif args.command == "import-profile":
                m = c["measurement"]
                result = read_profile(args.csv, m["selectors"][args.role], m["repeats"], m.get("duration_column", "Task Duration(us)"), m.get("traffic_column"))
                result["roofline"] = analyze(c, result["duration_us"], result["measured_bytes"])
            else:
                if args.fresh_search:
                    c["search"]["reuse_results"] = False
                if args.execution:
                    if args.execution == "kernel" and (not primary_source(c)
                                                       or not c["implementation"].get("build_cache", True)):
                        raise TuningError("--execution kernel requires BatchMatMulV3 and build_cache=true")
                    c["implementation"]["execution"] = args.execution
                if c.get("implementation", {}).get("execution") == "kernel" and args.command == "baseline":
                    raise TuningError("Use run --smoke --execution kernel for a direct source baseline")
                if args.smoke:
                    c["search"].update(population=2, generations=1, budget=1)
                    c["measurement"]["confirmation_rounds"] = 2
                    if official_enabled(c) and not add_enabled(c):
                        c["space"] = {k: [v] for k, v in default_params(c).items()}
                if c["backend"] == "ascend" and not sys.platform.startswith("linux"):
                    raise TuningError("Ascend backend requires the Linux 910B/CANN test machine; use simulation here")
                def progress(t, count):
                    print(f"[{count}/{c['search']['budget']}] {t['status']}: "+(f"{t['duration_us']:.4f} us" if t['status']=='ok' else t['error']), file=sys.stderr, flush=True)
                result = run(c, args.output, progress, baseline_only=args.command == "baseline", smoke=args.smoke)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if isinstance(result, dict) and result.get("successful_candidates") == 0 else 0
    except (TuningError, OSError, ImportError, subprocess.SubprocessError) as exc:
        print(f"ParetoPilot: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed trial artifacts were retained.", file=sys.stderr)
        return 130
