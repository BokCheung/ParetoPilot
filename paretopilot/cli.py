"""Kernel-only command-line interface."""
import argparse
import json
import sys
from . import __version__
from .artifacts import ConfigError
from .config import load_config
from .profiling import read_profile
from .roofline import analyze
from .search import tune_kernel
from .space import candidates, legal


def main(argv=None):
    parser = argparse.ArgumentParser(prog="paretopilot", description="Roofline-guided Ascend kernel genetic optimization using OM profiling")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "space", "analyze", "tune-kernel"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "analyze":
            p.add_argument("--csv", help="Optional baseline op_summary CSV")
        if name == "tune-kernel":
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        c = load_config(args.config)
        pool = candidates(c)
        if not pool or not legal(c["baseline_params"], c):
            raise ConfigError("No legal candidates or invalid baseline template parameters")
        if args.command == "validate":
            result = {"status": "valid", "mode": c["mode"], "operator": c["operator"], "legal_candidates": len(pool)}
        elif args.command == "space":
            result = {"legal_candidates": len(pool), "parameters": c["space"], "candidates": pool}
        elif args.command == "analyze":
            metrics = read_profile(args.csv, c) if args.csv else {}
            result = {**metrics, "roofline": analyze(c, metrics.get("duration_us"))}
        else:
            result = tune_kernel(args.config, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, OSError) as exc:
        print(f"paretopilot: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; partial stage records remain in the output directory.", file=sys.stderr)
        return 130
