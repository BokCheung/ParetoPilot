"""Command-line entry points for configuration validation, analysis and search."""

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, load_json, normalize_inputs, validate_search
from .graph import build_graph
from .native_config import load_model, normalize_native
from .native import NativeBuilder, evaluate_native
from .report import write_analysis, write_search
from .roofline import evaluate
from .search import run_search
from .store import RunStore


def parser():
    p = argparse.ArgumentParser(prog="paretopilot", description="NPU Roofline-guided recommendation model structure search")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("validate", "analyze", "search"):
        c = sub.add_parser(command)
        c.add_argument("--model", required=True, help="Model JSON or static native Jsonnet configuration")
        c.add_argument("--npu", required=True, help="NPU JSON specification")
        c.add_argument("--workload", required=True, help="Workload JSON configuration")
        if command != "analyze":
            c.add_argument("--space", required=(command == "search"), help="Search space JSON configuration")
        if command != "validate":
            c.add_argument("--output", required=True, help="New run directory")
            c.add_argument("--resume", action="store_true", help="Replay same experiment and reuse completed evaluations")
            c.add_argument("--quiet", action="store_true", help="Suppress search progress on stderr")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        model = load_model(args.model)
        native = isinstance(model, dict) and model.get("format") == "native_nodes"
        normalize = normalize_native if native else normalize_inputs
        model, npu, workload = normalize(model, load_json(args.npu), load_json(args.workload))
        if native and (args.command == "search" or getattr(args, "space", None)):
            raise ConfigError("Native node graphs support validate/analyze; structure search currently uses the compositional JSON schema")
        search = validate_search(load_json(args.space), model) if getattr(args, "space", None) else None
        if args.command == "validate":
            graph = NativeBuilder(model, workload).build() if native else build_graph(model, workload)
            print(json.dumps({"status": "valid", "operator_count": len(graph.operators),
                              "parameter_count": graph.parameter_count,
                              "note": "Schema and graph checks only; run analyze for hardware feasibility."}, indent=2))
            return 0
        inputs = {"command": args.command, "model": model, "npu": npu, "workload": workload, "search": search}
        store = RunStore(args.output, inputs, resume=args.resume)
        if args.command == "analyze":
            result = store.evaluate(model, npu, workload, evaluate_native if native else evaluate)
            write_analysis(store.root, result)
            summary = {"status": result["status"], "evaluation_scope": result["evaluation_scope"],
                       "metrics": result["metrics"], "violations": result["violations"],
                       "report": str((store.root / "report.md").resolve())}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "ok" else 2

        def progress(trial):
            if not args.quiet:
                print(f"[{trial['trial'] + 1}/{search['max_evaluations']}] {trial['candidate_id'][:10]} {trial['status']}", file=sys.stderr)

        result = run_search(model, npu, workload, search, store=store, progress=progress)
        write_search(store.root, result)
        print(json.dumps({**result["summary"], "evaluation_scope": result["baseline"]["evaluation_scope"],
                          "report": str((store.root / "report.md").resolve())}, ensure_ascii=False, indent=2))
        return 0 if result["pareto"] else 2
    except (ConfigError, OSError) as exc:
        print(f"paretopilot: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed evaluations are saved; rerun with --resume.", file=sys.stderr)
        return 130
