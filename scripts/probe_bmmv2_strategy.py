#!/usr/bin/env python3
"""Experimental installed V2 build-strategy comparison; no dependency installation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paretopilot.bmmv2_probe import run_probe
from paretopilot.config import load_config
from paretopilot.util import TuningError, write_json
from scripts.test_910b import resolve_cann, source_environment, create_output, positive, nonnegative, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-path", type=Path, required=True)
    parser.add_argument("--cann-env", type=Path)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--device", type=nonnegative, default=0)
    parser.add_argument("--deps", choices=("check",), default="check")
    parser.add_argument("--config", type=Path, default=ROOT/"examples/ascend910b_matmul.json")
    parser.add_argument("--timeout", type=positive, default=3600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Run on the Linux 910B2/CANN 8.5.0 machine")
    if getattr(os, "geteuid", lambda: None)() == 0:
        parser.error("Run as the normal Conda environment user, without sudo")
    root = None
    try:
        config = load_config(args.config)
        if config["backend"] != "ascend" or config["operator"]["kind"] != "matmul":
            raise TuningError("Probe requires an Ascend MatMul workload")
        config["implementation"], config["space"], config["agent"] = {"kind": "baseline_only"}, {}, {"kind": "dummy"}
        config["ascend"].update(soc_version=args.soc, device=args.device)
        config["measurement"].update(timeout_s=args.timeout, confirmation_rounds=2)
        if args.output is None:
            (ROOT/"runs").mkdir(exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix="bmmv2-strategy-", dir=ROOT/"runs"))
        else:
            root = create_output(args.output)
        write_json(root/"config.json", config)
        config = load_config(root/"config.json")
        print(f"Probe output: {root}", flush=True)
        cann, script, versions = resolve_cann(args.cann_path, args.cann_env)
        os.environ.update(source_environment(cann, script, root/"cann-env.log"))
        write_json(root/"cann.json", {"root": str(cann), "versions": versions})
        result = run_probe(config, root)
        print(json.dumps({"status": result["status"], "selected": result["selected"],
                          "code_generation_difference_verified": result["code_generation_difference_verified"],
                          "manual_tiling_control": False, "metrics": result["confirmation"], "report": str(root/"report.md")}, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if root:
            write_json(root/"failure.json", {"status": "failed", "error": str(exc) or "Interrupted"})
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    finally:
        if root:
            print(f"Diagnostics: {diagnostics(root)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
