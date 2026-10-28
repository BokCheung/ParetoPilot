"""Diagnose V3 OM task time versus direct task time and event intervals."""
import argparse
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

from .backend import environment, evaluate
from .config import load_config
from .direct_backend import KernelBackend, validate_result
from .official import batch_enabled, default_params
from .profiling import IDENTITY_COLUMNS, inspect_profiles, profile_reader, read_profile
from .util import TuningError, read_json, run_command, write_json


CORE_TASKS = {"AI_CORE", "AI_CUBE", "AI_CUBE_CORE", "AI_VECTOR_CORE", "MIX_AIC"}


def check_block_dims(rows, expected):
    observed, checked, unavailable = [], 0, 0
    for row in rows:
        raw = row.get("Block Dim")
        value = "" if raw is None else str(raw).strip()
        if value not in observed:
            observed.append(value)
        if value.upper() in {"", "N/A", "NA", "-", "--", "NONE", "NULL", "UNKNOWN"}:
            unavailable += 1
            continue
        try:
            number = Decimal(value)
            if not number.is_finite() or number <= 0 or number != number.to_integral_value():
                raise ValueError()
        except (InvalidOperation, ValueError) as exc:
            raise TuningError(f"Invalid profiler Block Dim {raw!r}; expected integer {expected} or an unavailable marker") from exc
        if number != expected:
            raise TuningError(f"Direct profiler Block Dim {raw!r} differs from Host tiling {expected}")
        checked += 1
    return {"status": "matched" if not unavailable else "partial" if checked else "unavailable",
            "expected": expected, "observed": observed, "checked_samples": checked, "unavailable_samples": unavailable}


def direct_profile(paths, repeats, block_dim):
    """The isolated capture contains exactly one launched kernel per iteration.

    Raw launches may have unknown Op Type/Name. Never select by fastest task,
    assume a V3 name, or silently accept extra core tasks/duplicate exports.
    """
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader, _ = profile_reader(stream, path)
            if "Task Type" not in reader.fieldnames:
                raise TuningError("Direct profiling needs Task Type to isolate core tasks; inspect profile_inventory.json")
            rows.extend(row for row in reader if row["Task Type"].strip().upper() in CORE_TASKS)
    if len(rows) != repeats:
        raise TuningError(f"Direct capture expected {repeats} core tasks, got {len(rows)}; inspect profile_inventory.json")
    selector = {k: rows[0][k] for k in (*IDENTITY_COLUMNS, "Task Type") if k in rows[0]}
    if any(any(row.get(k) != v for k, v in selector.items()) for row in rows):
        raise TuningError("Direct capture contains ambiguous core task identities; no target was guessed")
    block_check = check_block_dims(rows, block_dim)
    result = read_profile(paths, selector, repeats, "Task Duration(us)")
    result.update(metric="Direct target-kernel Task Duration(us) median",
                  identity_basis="isolated single-kernel capture; exact count, identity and available Block Dim checked",
                  task_rows=rows, block_dim_check=block_check)
    return result


def tiling_signature(trial):
    records = trial.get("selection", {}).get("effective_tiling", [])
    keys = ("used_core_num", "base_m", "base_n", "base_k", "tiling_key", "block_dim", "multi_batch", "load_mode")
    return sorted({(tuple(r.get(k) for k in keys),
                    tuple(sorted((k, v) for k, v in r.get("cube_plan", {}).items() if k != "raw"))) for r in records})


def diagnose(om, direct, profiled):
    om_us, event_us, task_us = om["duration_us"], direct["duration_us"], profiled["duration_us"]
    ratios = {"direct_task_over_om_task": task_us/om_us, "event_over_direct_task": event_us/task_us}
    findings = []
    if ratios["event_over_direct_task"] >= 2:
        findings.append("event_interval_exceeds_kernel_task: event samples include time outside the profiled kernel task; do not call this pure kernel latency")
    if ratios["direct_task_over_om_task"] >= 2:
        findings.append("direct_kernel_task_slower: profiler also sees a slower direct task; investigate tiling, offline binary/kernel dispatch and launch configuration")
    elif ratios["direct_task_over_om_task"] <= 0.5:
        findings.append("direct_kernel_task_faster: execution paths differ materially; inspect source/tiling/binary evidence")
    else:
        findings.append("task_times_within_factor_two: no order-of-magnitude task-time regression in this capture")
    same_tiling = bool(tiling_signature(om)) and tiling_signature(om) == tiling_signature(direct)
    if not same_tiling:
        findings.append("effective_tiling_differs_or_missing: identical requested parameters did not establish identical effective tiling")
    if profiled.get("block_dim_check", {}).get("status") in ("partial", "unavailable"):
        findings.append("profiler_block_dim_unavailable: some/all exported tasks lack numeric Block Dim; launch dimensions come from Host evidence, not an independent profiler check")
    return {**ratios, "effective_tiling_equal": same_tiling, "findings": findings,
            "limit": "Single captures and factor-two diagnostic heuristics, not a causal proof or tuning acceptance. Event and task samples come from separate runs. Same source does not establish identical OM/direct binaries."}


def complete_comparison(root, summary, config, result):
    direct, om = summary["kernel_event"], summary["om"]
    if tiling_signature(result) != tiling_signature(direct):
        raise TuningError("Profiled and unprofiled direct tiling differ")
    directory = root/"kernel_profile"
    paths = sorted((directory/"profiler").rglob("op_summary*.csv"))
    if not paths:
        raise TuningError("No direct op_summary CSV; inspect kernel_profile/profiler_export.log and runtime.log")
    write_json(directory/"profile_inventory.json", inspect_profiles(paths))
    profile = direct_profile(paths, config["measurement"]["repeats"], result["selection"]["effective_tiling"][0]["block_dim"])
    write_json(directory/"profile_metrics.json", profile)
    summary.pop("error", None)
    summary.update(status="passed", kernel_profile=profile, profiled_kernel_event=result,
                   diagnosis=diagnose(om, direct, profile),
                   metrics={"om_task_p50_us": om["duration_us"], "kernel_task_p50_us": profile["duration_us"],
                            "kernel_event_p50_us": direct["duration_us"], "profiled_kernel_event_p50_us": result["duration_us"]})
    metrics = summary["metrics"]
    lines = ["# V3 execution comparison", "", "| Measurement | P50 (us) |", "|---|---:|",
             *[f"| {key} | {value:.4f} |" for key, value in metrics.items()], "",
             *[f"- {item}" for item in summary["diagnosis"]["findings"]], "", summary["diagnosis"]["limit"], "",
             "See comparison.json, om/selection.json, kernel_event/host_tiling.json,",
             "kernel_profile/kernel_selection.json, profile_metrics.json and both profiler directories."]
    (root/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(root/"comparison.json", summary)
    print("Comparison: "+str(root/"comparison.json"), flush=True)
    print("P50 us: "+str(metrics), flush=True)
    print("Profiler Block Dim: "+str(profile["block_dim_check"]), flush=True)
    return summary


def reanalyze(directory):
    root = Path(directory).resolve()
    summary = read_json(root/"comparison.json")
    bootstrap = read_json(root/"kernel_profile/bootstrap.json")
    result = read_json(root/"kernel_profile/direct_result.json")
    if (summary.get("scope") != "execution_compare"
            or any(summary.get(stage, {}).get("status") != "ok" for stage in ("om", "kernel_event"))):
        raise TuningError("Reanalysis requires completed OM and direct Event measurements")
    validate_result(result, bootstrap["config"], summary["params"], bootstrap["build"])
    summary["analysis_source"] = "existing profiler CSVs; no compilation, device execution or export rerun"
    return complete_comparison(root, summary, bootstrap["config"], result)


def latest_capture(directory):
    candidates = [p.parent for p in Path(directory).resolve().rglob("comparison.json")
                  if (p.parent/"kernel_profile/direct_result.json").is_file()
                  and any((p.parent/"kernel_profile/profiler").rglob("op_summary*.csv"))]
    if not candidates:
        raise TuningError(f"No comparison with exported direct CSVs found under {directory}")
    return max(candidates, key=lambda p: (p/"comparison.json").stat().st_mtime_ns)


def compare(config, output):
    if config["backend"] != "ascend" or not batch_enabled(config):
        raise TuningError("Execution comparison requires the Ascend BatchMatMulV3 source adapter")
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise TuningError("Comparison output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    c = deepcopy(config)
    c["implementation"]["execution"] = "kernel"
    if not c["implementation"].get("build_cache", True):
        raise TuningError("Execution comparison requires build_cache=true")
    write_json(root/"config.json", c)
    params = default_params(c)
    summary = {"status": "running", "scope": "execution_compare", "params": params,
               "operator": c["operator"], "comparison": "same case files, pinned source, parameters, device and repeats; no tuning"}
    write_json(root/"comparison.json", summary)
    backend = None
    try:
        backend = KernelBackend(c, root)
        print("[compare 1/3] Direct kernel, unprofiled Event timing", flush=True)
        direct = evaluate(backend, c, params, root/"kernel_event", "source_baseline")
        if direct["status"] != "ok":
            raise TuningError(direct["error"])
        summary["kernel_event"] = direct
        write_json(root/"comparison.json", summary)
        backend.close()  # Separate runtimes, sequential use of the same device.
        print("[compare 2/3] Fresh OM compilation and target-task profiler", flush=True)
        om = evaluate(backend.om_backend, c, params, root/"om", "source_baseline")
        if om["status"] != "ok":
            raise TuningError(om["error"])
        summary["om"] = om
        write_json(root/"comparison.json", summary)
        print("[compare 3/3] Same direct kernel with task profiler", flush=True)
        directory = root/"kernel_profile"
        directory.mkdir()
        request = directory/"bootstrap.json"
        write_json(request, {"config": c, "build": backend.build, "case": str(backend.case),
                             "ready": str(directory/"ready.json"), "profile_directory": str(directory/"profiler"),
                             "one_shot": {"params": params, "directory": str(directory)}})
        env = environment(backend.build["opp_path"])
        env.update(ASCEND_SLOG_PRINT_TO_STDOUT="1", ASCEND_GLOBAL_LOG_LEVEL="1", PYTHONUNBUFFERED="1")
        record = run_command([sys.executable, "-m", "paretopilot.direct_worker", str(request)], root,
                             directory/"runtime.log", c["measurement"]["timeout_s"], env)
        write_json(directory/"runtime.command.json", record)
        result = read_json(directory/"direct_result.json")
        if result.get("status") != "ok":
            raise TuningError(result.get("error", "Profiled direct kernel failed"))
        validate_result(result, c, params, backend.build)
        if tiling_signature(result) != tiling_signature(direct):
            raise TuningError("Profiled and unprofiled direct tiling differ")
        backend.command([c["ascend"].get("msprof", "msprof"), "--export=on", "--output="+str(directory/"profiler")],
                        directory, "profiler_export", env)
        return complete_comparison(root, summary, c, result)
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if backend is not None:
            backend.close()
        write_json(root/"comparison.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output")
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--reanalyze", help="Reparse an existing case comparison directory without using the device")
    replay.add_argument("--reanalyze-latest", help="Find and reparse the most recent comparison with exported CSVs below this directory")
    args = parser.parse_args()
    if (args.reanalyze or args.reanalyze_latest) and (args.config or args.output):
        parser.error("Reanalysis cannot be combined with --config/--output")
    if not (args.reanalyze or args.reanalyze_latest) and not (args.config and args.output):
        parser.error("Provide --config and --output, or --reanalyze/--reanalyze-latest")
    try:
        if args.reanalyze or args.reanalyze_latest:
            reanalyze(args.reanalyze or latest_capture(args.reanalyze_latest))
        else:
            compare(load_config(args.config), args.output)
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
