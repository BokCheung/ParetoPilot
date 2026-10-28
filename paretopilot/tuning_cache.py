"""Historical parameters are hints; measurements are always performed again."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from .official import COMMIT
from .space import legal
from .util import TuningError, digest, file_hash, read_json
from .workload import shapes


def identity(config, source):
    build = source.get("selection") or {}
    runtime = config["search"].get("runtime_signature", {})
    if not build.get("installed_sha256") or not build.get("cache_identity") or not runtime.get("driver_version_lines"):
        return None  # No persistent reuse without verified build/environment evidence.
    root = Path(__file__).parent
    files = [*sorted(root.glob("*.py")), *sorted((root/"official_adapter").glob("*"))]
    family = {"format": 2, "ranking_metric": source.get("timing_method"), "commit": COMMIT, "installed_sha256": build["installed_sha256"],
              "build_environment": build["cache_identity"],
              "controller": {p.relative_to(root).as_posix(): file_hash(p) for p in files if p.is_file()},
              "soc": config["ascend"]["soc_version"], "cann": config["ascend"]["cann_version"],
              "device": config["ascend"]["device"], "input_seed": config["search"]["seed"],
              "runtime": runtime,
              "tensor_contract": {"layout": "ND", "transpose_a": False, "transpose_b": False, "bias": False},
              "platform": config["search"]["domain_limits"], "execution": config["implementation"].get("execution", "om"),
              "measurement": config["measurement"], "min_gain": config["search"]["min_gain"],
              "operator": {k: v for k, v in config["operator"].items() if k not in ("m", "n", "k", "batch")},
              "agent": config["agent"]}
    if config["agent"]["kind"] == "python":
        family["agent_sha256"] = file_hash(config["agent"]["path"])
    return {"family": family, "workload": {"operator": config["operator"], "shapes": list(shapes(config["operator"]))}}


def cache_root(config):
    return Path(config["search"].get("result_cache_dir", Path(__file__).resolve().parents[1]/".cache/tuning-results"))


def valid_record(record):
    try:
        return (record["key"] == digest(record["identity"]) and type(record["confirmed"]) is bool
                and record.get("timing_method") == record["identity"]["family"].get("ranking_metric")
                and isinstance(record["params"], dict)
                and all(type(record[k]) in (float, int) and math.isfinite(record[k]) and record[k] > 0
                        for k in ("duration_us", "source_us")))
    except (KeyError, TypeError, ValueError):
        return False


def distance(a, b):
    # Same operator family/ranks/broadcast mode is checked before this metric.
    av, bv = a["operator"], b["operator"]
    ratios = [abs(math.log2(av.get(k, 1)/bv.get(k, 1))) for k in ("m", "n", "k", "batch")]
    return sum(ratios) if max(ratios) <= 1 else math.inf


def lookup(config, source):
    current = identity(config, source)
    evidence = {"status": "miss", "identity": current, "key": digest(current) if current else None,
                "seed_attempted": False, "search_skipped": False}
    if not config["search"].get("result_cache", True) or not config["search"].get("reuse_results", True):
        evidence["status"] = "disabled"
        return None, evidence
    if current is None:
        evidence["status"] = "unavailable"
        return None, evidence
    pool = []
    # No global index to corrupt and no timing result is executed or trusted.
    for path in sorted(cache_root(config).glob("*.json"), key=lambda p: p.name):
        try:
            record = read_json(path)
            if not valid_record(record) or record["identity"]["family"] != current["family"]:
                continue
            params = record["params"]
            if not legal(params, config) or set(params) != set(config["space"]) or any(params[k] not in v for k, v in config["space"].items()):
                continue
            d = distance(current["workload"], record["identity"]["workload"])
            if math.isfinite(d):
                pool.append((d, record["duration_us"]/record["source_us"], record["key"], record))
        except (TuningError, OSError, TypeError, KeyError, ValueError, ZeroDivisionError):
            continue
    if not pool:
        return None, evidence
    record = min(pool, key=lambda item: item[:3])[-1]
    exact = record["identity"] == current
    evidence.update(status="exact" if exact else "nearby", cached_key=record["key"],
                    params=record["params"], previously_confirmed=record["confirmed"],
                    action="remeasure before reuse" if exact else "remeasure as search seed only")
    return record, evidence


def stable_retest(config, source, trial, record, match):
    if match.get("status") != "exact" or not record or not record["confirmed"] or trial.get("status") != "ok":
        return False
    if not (source.get("timing_method") == trial.get("timing_method") == record.get("timing_method")):
        return False
    if any(t.get("correctness", {}).get("passed") is not True or t.get("correctness", {}).get("simulated", False)
           for t in (source, trial)):
        return False
    b, c = source["duration_us"], trial["duration_us"]
    if not all(math.isfinite(v) and v > 0 for v in (b, c)):
        return False
    spread = max((max(t["samples_us"])-min(t["samples_us"]))/statistics.median(t["samples_us"])
                 for t in (source, trial))
    gain = 1-c/b
    old_speedup = record["source_us"]/record["duration_us"]
    return gain > max(config["search"]["min_gain"], spread) and b/c >= old_speedup*0.9


def save(config, source, winner, confirmation, run_directory):
    current = identity(config, source)
    if not config["search"].get("result_cache", True) or current is None or not winner or winner.get("status") != "ok":
        return None
    if source.get("timing_method") != winner.get("timing_method"):
        return None
    record = {"key": digest(current), "identity": current, "params": winner["params"],
              "duration_us": winner["duration_us"], "source_us": source["duration_us"],
              "timing_method": source.get("timing_method"),
              "confirmed": confirmation.get("tuning_accepted") is True,
              "run": str(run_directory), "saved_utc": datetime.now(timezone.utc).isoformat()}
    if not valid_record(record):
        return None
    root = cache_root(config)
    root.mkdir(parents=True, exist_ok=True)
    # Unique temporary name prevents concurrent runs from sharing a .tmp file.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=root, suffix=".tmp", delete=False) as tmp:
        json.dump(record, tmp, ensure_ascii=False, allow_nan=False)
        temporary = Path(tmp.name)
    try:
        os.replace(temporary, root/(record["key"]+".json"))
    finally:
        temporary.unlink(missing_ok=True)
    return record["key"]
