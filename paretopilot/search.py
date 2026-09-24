"""Bounded, reproducible discrete search with constraint-aware Pareto selection."""

from __future__ import annotations

import copy
import math
import random

from .agent import ExperimentAgent
from .config import ConfigError, fingerprint, normalize_inputs, set_path, validate_model, validate_search
from .roofline import evaluate
from .store import CacheError


def pareto_front(results):
    feasible = [r for r in results if r["status"] == "ok" and not r.get("search_violations")]
    front = []
    for result in feasible:
        a = result["metrics"]
        dominated = False
        for other in feasible:
            b = other["metrics"]
            if (b["latency_ms"] <= a["latency_ms"] and b["peak_memory_bytes"] <= a["peak_memory_bytes"]
                    and (b["latency_ms"] < a["latency_ms"] or b["peak_memory_bytes"] < a["peak_memory_bytes"])):
                dominated = True
                break
        if not dominated:
            front.append(result)
    return sorted(front, key=lambda r: (r["metrics"]["latency_ms"], r["metrics"]["peak_memory_bytes"], r["candidate_id"]))


def diff_config(baseline, candidate, prefix=""):
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        changes = {}
        for key in sorted(baseline.keys() | candidate.keys()):
            path = prefix + "." + key if prefix else key
            changes.update(diff_config(baseline.get(key), candidate.get(key), path))
        return changes
    if isinstance(baseline, list) and isinstance(candidate, list) and len(baseline) == len(candidate):
        changes = {}
        for i, (a, b) in enumerate(zip(baseline, candidate)):
            changes.update(diff_config(a, b, f"{prefix}.{i}"))
        return changes
    return {} if baseline == candidate else {prefix: {"before": baseline, "after": candidate}}


def constraint_violations(result, baseline_params, constraints):
    metrics = result["metrics"]
    ratio = metrics["parameter_count"] / baseline_params
    lo, hi = constraints["parameter_ratio"]
    violations = []
    if not lo <= ratio <= hi:
        violations.append(f"Parameter ratio {ratio:.6g} outside [{lo}, {hi}]")
    if metrics["latency_ms"] > constraints.get("max_latency_ms", math.inf):
        violations.append("Latency exceeds search limit")
    if metrics["peak_memory_bytes"] > constraints.get("max_memory_bytes", math.inf):
        violations.append("Memory exceeds search limit")
    return violations


def run_search(model, npu, workload, search, store=None, progress=None):
    model, npu, workload = normalize_inputs(model, npu, workload)
    spec = validate_search(search, model)
    agent, rng = ExperimentAgent(), random.Random(spec["seed"])
    parameters = dict(sorted(spec["parameters"].items()))
    paths = list(parameters)
    radices = [len(parameters[p]) for p in paths]
    space_size = math.prod(radices)
    results, trials, visited = [], [], set()
    seen_models = set()
    cursor = 0

    def evaluate_cached(candidate):
        return store.evaluate(candidate, npu, workload, evaluate) if store else evaluate(candidate, npu, workload)

    baseline = evaluate_cached(model)
    base_params = baseline["metrics"]["parameter_count"]

    def record(candidate, decision, result=None):
        candidate_id = fingerprint(candidate)
        try:
            normalized = validate_model(candidate)
            result = copy.deepcopy(result if result is not None else evaluate_cached(normalized))
            result["search_violations"] = constraint_violations(result, base_params, spec["constraints"])
            results.append(result)
            trial = {"candidate_id": result["candidate_id"], "evaluation_id": result["evaluation_id"],
                     "status": result["status"] if not result["search_violations"] else "constraint_rejected",
                     "metrics": result["metrics"], "violations": result["violations"] + result["search_violations"]}
        except CacheError:
            raise
        except ConfigError as exc:
            trial = {"candidate_id": candidate_id, "status": "invalid", "violations": [str(exc)]}
        trial.update({"trial": len(trials), "decision": decision,
                      "config_diff": diff_config(model, candidate)})
        trials.append(trial)
        seen_models.add(candidate_id)
        if store:
            store.checkpoint(trials)
        if progress:
            progress(trial)

    record(model, {"kind": "baseline"}, baseline)

    def decode(index):
        values = [0] * len(radices)
        for i in range(len(radices) - 1, -1, -1):
            index, values[i] = divmod(index, radices[i])
        return values

    def encode(indices):
        index = 0
        for size, value in zip(radices, indices):
            index = index * size + value
        return index

    while len(trials) < spec["max_evaluations"] and len(visited) < space_size:
        index, decision = None, {"kind": spec["method"]}
        if spec["method"] != "grid":
            for _ in range(64):
                if spec["method"] == "adaptive" and results and rng.random() >= spec["exploration_probability"]:
                    parents = pareto_front(results) or results
                    proposal = agent.mutate(rng.choice(parents), parameters, rng)
                    if proposal:
                        indices, decision = proposal
                        index = encode(indices)
                    else:
                        index, decision = rng.randrange(space_size), {"kind": "random_exploration"}
                else:
                    index, decision = rng.randrange(space_size), {"kind": "random_exploration"}
                if index not in visited:
                    break
                index = None
        if index is None:
            while cursor in visited:
                cursor += 1
            index = cursor
            decision = {"kind": "grid" if spec["method"] == "grid" else "unvisited_fallback"}
        visited.add(index)
        candidate = copy.deepcopy(model)
        for path, choice in zip(paths, decode(index)):
            set_path(candidate, path, parameters[path][choice])
        if fingerprint(candidate) in seen_models:
            continue
        record(candidate, decision)

    front = pareto_front(results)
    return {
        "search_spec": spec, "baseline": baseline, "trials": trials,
        "pareto": front, "diagnosis": agent.diagnose(baseline, parameters),
        "summary": {"trials": len(trials), "valid_evaluations": len(results),
                    "feasible": sum(r["status"] == "ok" and not r["search_violations"] for r in results),
                    "pareto_count": len(front), "discrete_space_size": space_size,
                    "stop_reason": "space_exhausted" if len(visited) == space_size else "budget_reached",
                    "cache_hits": store.cache_hits if store else 0},
        "quality_status": "unverified",
    }
