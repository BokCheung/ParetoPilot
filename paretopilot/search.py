"""Genetic kernel search and interleaved OM confirmation."""
import math
from pathlib import Path
import random
import statistics
from .artifacts import ConfigError, fingerprint, sha, write_json
from .backend import Evaluator
from .config import load_config
from .report import write_report
from .space import advise, candidates, legal


def breed(config, ranked, pool, seen, rng, advice):
    size = config["search"]["population"]
    if not ranked:
        unseen = [p for p in pool if fingerprint(p) not in seen]
        rng.shuffle(unseen)
        return unseen[:size]
    children = [t["params"] for t in ranked[:min(2, size-1)]]
    def parent():
        return min(rng.choices(ranked, k=3), key=lambda t: t["duration_us"])["params"]
    for _ in range(size*30):
        if len(children) >= size-1:
            break
        a, b = parent(), parent()
        child = {n: rng.choice([a[n], b[n]]) for n in config["space"]}
        for name in child:
            rate = config["search"]["mutation_rate"] * (2 if name in advice["priority"][:2] else 1)
            if rng.random() < min(1, rate):
                child[name] = rng.choice(advice["ranges"][name])
        if legal(child, config) and fingerprint(child) not in seen and child not in children:
            children.append(child)
        if len(children) >= size-1:
            break
    unseen = [p for p in pool if fingerprint(p) not in seen and p not in children]
    rng.shuffle(unseen)
    children.extend(unseen[:max(0, size-len(children))])
    return children


def confirm(evaluator, config, winner):
    trials = []
    if winner:
        for index in range(config["search"]["confirm_rounds"]):
            for role in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")):
                params = config["baseline_params"] if role == "baseline" else winner["params"]
                trials.append(evaluator.evaluate(params, f"confirmation/{index:03d}-{role}", role))
    if not trials or any(t["status"] != "ok" for t in trials):
        return trials, False, None, None
    baseline = [t["duration_us"] for t in trials if t["role"] == "baseline"]
    candidate = [t["duration_us"] for t in trials if t["role"] == "candidate"]
    gain = 1-statistics.median(candidate)/statistics.median(baseline)
    noise = max((max(v)-min(v))/statistics.median(v) for v in (baseline, candidate))
    return trials, gain > max(noise, config["search"]["min_relative_gain"]), gain, noise


def tune_kernel(config_path, output):
    config = load_config(config_path)
    root = Path(output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ConfigError("Output must be a new or empty directory")
    pool = candidates(config)
    if not pool or not legal(config["baseline_params"], config):
        raise ConfigError("No legal candidates or invalid baseline template parameters")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "config": config, "baseline_sha256": sha(config["baseline_source"]),
                "kernel_sha256": sha(config["kernel_source"]), "adapter_files": {x: sha(x) for x in config["adapter"] if Path(x).is_file()}, "legal_candidates": len(pool)}
    write_json(root / "manifest.json", manifest)
    for key in ("baseline_source", "kernel_source"):
        (root / f"{key}.snapshot").write_bytes(Path(config[key]).read_bytes())
    evaluator = Evaluator(config, root)
    baseline = evaluator.evaluate(config["baseline_params"], "baseline", "baseline")
    if baseline["status"] != "ok":
        raise ConfigError(f"Baseline failed; inspect baseline/trial.json: {baseline.get('error')}")
    rng = random.Random(config["search"]["seed"])
    population = rng.sample(pool, min(len(pool), config["search"]["population"]))
    if config["baseline_params"] in pool:
        population[0] = config["baseline_params"].copy()
    seen, history, domains = set(), [], []
    previous, stalled = math.inf, 0
    for generation in range(config["search"]["generations"]):
        for params in population:
            key = fingerprint(params)
            if key in seen or len(seen) >= config["search"]["max_evaluations"]:
                continue
            trial = evaluator.evaluate(params, f"trials/{len(seen):05d}")
            trial["generation"] = generation
            seen.add(key)
            history.append(trial)
            write_json(root / "history.json", history)
        ranked = sorted((t for t in history if t["status"] == "ok"), key=lambda t: t["duration_us"])
        if len(seen) >= min(len(pool), config["search"]["max_evaluations"]):
            break
        advice = {"ranges": config["space"], "priority": baseline["roofline"]["parameter_priority"], "reason": "global exploration after failed candidates"}
        if ranked:
            stalled = stalled+1 if ranked[0]["duration_us"] >= previous else 0
            previous = ranked[0]["duration_us"]
            advice = advise(config, ranked[0], stalled)
        domains.append({"generation": generation+1, **advice})
        population = breed(config, ranked, pool, seen, rng, advice)
    ranked = sorted((t for t in history if t["status"] == "ok"), key=lambda t: t["duration_us"])
    winner = ranked[0] if ranked else None
    confirmation, accepted, gain, noise = confirm(evaluator, config, winner)
    result = {"mode": config["mode"], "accepted": accepted, "baseline": baseline, "best_candidate": winner,
              "selected_params": winner["params"] if accepted else config["baseline_params"], "evaluations": len(history),
              "confirmation": confirmation, "confirmation_relative_gain": gain, "observed_relative_range": noise,
              "space_history": domains, "report": str(root / "report.md")}
    write_json(root / "summary.json", result)
    write_json(root / "best_params.json", result["selected_params"])
    write_report(root, result)
    return {k: result[k] for k in ("mode", "accepted", "selected_params", "evaluations", "confirmation_relative_gain", "report")}
