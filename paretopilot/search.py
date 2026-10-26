"""Deterministic genetic loop, durable experiments, and independent confirmation."""
from copy import deepcopy
from pathlib import Path
import random
import statistics
from .agent import load_agent, validate_proposal
from .backend import AscendBackend, SimulationBackend, evaluate
from .doctor import inspect_environment
from .report import write_report, write_baseline_report
from .workload import baseline_only as baseline_config
from .space import enumerate_space, legal
from .official import enabled as official_enabled, DEFAULT_PARAMS
from .util import TuningError, digest, file_hash, tree_hash, write_json


def next_population(config, ranked, pool, seen, proposal, rng):
    size = config["search"]["population"]
    population = [t["params"] for t in ranked[:min(2, size-1)]]
    if ranked:
        def parent():
            return min(rng.choices(ranked, k=3), key=lambda t: t["duration_us"])["params"]
        for _ in range(size*40):
            if len(population) >= size-1:
                break
            a, b = parent(), parent()
            child = {k: rng.choice([a[k], b[k]]) for k in config["space"]}
            for k in child:
                if child[k] not in proposal["space"][k] or rng.random() < min(1, config["search"]["mutation_rate"]*(2 if k in proposal.get("priority", []) else 1)):
                    child[k] = rng.choice(proposal["space"][k])
            if legal(child, config) and digest(child) not in seen and child not in population:
                population.append(child)
    unseen = [p for p in pool if digest(p) not in seen and p not in population]
    rng.shuffle(unseen)
    population.extend(unseen[:max(0, size-len(population))])
    return population


def confirmation_roles(config, round_index):
    roles = ("baseline", "source_baseline", "candidate") if official_enabled(config) else ("baseline", "candidate")
    return roles if round_index % 2 == 0 else tuple(reversed(roles))


def confirmation(config, winner, backend, root):
    trials = []
    if winner:
        for i in range(config["measurement"]["confirmation_rounds"]):
            for role in confirmation_roles(config, i):
                params = {} if role == "baseline" else DEFAULT_PARAMS.copy() if role == "source_baseline" else winner["params"]
                trial = evaluate(backend, config, params, root/f"confirmation/{i:03d}-{role}", role)
                trials.append(trial)
                write_json(root/"confirmation.json", trials)
    if not trials or any(t["status"] != "ok" for t in trials):
        return {"accepted": False, "reason": "no valid candidate or confirmation failed", "trials": trials}
    b = [t["duration_us"] for t in trials if t["role"] == "baseline"]
    c = [t["duration_us"] for t in trials if t["role"] == "candidate"]
    bm, cm = statistics.median(b), statistics.median(c)
    gain = 1-cm/bm
    spread = max((max(v)-min(v))/statistics.median(v) for v in (b, c))
    accepted = gain > max(spread, config["search"]["min_gain"])
    result = {"accepted": accepted, "reason": "gain exceeds minimum threshold and observed range" if accepted else "gain insufficient versus threshold/noise",
            "baseline_us": bm, "candidate_us": cm, "speedup": bm/cm, "relative_gain": gain, "relative_range": spread, "trials": trials}
    if official_enabled(config):
        s = [t["duration_us"] for t in trials if t["role"] == "source_baseline"]
        sm = statistics.median(s)
        source_range = max((max(v)-min(v))/statistics.median(v) for v in (s, c))
        source_gain = 1-cm/sm
        # Rebuilding the exact default parameters is a smoke check, never tuning evidence.
        tuned = winner["params"] != DEFAULT_PARAMS and source_gain > max(source_range, config["search"]["min_gain"])
        stock_source_range = max((max(v)-min(v))/statistics.median(v) for v in (b, s))
        source_beats_stock = 1-sm/bm > max(stock_source_range, config["search"]["min_gain"])
        selected = "candidate" if accepted and tuned else "source_baseline" if source_beats_stock else "baseline"
        result.update(accepted=selected != "baseline", selected=selected, tuning_accepted=tuned,
                      source_baseline_us=sm, source_speedup=sm/cm, source_relative_gain=source_gain,
                      source_relative_range=source_range, source_vs_stock_speedup=bm/sm,
                      reason=("tuned official source beats both references beyond threshold/noise" if selected == "candidate" else
                              "official source defaults beat stock; tuning gain not established" if selected == "source_baseline" else
                              "no official source configuration establishes a gain over stock"))
    return result


def run(config, output, progress=None, baseline_only=False):
    if baseline_config(config) and not baseline_only:
        raise TuningError("baseline_only workload: use the baseline command; 3D candidate tuning is not supported")
    root = Path(output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise TuningError("Output directory must be empty; choose a new run directory")
    original_pool = enumerate_space(config)
    if not original_pool and not baseline_config(config):
        raise TuningError("No legal template configurations")
    root.mkdir(parents=True, exist_ok=True)
    sources = Path(__file__).parent
    manifest = {"format_version": 1, "config": config, "controller_and_kernels_sha256": tree_hash(sources), "legal_candidates": len(original_pool)}
    if config["baseline"]["kind"] == "python":
        path = Path(config["baseline"]["path"])
        manifest["baseline_source_sha256"] = file_hash(path)
        (root/"baseline_source.py").write_bytes(path.read_bytes())
    if config["agent"]["kind"] == "python":
        manifest["agent_source_sha256"] = file_hash(config["agent"]["path"])
    write_json(root/"manifest.json", manifest)
    if config["backend"] == "ascend":
        environment = inspect_environment(config)
        write_json(root/"environment.json", environment)
        if not environment["ready"]:
            raise TuningError(f"Environment checks failed; see {root/'environment.json'} and run doctor")
    backend = AscendBackend(config, root) if config["backend"] == "ascend" else SimulationBackend(config, root)
    baseline = evaluate(backend, config, {}, root/"baseline", "baseline")
    if baseline["status"] != "ok":
        raise TuningError(f"Baseline failed: {baseline['error']}")
    if baseline_only:
        write_json(root/"baseline_summary.json", baseline)
        write_baseline_report(root, config, baseline)
        return {"mode": config["backend"], "baseline_us": baseline["duration_us"], "report": str(root/"report.html")}
    source_baseline = None
    if official_enabled(config):
        source_baseline = evaluate(backend, config, DEFAULT_PARAMS.copy(), root/"source_baseline", "source_baseline")
        if source_baseline["status"] != "ok":
            raise TuningError(f"Official source baseline failed: {source_baseline['error']}")
    agent = load_agent(config["agent"])
    rng = random.Random(config["search"]["seed"])
    history, decisions, seen, population = [], [], set(), []
    for generation in range(config["search"]["generations"]):
        ranked = sorted((t for t in history if t["status"] == "ok"), key=lambda t: t["duration_us"])
        context = {"schema_version": 1, "generation": generation, "operator": config["operator"], "hardware": config["hardware"],
                   "original_space": config["space"], "baseline": source_baseline or baseline, "best": ranked[0] if ranked else None,
                   "roofline": ranked[0]["roofline"] if ranked else (source_baseline or baseline)["roofline"], "history": history}
        if source_baseline:
            context["stock_baseline"] = baseline
        # A local plugin cannot mutate controller state through its input.
        proposal = validate_proposal(agent.propose(deepcopy(context)), config["space"])
        pool = enumerate_space(config, proposal["space"])
        if not pool:
            raise TuningError("Agent proposal contains no legal candidates")
        write_json(root/f"agent/{generation:03d}.input.json", context)
        write_json(root/f"agent/{generation:03d}.output.json", proposal)
        decisions.append(proposal)
        population = next_population(config, ranked, pool, seen, proposal, rng)
        fresh = [p for p in population if digest(p) not in seen]
        if not fresh:
            break
        for p in fresh:
            if len(seen) >= config["search"]["budget"]:
                break
            trial = evaluate(backend, config, p, root/f"trials/{len(seen):05d}", "candidate")
            trial["generation"] = generation
            seen.add(digest(p))
            history.append(trial)
            write_json(root/"history.json", history)
            if progress:
                progress(trial, len(seen))
        if len(seen) >= config["search"]["budget"]:
            break
    ranked = sorted((t for t in history if t["status"] == "ok"), key=lambda t: t["duration_us"])
    winner = ranked[0] if ranked else None
    final = confirmation(config, winner, backend, root)
    selected = final.get("selected", "candidate" if final["accepted"] else "baseline")
    result = {"mode": config["backend"], "agent": config["agent"], "baseline": baseline, "winner": winner,
              "confirmation": final, "selected": selected,
              "selected_params": winner["params"] if selected == "candidate" else DEFAULT_PARAMS.copy() if selected == "source_baseline" else None,
              "history": history, "agent_decisions": decisions}
    if source_baseline:
        result["source_baseline"] = source_baseline
    write_json(root/"summary.json", result)
    write_json(root/"best_params.json", {"selected": result["selected"], "params": result["selected_params"]})
    write_report(root, result)
    return {"mode": result["mode"], "selected": result["selected"], "evaluations": len(history), "successful_candidates": len(ranked), "accepted": final["accepted"], "report": str(root/"report.html")}
