"""An inspectable rule-based experiment agent; no hosted LLM or credentials needed."""

from .config import get_path


AGENT_VERSION = "bottleneck-policy-v2"


def affected_prefix(path):
    if path == "dense_mlp":
        return "dense."
    if path.startswith("embeddings."):
        return "embedding." + path.split(".")[1] + "."
    if path.startswith("sequence_encoder."):
        return "sequence."
    if path.startswith("towers."):
        return "tower." + path.split(".")[1] + "."
    return "interaction."


class ExperimentAgent:
    """Prioritize structure mutations using measured *analytical* operator shares."""

    def diagnose(self, result, paths=()):
        ops = sorted(result["operators"], key=lambda op: op["latency_ms"], reverse=True)
        total = result["metrics"]["latency_ms"] or 1.0
        priorities = []
        for path in paths:
            prefix = affected_prefix(path)
            share = sum(op["latency_ms"] for op in ops
                        if op["name"] == prefix.rstrip(".") or op["name"].startswith(prefix)) / total
            priorities.append({"path": path, "latency_share": share, "sampling_weight": 1 + 20 * share})
        priorities.sort(key=lambda p: p["sampling_weight"], reverse=True)
        top = ops[:5]
        return {
            "policy": AGENT_VERSION,
            "agent_type": "deterministic_rule_based",
            "dominant_bottleneck": max(result["bottleneck_time_ms"], key=result["bottleneck_time_ms"].get),
            "top_operators": [{"name": op["name"], "bottleneck": op["bottleneck"],
                               "latency_ms": op["latency_ms"], "share": op["latency_ms"] / total}
                              for op in top],
            "search_priorities": priorities,
            "rationale": "优先探索耗时占比较高模块的结构参数；保留随机探索，并由约束检查控制模型规模。",
            "quality_status": "unverified",
        }

    def mutate(self, parent, parameters, rng):
        diagnosis = self.diagnose(parent, parameters)
        mutable = [p for p in diagnosis["search_priorities"] if len(parameters[p["path"]]) > 1]
        if not mutable:
            return None
        chosen = rng.choices(mutable, weights=[p["sampling_weight"] for p in mutable], k=1)[0]
        path = chosen["path"]
        indices = []
        for key, choices in parameters.items():
            current = get_path(parent["model"], key)
            try:
                index = choices.index(current)
            except ValueError:
                index = rng.randrange(len(choices))
            if key == path:
                alternatives = [i for i in range(len(choices)) if i != index]
                index = rng.choice(alternatives)
            indices.append(index)
        return indices, {
            "kind": "bottleneck_guided_mutation", "parent_id": parent["candidate_id"],
            "path": path, "parent_module_latency_share": chosen["latency_share"],
            "rationale": "按父候选逐算子耗时占比加权选择变异维度；不预设缩小结构一定更快。",
        }
