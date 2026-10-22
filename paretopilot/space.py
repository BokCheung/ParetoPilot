"""Template resource constraints and Roofline-guided discrete domains."""
import itertools


def legal(params, config):
    p, c = params, config
    if p["buffers"] not in (1, 2) or p["block_dim"] > c["hardware"]["max_blocks"]:
        return False
    size = 4 if c["operator"]["dtype"] == "float32" else 2
    if c["operator"]["kind"] == "matmul":
        m, n, k = (p[x] for x in ("tile_m", "tile_n", "tile_k"))
        if any(x % c["alignment"] for x in (m, n, k)):
            return False
        memory = p["buffers"] * (m*k + k*n) * size + m*n*4
    else:
        if p["tile_elems"] % c["alignment"]:
            return False
        memory = p["tile_elems"] * p["buffers"] * size * (c["operator"]["input_count"]+1)
    return memory <= c["hardware"]["local_memory_bytes"]


def candidates(config):
    names = sorted(config["space"])
    return [p for values in itertools.product(*(config["space"][n] for n in names))
            if legal(p := dict(zip(names, values)), config)]


def advise(config, best, stalled):
    ranges = {key: list(values) for key, values in config["space"].items()}
    priority = best["roofline"]["parameter_priority"]
    if stalled < 2:
        for name in priority[:2]:
            values = sorted(ranges[name])
            index = values.index(best["params"][name])
            ranges[name] = values[max(0, index-1):index+2]
    return {"ranges": ranges, "priority": priority, "evidence_duration_us": best["duration_us"],
            "reason": "expand original supported domains after stagnation" if stalled >= 2 else "refine priority domains around measured winner; retain elites and global exploration"}
