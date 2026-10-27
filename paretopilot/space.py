import itertools
from .workload import baseline_only
from .official import enabled as official_enabled, batch_enabled, validate_params
from .util import TuningError


def legal(p, config):
    if baseline_only(config):
        return False
    hw, op = config["hardware"], config["operator"]
    if official_enabled(config):
        try:
            validate_params(p, batch=batch_enabled(config))
        except TuningError:
            return False
        if p["core_limit"] > min(24, hw["max_blocks"]):
            return False
        if p.get("base_m", 0):
            # Conservative shape filter. Real buffer limits are checked by the
            # CANN API/Host using the device platform, not Roofline estimates.
            return p["base_m"] <= (op["m"]+15)//16*16 and p["base_n"] <= (op["n"]+15)//16*16
        return True
    if p["buffers"] not in (1, 2) or p["block_dim"] > hw["max_blocks"]:
        return False
    size = 2 if op["dtype"] == "float16" else 4
    if op["kind"] == "matmul":
        m, n, k = p["tile_m"], p["tile_n"], p["tile_k"]
        if any(v % hw["alignment"] for v in (m, n, k)):
            return False
        memory = p["buffers"]*(m*k+k*n)*size + m*n*4
    else:
        if p["tile_elems"] % hw["alignment"]:
            return False
        memory = p["tile_elems"]*size*3*p["buffers"]
    return memory <= hw["local_memory_bytes"]


def enumerate_space(config, space=None):
    space = config["space"] if space is None else space
    names = sorted(space)
    return [p for vals in itertools.product(*(space[n] for n in names)) if legal(p := dict(zip(names, vals)), config)]
