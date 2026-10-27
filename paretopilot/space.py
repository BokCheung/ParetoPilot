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
            if op["k"] > 65535 or op["n"] > 65535:
                return False  # Current fixed-tile Host branch requires on-the-fly K/N.
            # Conservative shape filter. Real buffer limits are checked by the
            # CANN API/Host using the device platform, not Roofline estimates.
            m, n = p["base_m"], p["base_n"]
            if m > (op["m"]+15)//16*16 or n > (op["n"]+15)//16*16:
                return False
            limits = config["search"].get("domain_limits")
            if limits:
                # Necessary bounds using the smallest FP16 K block (16).
                # These do not substitute for GetTiling's full validation.
                return (m*n*4 <= limits["L0C_SIZE"] and m*16*2 <= limits["L0A_SIZE"]
                        and n*16*2 <= limits["L0B_SIZE"] and 2*16*(m+n)*2 <= limits["L1_SIZE"])
            return True
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
