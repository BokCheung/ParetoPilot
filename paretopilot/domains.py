"""Shape-based candidate domains using verified CANN platform limits."""
from copy import deepcopy
import ctypes as C
from pathlib import Path
import sys

from .space import enumerate_space
from .util import TuningError, read_json, run_command, write_json

MEMORY_KEYS = ("L0A_SIZE", "L0B_SIZE", "L0C_SIZE", "L1_SIZE")


def enabled(config):
    return config["search"].get("domain_mode", "manual") == "shape"


def validate_platform(platform, config):
    if (not isinstance(platform, dict) or platform.get("status") != "ok"
            or platform.get("runtime_soc") != config["ascend"]["soc_version"]
            or platform.get("compile_soc") != platform.get("runtime_soc")
            or type(platform.get("core_count")) is not int or not 1 <= platform["core_count"] <= 24
            or any(type(platform.get(k)) is not int or platform[k] <= 0 for k in MEMORY_KEYS)):
        raise TuningError("Shape domains require verified CANN SoC/core/L0/L1 platform evidence")
    return {k: platform[k] for k in ("runtime_soc", "compile_soc", "core_count", *MEMORY_KEYS)}


def tile_values(dimension):
    aligned = (dimension+15)//16*16
    values = {16, 32, 64, 128, 256}
    if aligned <= 128:
        values.update((48, 80, 96, 112, aligned))
    return [0, *sorted(v for v in values if v <= min(aligned, 256))]


def generate(config, platform):
    limits = validate_platform(platform, config)
    c = deepcopy(config)
    op = c["operator"]
    cores = limits["core_count"]
    task_bound = ((op["m"]+15)//16)*((op["n"]+15)//16)*op.get("batch", 1)
    c["space"] = {"base_m": tile_values(op["m"]), "base_n": tile_values(op["n"]),
                  "core_limit": [0, *sorted({cores, *(v for v in (1, 2, 4, 6, 8, 12, 16, 20, 24) if v <= min(cores, task_bound))})],
                  "tiling_policy": [0, 1], "batch_strategy": [0, 1]}
    c["hardware"]["max_blocks"] = cores
    c["search"]["domain_limits"] = limits
    pool = enumerate_space(c)
    if not pool:
        raise TuningError("Shape domain generator produced no legal candidates")
    return c, {"mode": "shape", "operator": op, "platform": limits, "space": c["space"],
               "legal_candidates": len(pool), "budget": c["search"]["budget"],
               "task_upper_bound": task_bound,
               "coverage": "budget_covers_domain" if len(pool) <= c["search"]["budget"] else "budgeted",
               "rules": ["Paired explicit tiles; heuristic/common only; baseK automatic",
                         "Aligned shape and necessary L0/L1 bounds; official Host remains final validator",
                         "Automatic policy retains L1/multibatch eligibility; fixed-tile specializations remain disabled"]}


def prepare(config, source, root, manifest):
    if not enabled(config):
        return config
    platform = (source.get("selection") or {}).get("platform_info")
    if not platform:
        from .backend import environment
        directory = root/"domain_probe"
        request = directory/"request.json"
        write_json(request, {"config": config, "directory": str(directory)})
        run_command([sys.executable, "-m", "paretopilot.domains", str(request)], root,
                    directory/"probe.log", config["measurement"]["timeout_s"], environment())
        platform = read_json(directory/"platform_init.json")
    c, evidence = generate(config, platform)
    write_json(root/"domain.json", evidence)
    write_json(root/"effective_config.json", c)
    manifest.update(config=c, requested_config=config, legal_candidates=evidence["legal_candidates"])
    write_json(root/"manifest.json", manifest)
    print(f"Shape domains: {evidence['legal_candidates']} legal candidates; budget {evidence['budget']}", flush=True)
    return c


def main():
    from .direct_worker import bind, initialize_compile_platform
    request = read_json(sys.argv[1])
    lib = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
    def call(name, types, *args):
        code = bind(lib, name, types)(*args)
        if code:
            raise TuningError(f"{name} failed with ACL code {code}")
    call("aclInit", [C.c_char_p], None)
    device = request["config"]["ascend"]["device"]
    try:
        call("aclrtSetDevice", [C.c_int], device)
        try:
            initialize_compile_platform(lib, request["config"], Path(request["directory"]))
        finally:
            call("aclrtResetDevice", [C.c_int], device)
    finally:
        call("aclFinalize", [])


if __name__ == "__main__":
    main()
