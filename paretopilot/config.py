import math
from pathlib import Path
from .util import TuningError, number, read_json
from .official import KIND, BATCH_KIND, ADD_KIND, add_enabled, batch_enabled, enabled as official_enabled
from .workload import BASELINE_ONLY, baseline_only


def load_config(path):
    source = Path(path).resolve()
    c = read_json(source)
    try:
        if not isinstance(c, dict):
            raise TuningError("Config must be an object")
        for key in ("operator", "hardware", "space", "search", "measurement", "agent"):
            if not isinstance(c[key], dict):
                raise TuningError(f"{key} must be an object")
        if c["backend"] not in ("simulation", "ascend"):
            raise TuningError("backend must be simulation or ascend")
        op = c["operator"]
        if op["kind"] not in ("matmul", "add") or op["dtype"] not in ("float16", "float32"):
            raise TuningError("V1 supports matmul/add with float16/float32")
        implementation = c.get("implementation", {"kind": "bundled"})
        if not isinstance(implementation, dict) or implementation.get("kind") not in ("bundled", KIND, BATCH_KIND, ADD_KIND, BASELINE_ONLY):
            raise TuningError("implementation.kind must be bundled, cann_matmul_v3, cann_batch_matmul_v3, cann_add_v2 or baseline_only")
        if type(implementation.get("require_same_operator", True)) is not bool:
            raise TuningError("implementation.require_same_operator must be a boolean")
        if type(implementation.get("build_cache", True)) is not bool:
            raise TuningError("implementation.build_cache must be a boolean")
        execution = implementation.get("execution", "om")
        if execution not in ("om", "kernel"):
            raise TuningError("implementation.execution must be om or kernel")
        if execution == "kernel" and (implementation["kind"] not in (BATCH_KIND, ADD_KIND)
                                      or c["backend"] != "ascend" or not implementation.get("build_cache", True)):
            raise TuningError("Kernel execution requires Ascend BatchMatMulV3/AddV2 and build_cache=true")
        if "cache_dir" in implementation:
            if not isinstance(implementation["cache_dir"], str) or not implementation["cache_dir"]:
                raise TuningError("implementation.cache_dir must be a nonempty path")
            implementation["cache_dir"] = str((source.parent/Path(implementation["cache_dir"]).expanduser()).resolve())
        baseline = baseline_only(c)
        if baseline and (c["backend"] != "ascend" or op["kind"] != "matmul"):
            raise TuningError("baseline_only requires Ascend MatMul")
        if "batch" in op or "batch_mode" in op:
            if op["kind"] != "matmul" or "batch" not in op or op.get("batch_mode") not in ("shared_rhs", "batched_rhs"):
                raise TuningError("3D MatMul requires batch and batch_mode=shared_rhs or batched_rhs")
            number(op["batch"], "batch", 1, True)
            if op["batch"] > 2147483647:
                raise TuningError("batch must fit int32")
            if not baseline and not batch_enabled(c):
                raise TuningError("3D MatMul requires baseline_only or cann_batch_matmul_v3")
        official = official_enabled(c)
        if official and (c["backend"] != "ascend" or op["kind"] != ("add" if add_enabled(c) else "matmul") or op["dtype"] != "float16"):
            raise TuningError("Official MatMul currently requires Ascend FP16 MatMul")
        if official:
            number(implementation.get("build_jobs", 4), "build_jobs", 1, True)
            if not batch_enabled(c) and not add_enabled(c) and any(op[k] % 16 for k in ("m", "n", "k")):
                raise TuningError("Official MatMul acceptance currently requires M/N/K multiples of 16")
            if add_enabled(c) and (op["numel"] % 16 or op["numel"] > 2147483647):
                raise TuningError("Unmodified official AddV2 requires FP16 numel divisible by 16 and <= int32; use stock/bundled for unaligned Add")
            if add_enabled(c) and not implementation.get("build_cache", True):
                raise TuningError("Official AddV2 requires build_cache=true for its verified runtime-parameter package")
        for key in (("m", "n", "k") if op["kind"] == "matmul" else ("numel",)):
            number(op[key], key, 1, True)
        for key in ("peak_flops", "bandwidth_bytes_s", "local_memory_bytes", "max_blocks", "alignment"):
            number(c["hardware"][key], key, 1, key not in ("peak_flops", "bandwidth_bytes_s"))
        for key, values in c["space"].items():
            if not key.isidentifier() or not isinstance(values, list) or not values:
                raise TuningError(f"Invalid parameter domain: {key}")
            for value in values:
                number(value, key, 0 if official else 1, True)
            if len(set(values)) != len(values):
                raise TuningError(f"Duplicate values: {key}")
        required = {"tile_m", "tile_n", "tile_k", "buffers", "block_dim"} if op["kind"] == "matmul" else {"tile_elems", "buffers", "block_dim"}
        if official:
            required = {"core_limit", "tile_elems"} if add_enabled(c) else {"core_limit", "tiling_policy"}
            if batch_enabled(c):
                required.add("batch_strategy")
                if "base_m" in c["space"] or "base_n" in c["space"]:
                    required.update(("base_m", "base_n"))
        if baseline:
            required = set()
        if set(c["space"]) != required:
            raise TuningError(f"Template requires parameters {sorted(required)}")
        if official and not add_enabled(c) and (any(v > min(24, c["hardware"]["max_blocks"]) for v in c["space"]["core_limit"])
                         or any(v not in (0, 1) for v in c["space"]["tiling_policy"])):
            raise TuningError("Official MatMul: core_limit must fit the configured AI Core limit (0=auto); tiling_policy must be 0 or 1")
        if add_enabled(c):
            from .official_add import validate_params as validate_add
            for core in c["space"]["core_limit"]:
                for tile in c["space"]["tile_elems"]:
                    validate_add({"core_limit": core, "tile_elems": tile})
        if batch_enabled(c) and any(v not in (0, 1) for v in c["space"]["batch_strategy"]):
            raise TuningError("batch_strategy must be 0 (upstream) or 1 (common)")
        if batch_enabled(c) and "base_m" in c["space"]:
            if any(v > 256 or v % 16 for key in ("base_m", "base_n") for v in c["space"][key]):
                raise TuningError("base_m/base_n must be 0 or 16-aligned values <=256")
        if math.prod(map(len, c["space"].values())) > 200000:
            raise TuningError("Limit discrete template space to 200000 combinations")
        s = c["search"]
        if s.get("domain_mode", "manual") not in ("manual", "shape"):
            raise TuningError("search.domain_mode must be manual or shape")
        if s.get("domain_mode") == "shape" and (not batch_enabled(c) or "base_m" not in c["space"]):
            raise TuningError("Shape domains require BatchMatMulV3 with base_m/base_n controls")
        for option in ("result_cache", "reuse_results"):
            if type(s.get(option, True)) is not bool:
                raise TuningError(f"search.{option} must be boolean")
        if "result_cache_dir" in s:
            if not isinstance(s["result_cache_dir"], str) or not s["result_cache_dir"]:
                raise TuningError("search.result_cache_dir must be a nonempty path")
            s["result_cache_dir"] = str((source.parent/Path(s["result_cache_dir"]).expanduser()).resolve())
        for key in ("population", "generations", "budget", "seed"):
            number(s[key], key, 2 if key == "population" else 0 if key == "seed" else 1, True)
        for key in ("mutation_rate", "min_gain"):
            number(s[key], key)
            if s[key] > 1:
                raise TuningError(f"{key} must be <= 1")
        m = c["measurement"]
        for key in ("warmup", "repeats", "confirmation_rounds"):
            number(m[key], key, 0 if key == "warmup" else 2, True)
        for key in ("rtol", "atol"):
            number(m[key], key)
        number(m["timeout_s"], "timeout_s", 1)
        for role in (("baseline",) if baseline else ("baseline", "source_baseline", "candidate") if official else ("baseline", "candidate")):
            selector = m["selectors"][role]
            if not isinstance(selector, dict) or not selector or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in selector.items()):
                raise TuningError("Profiling selectors must contain exact nonempty column/value pairs")
        c.setdefault("baseline", {"kind": "builtin"})
        if c["baseline"]["kind"] == "python":
            c["baseline"]["path"] = str((source.parent / c["baseline"]["path"]).resolve())
            if not Path(c["baseline"]["path"]).is_file():
                raise TuningError("Missing baseline Python module")
        elif c["baseline"]["kind"] != "builtin":
            raise TuningError("baseline.kind must be builtin or python")
        if c["agent"]["kind"] not in ("dummy", "python"):
            raise TuningError("agent.kind must be dummy or python")
        if c["agent"]["kind"] == "python":
            c["agent"]["path"] = str((source.parent / c["agent"]["path"]).resolve())
            if not Path(c["agent"]["path"]).is_file():
                raise TuningError("Missing Agent Python module")
        if c["backend"] == "ascend":
            a = c["ascend"]
            for key in ("soc_version", "cann_version", "source_tag"):
                if not isinstance(a[key], str) or not a[key] or "REPLACE" in a[key]:
                    raise TuningError(f"Set ascend.{key} for the target machine")
            number(a["device"], "device", 0, True)
            extra = a.get("atc_extra", [])
            if not isinstance(extra, list) or not all(isinstance(v, str) and v.startswith("--") for v in extra):
                raise TuningError("ascend.atc_extra must be a list of --flag=value strings")
            owned = {"--framework", "--model", "--output", "--soc_version", "--input_shape", "--input_format", "--log"}
            if any(v.split("=", 1)[0] in owned for v in extra):
                raise TuningError("atc_extra cannot override controller-owned ATC flags")
            if c["hardware"]["alignment"] % 16:
                raise TuningError("Ascend templates require alignment to be a multiple of 16 elements")
            if any(value > 2147483647 for values in c["space"].values() for value in values):
                raise TuningError("Ascend template parameters must fit signed int32")
            if op["kind"] == "add" and op["numel"] > 4294967295:
                raise TuningError("Add numel must fit uint32")
            if op["kind"] == "matmul" and any(op[k] > 2147483647 for k in ("m", "n", "k")):
                raise TuningError("MatMul dimensions must fit int32")
            if op["dtype"] != "float16":
                raise TuningError("Bundled 910B kernels currently support float16")
            if op["kind"] == "matmul" and not official and not baseline and (any(op[k] % 16 for k in ("m", "n", "k")) or c["space"]["buffers"] != [1]):
                raise TuningError("Bundled MatMul requires M/N/K multiples of 16 and buffers=[1] (CANN owns internal buffers)")
            if not a["soc_version"].startswith("Ascend910B") or not a["cann_version"].startswith("8.5.0"):
                raise TuningError("Bundled kernels target Ascend910B and CANN 8.5.0")
    except (KeyError, TypeError, AttributeError) as exc:
        raise TuningError(f"Invalid configuration: {exc}") from exc
    return c
