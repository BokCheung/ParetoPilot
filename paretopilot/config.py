"""Explicit fixed-workload descriptions; never execute arbitrary baseline scripts."""
from pathlib import Path
import math
import sys
from .artifacts import ConfigError, load_json, number


def load_config(path):
    path = Path(path).resolve()
    c = load_json(path)
    try:
        if not isinstance(c, dict):
            raise ConfigError("Kernel configuration must be an object")
        for section in ("operator", "hardware", "space", "baseline_params", "search", "correctness", "profiling"):
            if not isinstance(c[section], dict):
                raise ConfigError(f"{section} must be an object")
        if c["mode"] not in ("real", "simulation"):
            raise ConfigError("mode must be real or simulation")
        for name in ("baseline_source", "kernel_source"):
            c[name] = str((path.parent / c[name]).resolve())
            if not Path(c[name]).is_file():
                raise ConfigError(f"Missing {name}: {c[name]}")
        for key in ("chip", "cann_version", "source_tag"):
            if not isinstance(c[key], str) or not c[key].strip():
                raise ConfigError(f"{key} is required")
        op = c["operator"]
        if op["kind"] not in ("matmul", "elementwise") or op["dtype"] not in ("float16", "bfloat16", "float32"):
            raise ConfigError("Unsupported operator kind or dtype")
        dims = ("m", "n", "k") if op["kind"] == "matmul" else ("numel", "input_count", "flops_per_element")
        for key in dims:
            number(op[key], key, integer=True, strict=True)
        for key in ("peak_flops", "bandwidth_bytes_s", "local_memory_bytes", "max_blocks"):
            number(c["hardware"][key], key, strict=True)
        number(c["alignment"], "alignment", integer=True, strict=True)
        expected = {"tile_m", "tile_n", "tile_k", "buffers", "block_dim"} if op["kind"] == "matmul" else {"tile_elems", "buffers", "block_dim"}
        if set(c["space"]) != expected or set(c["baseline_params"]) != expected:
            raise ConfigError(f"Parameters must be {sorted(expected)}")
        for name, values in c["space"].items():
            if not isinstance(values, list) or not values:
                raise ConfigError(f"space.{name} must be a nonempty list")
            for value in values + [c["baseline_params"][name]]:
                number(value, name, integer=True, strict=True)
            if len(set(values)) != len(values):
                raise ConfigError(f"Duplicate space.{name} values")
        if math.prod(len(v) for v in c["space"].values()) > 200000:
            raise ConfigError("V1 supports at most 200000 Cartesian combinations")
        search = c["search"]
        for name in ("population", "generations", "max_evaluations", "confirm_rounds", "repeats"):
            number(search[name], name, minimum=2 if name in ("population", "repeats", "confirm_rounds") else 1, integer=True)
        number(search["warmup"], "warmup", integer=True)
        number(search["seed"], "seed", integer=True)
        for key in ("mutation_rate", "min_relative_gain"):
            number(search[key], key)
            if search[key] > 1:
                raise ConfigError(f"{key} must be <= 1")
        for key in ("atol", "rtol"):
            number(c["correctness"][key], key)
        number(c["timeout_s"], "timeout_s", strict=True)
        if not isinstance(c["adapter"], list) or not c["adapter"] or not all(isinstance(x, str) and x for x in c["adapter"]):
            raise ConfigError("adapter must be a nonempty argv array")
        c["adapter"] = [sys.executable if x == "{python}" else str((path.parent / x).resolve()) if (path.parent / x).is_file() else x for x in c["adapter"]]
        for key in ("target_name", "name_column", "duration_column"):
            if not isinstance(c["profiling"][key], str) or not c["profiling"][key]:
                raise ConfigError(f"profiling.{key} is required")
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Invalid kernel config: {exc}") from exc
    return c
