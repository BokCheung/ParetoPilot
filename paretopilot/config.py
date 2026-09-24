"""Strict, dependency-free validation of the public JSON configuration format."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
OPS = {"embedding", "matmul", "relu", "concat", "gather", "softmax",
       "add", "layernorm", "mean", "sigmoid"}


class ConfigError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _bad_constant(value):
    raise ConfigError(f"JSON contains non-finite number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                          parse_constant=_bad_constant, object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def obj(value, allowed, required, path):
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected an object")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown:
        raise ConfigError(f"{path}: unknown fields {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{path}: missing fields {sorted(missing)}")


def integer(value, path, minimum=1):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{path}: expected integer >= {minimum}")


def number(value, path, minimum=0, maximum=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{path}: expected finite number")
    if value < minimum or (positive and value == minimum) or (maximum is not None and value > maximum):
        raise ConfigError(f"{path}: number outside allowed range")


def name(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected nonempty string")


def sizes(value, path):
    if not isinstance(value, list):
        raise ConfigError(f"{path}: expected list of layer widths")
    for i, width in enumerate(value):
        integer(width, f"{path}[{i}]")


def validate_model(value):
    m = copy.deepcopy(value)
    fields = {"schema_version", "name", "dtype", "dense_features", "embeddings", "dense_mlp",
              "interaction", "sequence_encoder", "towers"}
    obj(m, fields, {"name", "dtype", "dense_features", "embeddings", "towers"}, "model")
    m.setdefault("schema_version", 1)
    if type(m["schema_version"]) is not int or m["schema_version"] != 1:
        raise ConfigError("model.schema_version: only version 1 is supported")
    name(m["name"], "model.name")
    if not isinstance(m["dtype"], str) or m["dtype"] not in DTYPE_BYTES:
        raise ConfigError(f"model.dtype: expected one of {sorted(DTYPE_BYTES)}")
    integer(m["dense_features"], "model.dense_features", 0)
    if not isinstance(m["embeddings"], list):
        raise ConfigError("model.embeddings: expected list")
    seen = set()
    for i, e in enumerate(m["embeddings"]):
        p = f"model.embeddings[{i}]"
        obj(e, {"name", "vocab_size", "dim"}, {"name", "vocab_size", "dim"}, p)
        name(e["name"], p + ".name")
        if e["name"] in seen:
            raise ConfigError(f"{p}: duplicate embedding name")
        seen.add(e["name"])
        integer(e["vocab_size"], p + ".vocab_size")
        integer(e["dim"], p + ".dim")
    m.setdefault("dense_mlp", [])
    sizes(m["dense_mlp"], "model.dense_mlp")
    if m["dense_mlp"] and not m["dense_features"]:
        raise ConfigError("model.dense_mlp requires dense_features > 0")
    m.setdefault("interaction", {"kind": "concat"})
    interaction = m["interaction"]
    obj(interaction, {"kind", "projection_dim"}, {"kind"}, "model.interaction")
    if interaction["kind"] not in ("concat", "dot"):
        raise ConfigError("model.interaction.kind: expected concat or dot")
    if "projection_dim" in interaction:
        integer(interaction["projection_dim"], "model.interaction.projection_dim")
    if interaction["kind"] == "dot" and "projection_dim" not in interaction:
        raise ConfigError("dot interaction requires projection_dim")
    if interaction["kind"] == "concat" and "projection_dim" in interaction:
        raise ConfigError("concat interaction does not use projection_dim")
    m.setdefault("sequence_encoder", None)
    seq = m["sequence_encoder"]
    if seq is not None:
        fields = {"vocab_size", "embedding_dim", "hidden_size", "num_heads", "ffn_size", "num_layers"}
        obj(seq, fields, fields, "model.sequence_encoder")
        for key, val in seq.items():
            integer(val, "model.sequence_encoder." + key)
        if seq["hidden_size"] % seq["num_heads"]:
            raise ConfigError("sequence hidden_size must be divisible by num_heads")
    if not (m["dense_features"] or m["embeddings"] or seq):
        raise ConfigError("model needs at least one input feature")
    if not isinstance(m["towers"], list) or not m["towers"]:
        raise ConfigError("model.towers: expected nonempty list")
    seen = set()
    for i, tower in enumerate(m["towers"]):
        p = f"model.towers[{i}]"
        obj(tower, {"name", "hidden_sizes", "output_dim"}, {"name", "hidden_sizes", "output_dim"}, p)
        name(tower["name"], p + ".name")
        if tower["name"] in seen:
            raise ConfigError(f"{p}: duplicate tower name")
        seen.add(tower["name"])
        sizes(tower["hidden_sizes"], p + ".hidden_sizes")
        integer(tower["output_dim"], p + ".output_dim")
    return m


def validate_npu(value):
    n = copy.deepcopy(value)
    fields = {"name", "compute_tops", "memory_bandwidth_gbps", "memory_capacity_bytes",
              "on_chip_memory_bytes", "on_chip_bandwidth_gbps", "matrix_alignment",
              "compute_efficiency", "bandwidth_efficiency", "random_access_efficiency",
              "launch_overhead_us", "supported_ops"}
    obj(n, fields, {"name", "compute_tops", "memory_bandwidth_gbps", "memory_capacity_bytes"}, "npu")
    name(n["name"], "npu.name")
    obj(n["compute_tops"], DTYPE_BYTES, set(), "npu.compute_tops")
    if not n["compute_tops"]:
        raise ConfigError("npu.compute_tops: at least one dtype is required")
    for dtype, engines in n["compute_tops"].items():
        obj(engines, {"matrix", "vector"}, {"matrix", "vector"}, f"npu.compute_tops.{dtype}")
        for engine, rate in engines.items():
            number(rate, f"npu.compute_tops.{dtype}.{engine}", positive=True)
    number(n["memory_bandwidth_gbps"], "npu.memory_bandwidth_gbps", positive=True)
    integer(n["memory_capacity_bytes"], "npu.memory_capacity_bytes")
    defaults = {"on_chip_memory_bytes": 0, "on_chip_bandwidth_gbps": 0,
                "matrix_alignment": 1, "compute_efficiency": 1.0, "bandwidth_efficiency": 1.0,
                "random_access_efficiency": 1.0, "launch_overhead_us": 0.0, "supported_ops": sorted(OPS)}
    for key, default in defaults.items():
        n.setdefault(key, default)
    integer(n["on_chip_memory_bytes"], "npu.on_chip_memory_bytes", 0)
    integer(n["matrix_alignment"], "npu.matrix_alignment")
    number(n["on_chip_bandwidth_gbps"], "npu.on_chip_bandwidth_gbps")
    for key in ("compute_efficiency", "bandwidth_efficiency", "random_access_efficiency"):
        number(n[key], "npu." + key, maximum=1, positive=True)
    number(n["launch_overhead_us"], "npu.launch_overhead_us")
    supported = n["supported_ops"]
    if not isinstance(supported, list) or any(not isinstance(op, str) or op not in OPS for op in supported):
        raise ConfigError(f"npu.supported_ops: expected subset of {sorted(OPS)}")
    return n


def validate_workload(value, model, npu):
    w = copy.deepcopy(value)
    obj(w, {"batch_size", "lookups_per_sample", "sequence_length", "embedding_cache_hit_rate"},
        {"batch_size", "lookups_per_sample"}, "workload")
    integer(w["batch_size"], "workload.batch_size")
    keys = {e["name"] for e in model["embeddings"]}
    obj(w["lookups_per_sample"], keys, keys, "workload.lookups_per_sample")
    for key, val in w["lookups_per_sample"].items():
        integer(val, "workload.lookups_per_sample." + key)
    w.setdefault("sequence_length", 0)
    w.setdefault("embedding_cache_hit_rate", 0.0)
    integer(w["sequence_length"], "workload.sequence_length", 0)
    number(w["embedding_cache_hit_rate"], "workload.embedding_cache_hit_rate", maximum=1)
    if model["sequence_encoder"] is not None and w["sequence_length"] == 0:
        raise ConfigError("sequence encoder requires sequence_length > 0")
    if model["sequence_encoder"] is None and w["sequence_length"] != 0:
        raise ConfigError("sequence_length must be 0 without sequence encoder")
    if w["embedding_cache_hit_rate"] and not (npu["on_chip_memory_bytes"] and npu["on_chip_bandwidth_gbps"]):
        raise ConfigError("embedding cache hits require on-chip capacity and bandwidth")
    if model["dtype"] not in npu["compute_tops"]:
        raise ConfigError(f"npu has no throughput for dtype {model['dtype']}")
    return w


def normalize_inputs(model, npu, workload):
    m, n = validate_model(model), validate_npu(npu)
    return m, n, validate_workload(workload, m, n)


def get_path(model, path):
    current = model
    try:
        for part in path.split("."):
            current = current[int(part)] if isinstance(current, list) else current[part]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ConfigError(f"Unknown model path: {path}") from exc
    return current


def set_path(model, path, value):
    parts = path.split(".")
    parent = get_path(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
    parent[key] = copy.deepcopy(value)


def validate_search(value, model):
    import re
    s = copy.deepcopy(value)
    obj(s, {"method", "seed", "max_evaluations", "parameters", "constraints", "exploration_probability"},
        {"parameters"}, "search")
    s.setdefault("method", "adaptive")
    s.setdefault("seed", 0)
    s.setdefault("max_evaluations", 50)
    s.setdefault("exploration_probability", 0.25)
    s.setdefault("constraints", {})
    if s["method"] not in ("adaptive", "random", "grid"):
        raise ConfigError("search.method: expected adaptive, random or grid")
    integer(s["seed"], "search.seed", 0)
    integer(s["max_evaluations"], "search.max_evaluations")
    number(s["exploration_probability"], "search.exploration_probability", maximum=1)
    if not isinstance(s["parameters"], dict) or not s["parameters"]:
        raise ConfigError("search.parameters: expected nonempty object")
    allowed = r"(?:dense_mlp|embeddings\.\d+\.dim|towers\.\d+\.hidden_sizes|interaction|sequence_encoder\.(?:embedding_dim|hidden_size|num_heads|ffn_size|num_layers))"
    for path, choices in s["parameters"].items():
        if not re.fullmatch(allowed, path):
            raise ConfigError(f"search path is not a mutable structure field: {path}")
        get_path(model, path)
        if not isinstance(choices, list) or not choices:
            raise ConfigError(f"search.parameters.{path}: expected nonempty choices")
        if len({canonical(c) for c in choices}) != len(choices):
            raise ConfigError(f"search.parameters.{path}: duplicate choices")
        for choice in choices:
            if path in ("dense_mlp",) or path.endswith(".hidden_sizes"):
                sizes(choice, path)
            elif path == "interaction":
                probe = copy.deepcopy(model)
                probe["interaction"] = choice
                validate_model(probe)
            else:
                integer(choice, path)
    c = s["constraints"]
    obj(c, {"parameter_ratio", "max_latency_ms", "max_memory_bytes"}, set(), "search.constraints")
    c.setdefault("parameter_ratio", [0.9, 1.1])
    ratio = c["parameter_ratio"]
    if not isinstance(ratio, list) or len(ratio) != 2:
        raise ConfigError("parameter_ratio: expected [min, max]")
    for bound in ratio:
        number(bound, "parameter_ratio", positive=True)
    if ratio[0] > ratio[1]:
        raise ConfigError("parameter_ratio: min exceeds max")
    if "max_latency_ms" in c:
        number(c["max_latency_ms"], "max_latency_ms", positive=True)
    if "max_memory_bytes" in c:
        integer(c["max_memory_bytes"], "max_memory_bytes")
    return s
