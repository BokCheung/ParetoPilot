"""Static Jsonnet literals and a deliberately small, non-executing ref DSL."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import re

from .config import ConfigError, DTYPE_BYTES, integer, load_json, name, obj, validate_npu


TOKEN = re.compile(r'''\s+|//[^\n]*|\#[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|[{}\[\]:,;=]''')


def parse_static_jsonnet(source):
    """Accept literals, comments, trailing commas and one local result binding.

    No imports, function calls, interpolation, or code execution. Dynamic Jsonnet
    must first be exported to JSON using the user's own Jsonnet runtime.
    """
    tokens, position = [], 0
    while position < len(source):
        match = TOKEN.match(source, position)
        if not match:
            line = source.count("\n", 0, position) + 1
            raise ConfigError(f"Unsupported Jsonnet syntax at line {line}; export dynamic Jsonnet to JSON first")
        token = match.group()
        if not (token.isspace() or token.startswith(("//", "/*", "#"))):
            tokens.append(token)
        position = match.end()
    cursor = 0

    def peek():
        return tokens[cursor] if cursor < len(tokens) else None

    def take(expected=None):
        nonlocal cursor
        token = peek()
        if token is None or (expected is not None and token != expected):
            raise ConfigError(f"Static Jsonnet: expected {expected or 'value'}, got {token!r}")
        cursor += 1
        return token

    def value():
        token = take()
        if token in ("[", "{"):
            result = [] if token == "[" else {}
            close = "]" if token == "[" else "}"
            while peek() != close:
                if token == "[":
                    result.append(value())
                else:
                    key = take()
                    if key.startswith(("'", '"')):
                        key = ast.literal_eval(key)
                    elif not re.fullmatch(r"[A-Za-z_]\w*", key):
                        raise ConfigError("Static Jsonnet: expected object key")
                    if key in result:
                        raise ConfigError(f"Duplicate Jsonnet key: {key}")
                    take(":")
                    result[key] = value()
                if peek() == close:
                    break
                take(",")
            take(close)
            return result
        if token in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[token]
        if token.startswith(("'", '"')):
            return ast.literal_eval(token)
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token):
            result = float(token) if any(c in token for c in ".eE") else int(token)
            if isinstance(result, float) and not math.isfinite(result):
                raise ConfigError("Static Jsonnet: non-finite number")
            return result
        raise ConfigError(f"Unsupported Jsonnet expression {token!r}; export dynamic Jsonnet to JSON first")

    try:
        binding = None
        if peek() == "local":
            take("local")
            binding = take()
            if not re.fullmatch(r"[A-Za-z_]\w*", binding):
                raise ConfigError("Static Jsonnet: invalid local binding")
            take("=")
        result = value()
        if binding:
            take(";")
            take(binding)
        if cursor != len(tokens):
            raise ConfigError("Unsupported Jsonnet expression after literal result")
        return result
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ConfigError(f"Invalid static Jsonnet: {exc}") from exc


def load_model(path):
    path = Path(path)
    value = parse_static_jsonnet(path.read_text(encoding="utf-8-sig")) if path.suffix.lower() == ".jsonnet" else load_json(path)
    if isinstance(value, list):
        return {"format": "native_nodes", "name": path.stem, "nodes": value}
    if isinstance(value, dict) and "model_structure" in value:
        return {"format": "native_nodes", "name": value.get("name", path.stem), "nodes": value["model_structure"]}
    return value


def normalize_native(model, npu, workload):
    model, workload = copy.deepcopy(model), copy.deepcopy(workload)
    obj(workload, {"batch_size", "dtype", "ready_features", "dense_features", "module_options",
                   "node_options", "description", "synthetic", "outputs"},
        {"batch_size", "dtype", "ready_features"}, "native workload")
    integer(workload["batch_size"], "workload.batch_size")
    if not isinstance(workload["dtype"], str) or workload["dtype"] not in DTYPE_BYTES:
        raise ConfigError("workload.dtype: unsupported dtype")
    model["dtype"] = workload["dtype"]
    npu = validate_npu(npu)
    if model["dtype"] not in npu["compute_tops"]:
        raise ConfigError("NPU has no compute rates for workload.dtype")
    nodes = model.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ConfigError("Native model must contain a nonempty list of nodes")
    seen = set()
    for node in nodes:
        obj(node, {"name", "type", "parameters", "inputs", "outputs"}, {"name", "type", "inputs", "outputs"}, "node")
        for field in ("name", "type", "outputs"):
            name(node[field], "node." + field)
        if node["name"] in seen or node["name"] == "inputs":
            raise ConfigError(f"Duplicate/reserved node name: {node['name']}")
        seen.add(node["name"])
        if not isinstance(node.get("parameters", {}), dict) or not isinstance(node["inputs"], dict):
            raise ConfigError(f"{node['name']}: inputs and parameters must be objects")
    for field in ("ready_features", "dense_features"):
        features = workload.setdefault(field, [])
        if not isinstance(features, list):
            raise ConfigError(f"workload.{field}: expected list")
        names = set()
        for feature in features:
            obj(feature, {"name", "shape", "region", "kind"}, {"name", "shape", "region"}, field)
            name(feature["name"], field + ".name")
            if feature["name"] in names:
                raise ConfigError(f"Duplicate feature: {field}.{feature['name']}")
            names.add(feature["name"])
            if not isinstance(feature["shape"], list) or not feature["shape"]:
                raise ConfigError(f"{feature['name']}: shape excludes batch and must be a nonempty list")
            for dim in feature["shape"]:
                integer(dim, feature["name"] + ".shape")
            if not isinstance(feature["region"], list) or any(not isinstance(r, str) for r in feature["region"]):
                raise ConfigError("Feature region must be a list of strings")
            if field == "ready_features":
                feature.setdefault("kind", "single_discrete" if feature["shape"][0] == 1 else "multiple_discrete")
                if feature["kind"] not in ("single_discrete", "multiple_discrete"):
                    raise ConfigError("Feature kind must be single_discrete or multiple_discrete")
    for field in ("module_options", "node_options"):
        workload.setdefault(field, {})
        if not isinstance(workload[field], dict) or any(not isinstance(v, dict) for v in workload[field].values()):
            raise ConfigError(f"workload.{field}: expected objects indexed by type/name")
    unknown = set(workload["node_options"]) - seen
    if unknown:
        raise ConfigError(f"Unknown node_options: {sorted(unknown)}")
    if "synthetic" in workload and type(workload["synthetic"]) is not bool:
        raise ConfigError("workload.synthetic must be boolean")
    if "outputs" in workload:
        if not isinstance(workload["outputs"], list) or not workload["outputs"] or any(not isinstance(x, str) or not x.startswith("ref::") for x in workload["outputs"]):
            raise ConfigError("workload.outputs: expected nonempty list of ref:: strings")
    return model, npu, workload


@dataclass
class Feature:
    tensor: str
    metadata: dict


@dataclass
class FeatureSet:
    features: dict[str, Feature]


def flatten(value):
    if isinstance(value, FeatureSet):
        return [f.tensor for f in value.features.values()]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in flatten(child)]
    if isinstance(value, str):
        return [value]
    raise ConfigError(f"Expected tensor/list of tensors, got {type(value).__name__}")


def predicate(expression, metadata):
    """Evaluate only metadata comparisons; never eval Python or the business DSL."""
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
            return node.value
        if isinstance(node, ast.Name) and node.id in metadata:
            return metadata[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [visit(x) for x in node.elts]
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [visit(x) for x in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            results = []
            for operation, right_node in zip(node.ops, node.comparators):
                right = visit(right_node)
                if isinstance(operation, ast.Eq):
                    result = left == right
                elif isinstance(operation, ast.NotEq):
                    result = left != right
                elif isinstance(operation, ast.In):
                    result = left in right
                elif isinstance(operation, ast.NotIn):
                    result = left not in right
                else:
                    raise ConfigError("Unsupported feature comparison")
                results.append(result)
                left = right
            return all(results)
        raise ConfigError(f"Unsupported feature predicate: {expression}")
    try:
        return bool(visit(ast.parse(expression, mode="eval").body))
    except (SyntaxError, TypeError, RecursionError) as exc:
        raise ConfigError(f"Invalid feature predicate: {expression}") from exc


def resolve(value, environment):
    if isinstance(value, list):
        return [resolve(v, environment) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, environment) for k, v in value.items()}
    if not isinstance(value, str) or not value.startswith("ref::"):
        return value
    expression = value[5:]
    if expression.startswith("values("):
        match = re.fullmatch(r"values\((.*)\)(?:\[(\d+)\])?", expression)
        if not match:
            raise ConfigError(f"Unsupported reference: {value}")
        selected = flatten(resolve("ref::" + match[1], environment))
        if match[2] is not None:
            index = int(match[2])
            if index >= len(selected):
                raise ConfigError(f"Reference index out of range: {value}")
            return selected[index]
        return selected
    match = re.match(r"([A-Za-z_]\w*)(.*)", expression)
    if not match or match[1] not in environment:
        raise ConfigError(f"Unknown or forward reference: {value}")
    current, rest = environment[match[1]], match[2]
    while rest:
        if isinstance(current, FeatureSet):
            expression = rest.removeprefix(".")
            if expression in current.features:
                return current.features[expression].tensor
            if expression.startswith("features#"):
                tag = expression.split("#", 1)[1]
                selected = {k: f for k, f in current.features.items()
                            if f.metadata.get("kind") == tag or f.metadata["type"] == tag}
            else:
                selected = {k: f for k, f in current.features.items() if predicate(expression, f.metadata)}
            if not selected:
                raise ConfigError(f"No input features match {value}; supply feature shapes and region metadata in workload")
            return FeatureSet(selected)
        match = re.match(r"\.([A-Za-z_]\w*)|\[(\d+)\]", rest)
        if not match:
            raise ConfigError(f"Unsupported reference suffix: {rest}")
        key = match[1] if match[1] is not None else int(match[2])
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ConfigError(f"Unresolved reference: {value}") from exc
        rest = rest[match.end():]
    return current
