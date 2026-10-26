"""Experimental child-process-only instrumentation of the installed V2 wrapper.

No installed file is written. A successful build record is mandatory: Python
startup hooks and import loaders are not assumed to work in every ATC release.
"""
import ast
import copy
import hashlib
import importlib.machinery
import inspect
import json
import os
from pathlib import Path

HELPER = "__paretopilot_v2_build_98cb"
CALLERS = {"_dynamic_build", "_binary_constant_build"}


def emit(record):
    payload = (json.dumps(record, sort_keys=True, allow_nan=False)+"\n").encode()
    fd = os.open(os.environ["PARETOPILOT_TBE_RECORD"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def invoke_build(source_path, source_sha256, original, *args, **kwargs):
    caller = inspect.currentframe().f_back
    # This source file also serves related operators. Leave those untouched.
    if caller.f_locals.get("extra_params", {}).get("op_type") != "BatchMatMulV2":
        return original(*args, **kwargs)
    policy = json.loads(os.environ["PARETOPILOT_TBE_POLICY"])
    if set(policy) != {"enable_db_fold"} or type(policy["enable_db_fold"]) is not bool:
        raise ValueError("Only the audited enable_db_fold boolean is supported")
    args = list(args)
    config = args[1] if len(args) > 1 else kwargs.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("build_args"), dict):
        raise ValueError("Installed V2 build config differs from the audited layout")
    before = copy.deepcopy(config["build_args"])
    if before.get("enable_db_fold") is not True:
        raise ValueError("Installed V2 default enable_db_fold is not True; audit this branch first")
    patched = {**config, "build_args": {**config["build_args"], **policy}}
    if len(args) > 1:
        args[1] = patched
    else:
        kwargs["config"] = patched
    record = {"event": "build_call", "pid": os.getpid(), "source_path": source_path,
              "source_sha256": source_sha256, "caller": caller.f_code.co_name,
              "operator": "BatchMatMulV2", "kernel_name": config.get("name"),
              "before_build_args": before, "effective_build_args": copy.deepcopy(patched["build_args"])}
    emit(record)
    result = original(*args, **kwargs)
    emit({**record, "event": "build_return"})
    return result


def transform_source(source, filename):
    tree = ast.parse(source, filename=filename)
    source_hash = hashlib.sha256(source).hexdigest()
    if any(isinstance(n, ast.Name) and n.id == HELPER for n in ast.walk(tree)):
        raise ValueError("Probe helper name already exists in installed source")
    class Transform(ast.NodeTransformer):
        current = None
        count = 0

        def visit_FunctionDef(self, node):
            previous = self.current
            self.current = node.name
            self.generic_visit(node)
            self.current = previous
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (self.current in CALLERS and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "tbe"
                    and node.func.attr == "build"):
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id=HELPER, ctx=ast.Load()),
                    args=[ast.Constant(str(filename)), ast.Constant(source_hash), node.func, *node.args],
                    keywords=node.keywords), node)
            return node
    transformer = Transform()
    tree = transformer.visit(tree)
    if not transformer.count:
        raise ValueError("No audited tbe.build call sites found in installed V2 source")
    insertion = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str) else 0
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom) and tree.body[insertion].module == "__future__":
        insertion += 1
    tree.body.insert(insertion, ast.ImportFrom(module="paretopilot.tbe_probe_hook",
                                              names=[ast.alias(name="invoke_build", asname=HELPER)], level=0))
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def install():
    target = Path(os.environ["PARETOPILOT_TBE_SOURCE"]).resolve(strict=True)
    expected = os.environ["PARETOPILOT_TBE_SOURCE_SHA256"]
    original = importlib.machinery.SourceFileLoader.get_code
    def get_code(loader, fullname):
        if Path(loader.path).resolve() != target:
            return original(loader, fullname)
        source = target.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Installed V2 source changed after probe preparation")
        return transform_source(source, str(target))
    importlib.machinery.SourceFileLoader.get_code = get_code
    emit({"event": "hook_ready", "pid": os.getpid(), "source_path": str(target)})
