"""Process-local tiling override for the audited CANN 8.1.RC1 FP16 Add scheduler."""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path

from .util import TuningError


SCHEDULER = "python/site-packages/tbe/dsl/unify_schedule/vector/elewise/elewise_schedule.py"
SOURCE_SHA256 = "ab69a221f17be927b493320b0915ba06095ef5594fd088ee3f76ff76fcf9d9a8"


def validate(config, params):
    op = config["operator"]
    if (config["backend"] != "ascend" or op["kind"] != "add" or op["dtype"] != "float16"
            or config["ascend"]["cann_version"] != "8.1.RC1"
            or config.get("baseline", {}).get("kind", "builtin") != "builtin"):
        raise TuningError("stock_add_tiling requires built-in FP16 Add on CANN 8.1.RC1")
    if not isinstance(params, dict) or set(params) != {"block_elems", "ub_elems", "double_buffer"}:
        raise TuningError("stock_add_tiling requires block_elems, ub_elems and double_buffer")
    for key in ("block_elems", "ub_elems"):
        if type(params[key]) != int or params[key] <= 0 or params[key] % 16:
            raise TuningError(f"stock_add_tiling.{key} must be a positive multiple of 16")
    if type(params["double_buffer"]) != bool or params["ub_elems"] > params["block_elems"]:
        raise TuningError("Invalid stock Add buffering or UB tile exceeds block tile")


def compiler_environment(config, directory, env):
    scheduler = Path(env["ASCEND_HOME_PATH"])/SCHEDULER
    if not scheduler.is_file() or hashlib.sha256(scheduler.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise TuningError(f"Unrecognized CANN 8.1.RC1 Add scheduler: {scheduler}")
    hook = directory/"stock_add_hook"
    hook.mkdir()
    (hook/"sitecustomize.py").write_text("from paretopilot.stock_add_81 import install\ninstall()\n")
    cache, work = directory/"compiler_cache", directory/"compiler_work"
    cache.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    return {**env, "PYTHONPATH": str(hook)+os.pathsep+env.get("PYTHONPATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1", "ASCEND_CACHE_PATH": str(cache), "ASCEND_WORK_PATH": str(work),
            "PP_ADD_SCHEDULER": str(scheduler.resolve()), "PP_ADD_RECORD": str(directory/"stock-add-tiling.jsonl"),
            "PP_ADD_TILING": json.dumps({**config["ascend"]["stock_add_tiling"], "numel": config["operator"]["numel"]})}


def apply_tiling(tiling, run_info, shape, inputs, alignment, capacity, capacity_db, cores):
    params = json.loads(os.environ["PP_ADD_TILING"])
    if (shape != [params["numel"]] or len(inputs) != 2
            or any(i["shape"] != shape or i["dtype"] != "float16" for i in inputs)):
        return
    block, ub, db = params["block_elems"], params["ub_elems"], params["double_buffer"]
    blocks = (params["numel"]+block-1)//block
    if (not tiling["need_multi_core"] or tiling["ub_axis"] or tiling["block_axis"] or blocks > cores
            or ub % alignment or ub > (capacity_db if db else capacity)):
        raise TuningError("Requested Add schedule exceeds the installed scheduler limits")
    before = dict(tiling)
    tiling.update(block_factor=block, ub_factor=ub, is_need_db=int(db), is_need_nb=0)
    run_info["block_dim"] = blocks
    with open(os.environ["PP_ADD_RECORD"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"before": before, "after": tiling, "block_dim": blocks})+"\n")


def install():
    target = Path(os.environ["PP_ADD_SCHEDULER"])
    original = importlib.machinery.SourceFileLoader.get_code

    def get_code(loader, fullname):
        if Path(loader.path).resolve() != target:
            return original(loader, fullname)
        data = target.read_bytes()
        marker = '        self._const_block_dims = run_info.get("block_dim")'
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256 or data.decode().count(marker) != 1:
            raise TuningError("Installed Add scheduler changed before compilation")
        invocation = ("        from paretopilot.stock_add_81 import apply_tiling\n"
                      "        apply_tiling(tiling_data, run_info, output_shape_list, inputs, self._ub_factor_align,\n"
                      "                     max_available_ub, max_available_ub_db, util.get_core_num())\n")
        return compile(data.decode().replace(marker, invocation+marker), str(target), "exec")

    importlib.machinery.SourceFileLoader.get_code = get_code
