from copy import deepcopy
import ctypes as C
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from paretopilot import official, official_add
from paretopilot.config import load_config
from paretopilot.direct_worker import Runtime
from paretopilot.direct_backend import validate_result
from paretopilot.search import run
from paretopilot.util import TuningError, digest, file_hash, read_json, tree_hash, write_json
from tests.test_direct_kernel import FakeNative

ROOT = Path(__file__).resolve().parents[1]


class AddNative(FakeNative):
    def __getattr__(self, name):
        if name not in ("OpTilingForCompile", "aclrtLaunchKernelWithConfig"):
            return super().__getattr__(name)
        def call(*args):
            self.calls.append(name)
            if name == "OpTilingForCompile":
                assert args[0] == b"AddV2"
                assert len(json.loads(args[3])) == 2
                assert json.loads(args[5]) == []
                p = json.loads(args[1])["paretopilot_params"]
                assert os.environ["PARETOPILOT_TILE_ELEMS"] == str(p["tile_elems"])
                r = {**p, "physical_cores": 48, "block_dim": p["core_limit"] or 48, "used_core_num": p["core_limit"] or 48,
                     "tiling_key": 3, "tile_data_num": p["tile_elems"] or 16384}
                small_blocks, extra = divmod(self.reference.size//16, r["block_dim"])
                small, big, tile = small_blocks*16, (small_blocks+1)*16, r["tile_data_num"]
                small_tiles, big_tiles = (small+tile-1)//tile, (big+tile-1)//tile
                r.update(small_core_data_num=small, big_core_data_num=big, small_tiles=small_tiles,
                         big_tiles=big_tiles, tail_block_num=extra, ub_size=196608)
                os.write(2, ("PARETOPILOT_PARAMS="+digest(p)+"\nPARETOPILOT_OFFICIAL="+official_add.COMMIT+
                             "\nPARETOPILOT_ADD_TILING="+json.dumps(r)+"\n").encode())
                info = {"block_dim": r["block_dim"], "tiling_key": 3, "workspaces": [32], "clear_atomic": False,
                        "tiling_data": struct.pack("<8q", small, big, big_tiles, small_tiles, tile,
                                                 small-(small_tiles-1)*tile, big-(big_tiles-1)*tile, extra).hex()}
                raw = json.dumps(info).encode()+b"\0"
                C.memmove(args[6], raw, len(raw))
                return 1
            assert len(self.arguments) == 5
            import numpy as np
            a = np.frombuffer(C.string_at(self.arguments[0], self.reference.nbytes), dtype=np.float16)
            b = np.frombuffer(C.string_at(self.arguments[1], self.reference.nbytes), dtype=np.float16)
            output = (a+b).astype(np.float16)
            C.memmove(self.arguments[2], output.ctypes.data, output.nbytes)
            return 0
        setattr(self, name, call)
        return call


class OfficialAddTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_add_v2.json")

    def test_pinned_source_controls_precede_dependencies_device_is_unchanged(self):
        p = self.root/"project"
        result = official_add.materialize(p)
        self.assertEqual(result["commit"], official_add.COMMIT)
        with zipfile.ZipFile(official_add.VENDOR/"ops-math-add-v2.zip") as archive:
            for name in archive.namelist():
                if "/op_kernel/" in name and not name.endswith("/"):
                    self.assertEqual((p/name).read_bytes(), archive.read(name))
        host = (p/official_add.SOURCE/"op_host/add_v2_tiling.cpp").read_text(encoding="utf8")
        self.assertLess(host.index("if (ppCore) coreNum"), host.index("uint64_t coreNum64"))
        self.assertLess(host.index("if (ppTile)"), host.index("uint32_t tileDataNum ="))
        self.assertLess(host.index("if (ppTile)"), host.index("uint32_t finalSmallTileNum"))
        self.assertIn("PARETOPILOT_ADD_TILING", host)
        self.assertIn("InferDataType(PpInferType)", (p/official_add.SOURCE/"op_host/add_v2_infershape.cpp").read_text(encoding="utf8"))
        with self.assertRaises(TuningError):
            official_add.validate_params({"core_limit": 49, "tile_elems": 0})
        bad = deepcopy(self.c)
        bad["operator"]["numel"] += 1
        write_json(self.root/"bad.json", bad)
        with self.assertRaisesRegex(TuningError, "divisible by 16"):
            load_config(self.root/"bad.json")

    def test_add_direct_abi_has_two_inputs_and_five_pointers_and_48_vector_cores(self):
        import numpy as np
        c = deepcopy(self.c)
        c["operator"]["numel"] = 1024
        c["measurement"].update(repeats=3, warmup=1)
        for i in range(2):
            np.save(self.root/f"input_{i}.npy", np.full((1024,), i+1, dtype=np.float16))
        reference = np.full((1024,), 3, dtype=np.float16)
        np.save(self.root/"reference.npy", reference)
        opp = self.root/"opp/op_impl/ai_core/tbe/op_tiling"
        opp.mkdir(parents=True)
        (opp/"host.so").write_bytes(b"fixture")
        binary = self.root/"kernel.o"
        binary.write_bytes(b"fixture")
        meta = self.root/"kernel.json"
        write_json(meta, {"compileInfo": {}, "kernelList": [{"tilingKey": 3, "kernelName": "add_v2_3"}]})
        launcher = self.root/"launcher.so"
        launcher.write_bytes(b"fixture")
        build = {"commit": official_add.COMMIT, "operator": "AddV2", "opp_path": str(self.root/"opp"),
                 "installed_sha256": tree_hash(self.root/"opp"), "launcher_library": str(launcher), "launcher_sha256": file_hash(launcher),
                 "kernel_catalogs": [{"metadata": str(meta), "metadata_sha256": file_hash(meta), "binary": str(binary), "binary_sha256": file_hash(binary)}]}
        fake = AddNative(reference)
        from tests.test_direct_kernel import DirectKernelTests
        platform = DirectKernelTests().platform_fixture()
        with patch("paretopilot.direct_worker.C.CDLL", return_value=fake), patch.dict(os.environ), \
                patch.dict("sys.modules", {"tbe.common.platform": platform}):
            runtime = Runtime({"config": c, "build": build, "case": str(self.root)})
            try:
                for i, params in enumerate((official.default_params(c), {"core_limit": 24, "tile_elems": 256})):
                    directory = self.root/str(i)
                    directory.mkdir()
                    result = runtime.evaluate({"params": params, "directory": str(directory)})
                    validate_result(result, c, params, build)
                    self.assertEqual(result["operator_identity"]["Op Type"], "AddV2")
                    self.assertTrue(result["correctness"]["passed"])
            finally:
                runtime.close()
        self.assertFalse(fake.allocations)

    def test_add_source_smoke_search_profiler_metric_and_om_confirmation(self):
        c = self.c
        c["search"].update(population=2, generations=2, budget=3)
        outer = self
        calls = []
        class Backend:
            def __init__(self, config, root):
                self.om_backend = SimpleNamespace(evaluate=lambda p, d, r: self.measure(p, d, r, "om"))
            def evaluate(self, p, d, r):
                return self.measure(p, d, r, "om" if r == "baseline" else "kernel")
            def measure(self, p, d, r, execution):
                calls.append((r, execution, p))
                changed = r == "candidate" and p != official.default_params(c)
                us = 6 if r == "baseline" else 7 if changed else 10
                records = [{**p, "tile_data_num": p.get("tile_elems") or 16384, "block_dim": p.get("core_limit") or 48, "tiling_key": 3}]
                return {"status": "ok", "mode": "ascend", "execution": execution, "timing_method": "kernel_task_duration_us" if execution == "kernel" else "task_duration_us",
                        "duration_us": us, "samples_us": [us]*20, "measured_bytes": None, "correctness": {"passed": True},
                        "input_shapes": [[c["operator"]["numel"]]]*2, "output_shape": [c["operator"]["numel"]],
                        "operator_identity": {"Op Type": "Add" if r == "baseline" else "AddV2"},
                        "selection": {"commit": official_add.COMMIT, "operator": "AddV2", "effective_tiling": records}}
            def close(self): pass
        from scripts.test_910b import verify_result
        with patch("paretopilot.search.inspect_environment", return_value={"ready": True}), patch("paretopilot.direct_backend.KernelBackend", Backend):
            run(c, self.root/"smoke", smoke=True)
            self.assertEqual(verify_result(self.root/"smoke", True)["status"], "passed")
            calls.clear()
            run(c, self.root/"full")
            result = read_json(self.root/"full/summary.json")
            self.assertEqual(result["selected"], "candidate")
            self.assertTrue(result["confirmation"]["tuning_accepted"])
            self.assertEqual(verify_result(self.root/"full", False)["status"], "passed")
            self.assertTrue(all(t["execution"] == "om" for t in result["confirmation"]["trials"]))
            text = (self.root/"full/report.md").read_text(encoding="utf8")
            self.assertIn("experimental community contribution", text)
            self.assertIn("Primary AddV2 baseline", text)
            self.assertEqual(result["agent"]["kind"], "dummy")

    def test_raw_tiling_bytes_are_checked_before_launch(self):
        fields = (16, 32, 1, 1, 256, 16, 32, 16)
        info = {"block_dim": 48, "tiling_key": 3, "tiling_data": struct.pack("<8q", *fields).hex()}
        records = [{"small_core_data_num": 16, "big_core_data_num": 32, "big_tiles": 1, "small_tiles": 1,
                    "tile_data_num": 256, "tail_block_num": 16, "ub_size": 196608}]
        official_add.validate_tiling_bytes(info, records, 1024)
        for bad in ({**info, "tiling_key": 0}, {**info, "tiling_data": "00"}, {**info, "block_dim": 24}):
            with self.assertRaises(TuningError):
                official_add.validate_tiling_bytes(bad, records, 1024)
        with self.assertRaises(TuningError):
            official_add.validate_tiling_bytes(info, [{**records[0], "tile_data_num": 512}], 1024)

    def test_upstream_generated_attribute_free_recipe_filters_fp16_without_inventing_keys(self):
        from paretopilot.direct_build import fp16_config
        project = self.root/"recipe-project"
        official_add.materialize(project)
        ini = self.root/"ops.ini"
        lines = ["[AddV2]"]
        for prefix, name in (("input0", "x1"), ("input1", "x2"), ("output0", "y")):
            lines += [prefix+".name="+name, prefix+".paramType=required", prefix+".dtype=float,float16", prefix+".format=ND,ND"]
        ini.write_text("\n".join(lines)+"\n", encoding="utf8")
        output = self.root/"add_v2"
        output.mkdir()
        subprocess.run([sys.executable, str(project/"scripts/util/ascendc_bin_param_build.py"), str(ini), str(output), "ascend910b", "--ops", "add_v2"],
                       check=True, capture_output=True, timeout=30)
        subprocess.run([sys.executable, str(project/"scripts/kernel/binary_script/merge_ops_config_json.py"), str(output)],
                       check=True, capture_output=True, timeout=30)
        recipe = read_json(output/"add_v2_binary.json")
        self.assertEqual(len(recipe["op_list"]), 2)
        selected = fp16_config(recipe)
        self.assertEqual(len(selected["op_list"]), 1)
        self.assertNotIn("attrs", selected["op_list"][0])
        self.assertIn(selected["op_list"][0], recipe["op_list"])

    def test_add_build_cache_reuses_only_verified_binary_and_opp(self):
        c = deepcopy(self.c)
        c["implementation"]["cache_dir"] = str(self.root/"cache")
        tool = self.root/"compiler"
        tool.write_bytes(b"fixture compiler")
        def build(config, directory, env):
            opp = directory/"opp"
            opp.mkdir()
            (opp/"host.so").write_bytes(b"fixture")
            binary, meta, launcher = directory/"k.o", directory/"k.json", directory/"launcher.so"
            binary.write_bytes(b"fixture kernel")
            launcher.write_bytes(b"fixture launcher")
            write_json(meta, {"kernelList": [{"kernelName": "add_v2", "tilingKey": 3}]})
            return {"opp_path": str(opp), "installed_sha256": tree_hash(opp), "launcher_library": str(launcher), "launcher_sha256": file_hash(launcher),
                    "kernel_catalogs": [{"metadata": str(meta), "metadata_sha256": file_hash(meta), "binary": str(binary), "binary_sha256": file_hash(binary)}]}
        with patch("paretopilot.official_add.shutil.which", return_value=str(tool)), patch("paretopilot.official_add._build", side_effect=build) as compiler:
            for i, p in enumerate((official_add.DEFAULT_PARAMS, {"core_limit": 24, "tile_elems": 256})):
                directory = self.root/str(i)
                directory.mkdir()
                result = official_add.build(c, p, directory, {})
                self.assertEqual(result["cache_hit"], bool(i))
            self.assertEqual(compiler.call_count, 1)
            Path(result["kernel_catalogs"][0]["binary"]).write_bytes(b"corrupt")
            directory = self.root/"rebuild"
            directory.mkdir()
            self.assertFalse(official_add.build(c, official_add.DEFAULT_PARAMS, directory, {})["cache_hit"])
            self.assertEqual(compiler.call_count, 2)

    def test_stock_add_registration_and_script_route_do_not_claim_source_tuning(self):
        from scripts import inspect_cann_operator as inspector, test_910b as script
        config = self.root/"aic-ascend910b-ops-info.json"
        write_json(config, {"Add": {"opFile": {"value": "add"}, "opInterface": {"value": "add"}}})
        try:
            inspector.select_operator("Add")
            self.assertEqual(inspector.registration(config)["opFile"]["value"], "add")
            self.assertTrue(inspector.OP_MATCH.search("/impl/dynamic/add.py"))
            self.assertFalse(inspector.OP_MATCH.search("AddV2"))
        finally:
            inspector.select_operator("BatchMatMulV2")
        args = script.parser().parse_args(["--cann-path", str(self.root/"cann"), "--soc", "Ascend910B2", "--operator", "add", "--add-implementation", "stock", "--scope", "inspect", "--deps", "check"])
        configs = script.prepare_configs(args, self.root)
        self.assertFalse(official.enabled(read_json(configs["add"])))
        session = script.Session(self.root, args)
        stages = []
        def execute(name, argv):
            stages.append(name)
            if name == "stock-add-inspect":
                self.assertIn("Add", argv)
                write_json(self.root/"stock-add-inspect/registration-and-sources.json", {"registrations": [], "candidate_files": []})
        session.run = execute
        with patch("scripts.test_910b.verify_baseline_result", return_value={"status": "passed"}):
            script.execute_suite(session, args, "python3", configs)
        self.assertIn("add-stock-baseline", stages)
        self.assertNotIn("add-smoke", stages)
        self.assertNotIn("unit-tests", stages)
        self.assertFalse(session.data["results"]["add-stock-baseline"]["tuning_supported"])
