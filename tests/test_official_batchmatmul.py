from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from paretopilot import official
from paretopilot.backend import AscendBackend
from paretopilot.build import op_names
from paretopilot.config import load_config
from paretopilot.roofline import analyze
from paretopilot.routing import same_operator
from paretopilot.search import confirmation, run
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, read_json, write_json, file_hash, tree_hash, digest
from paretopilot.workload import shapes
from scripts import test_910b as script

ROOT = Path(__file__).resolve().parents[1]


class OfficialBatchMatmulTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        self.c["measurement"]["confirmation_rounds"] = 2

    def trial(self, role, params=None, time=None):
        params = params or official.default_params(self.c)
        inputs, output = shapes(self.c["operator"])
        changed = params != official.BATCH_DEFAULT_PARAMS
        record = {**params, "base_m": 64 if changed else 128, "base_n": 128, "base_k": 64,
                  "used_core_num": 12, "tiling_key": 1, "block_dim": 12, "multi_batch": 0,
                  "load_mode": 0, "batch_count": 1}
        return {"status": "ok", "mode": "ascend", "role": role, "params": params,
                "duration_us": time or {"baseline": 7, "source_baseline": 10, "candidate": 8}[role],
                "samples_us": [8]*20, "measured_bytes": None, "correctness": {"passed": True},
                "input_shapes": inputs, "output_shape": output,
                "operator_identity": {"Op Type": "BatchMatMulV2" if role == "baseline" else "BatchMatMulV3"},
                "selection": None if role == "baseline" else {"commit": official.COMMIT,
                    "operator": "BatchMatMulV3", "effective_tiling": [record]}}

    def test_all_shapes_validate_and_have_real_host_parameter_domains(self):
        for suffix in ("", "_shared_rhs", "_batched_rhs"):
            c = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            self.assertTrue(official.batch_enabled(c))
            self.assertFalse(same_operator(c))  # Source baseline policy, not stock-V2 matching.
            self.assertEqual(len(enumerate_space(c)), 12)
            self.assertIn("batch_strategy", analyze(c)["suggested_priority"])
            self.assertEqual(op_names(c), ("ParetoPilotOfficialBatchMatMul", "batch_mat_mul_v3"))
        self.assertEqual(shapes(load_config(ROOT/"examples/ascend910b_batchmatmul_batched_rhs.json")["operator"]),
                         ([[32, 1, 128], [32, 128, 100]], [32, 1, 100]))

    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("numpy", "torch", "onnx")), "export dependencies required")
    def test_three_workloads_export_reference_and_batch_source_graph_shapes(self):
        import numpy as np
        import onnx
        from paretopilot.worker import export_case
        for suffix in ("", "_shared_rhs", "_batched_rhs"):
            config = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            directory = self.root/(suffix or "2d")
            directory.mkdir()
            export_case(config, directory)
            backend = object.__new__(AscendBackend)
            backend.c, backend.case = config, directory
            backend.candidate_graph(directory/"candidate.onnx")
            expected_inputs, expected_output = shapes(config["operator"])
            model = onnx.load(str(directory/"candidate.onnx"))
            dims = lambda tensor: [d.dim_value for d in tensor.type.tensor_type.shape.dim]
            self.assertEqual([dims(v) for v in model.graph.input], expected_inputs)
            self.assertEqual(dims(model.graph.output[0]), expected_output)
            self.assertEqual(model.graph.node[0].op_type, "ParetoPilotOfficialBatchMatMul")
            a, b = [np.load(directory/f"input_{i}.npy") for i in range(2)]
            reference = np.load(directory/"reference.npy")
            self.assertEqual(list(reference.shape), expected_output)
            np.testing.assert_allclose(reference, np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16), rtol=0.01, atol=0.01)

    def test_materialize_preserves_kernels_and_default_strategy(self):
        project = self.root/"project"
        provenance = official.materialize(project, official.BATCH_DEFAULT_PARAMS.copy(), batch=True)
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as z:
            for name in z.namelist():
                if "/op_kernel/" in name and not name.endswith("/"):
                    self.assertEqual((project/name).read_bytes(), z.read(name), name)
        host = (project/official.BATCH_HOST).read_text(encoding="utf-8")
        self.assertIn("PP_BATCH_STRATEGY == 0", host)
        self.assertIn("PARETOPILOT_BATCH_TILING", host)
        self.assertIn("bmmTilingData_.matmulTiling.matmulTiling", host)
        self.assertIn("#define PP_BATCH_STRATEGY 0", (project/official.BATCH_HOST).with_name("paretopilot_params.h").read_text(encoding="utf-8"))
        self.assertIn(".InferDataType(InferDataTypeForBatchMatMulV3)", (project/official.BATCH_INFER).read_text(encoding="utf-8"))
        self.assertIn('REGISTER_CUSTOM_OP("BatchMatMulV3")', (project/"parser.cpp").read_text(encoding="utf-8"))
        self.assertIn('"adj_x1"', (project/"parser.cpp").read_text(encoding="utf-8"))
        self.assertNotIn('"transpose_x1"', (project/"parser.cpp").read_text(encoding="utf-8"))
        self.assertEqual(provenance["operator"], "BatchMatMulV3")
        self.assertTrue(provenance["device_dependency_sha256"])

    def test_batch_effective_tiling_requires_applied_strategy_and_valid_core_count(self):
        params = {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        record = self.trial("candidate", params)["selection"]["effective_tiling"][0]
        log = self.root/"atc.log"
        def write(value):
            log.write_text(f"PARETOPILOT_OFFICIAL={official.COMMIT}\nPARETOPILOT_BATCH_TILING="+json.dumps(value))
        write(record)
        self.assertEqual(official.read_effective_tiling(log, params, batch=True), [record])
        for key, value in (("multi_batch", 1), ("load_mode", 1), ("block_dim", 24), ("batch_count", 0)):
            write({**record, key: value})
            with self.subTest(key=key), self.assertRaises(TuningError):
                official.read_effective_tiling(log, params, batch=True)

    def test_confirmation_selects_source_gain_even_when_v2_is_faster(self):
        params = {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        class Backend:
            def evaluate(inner, p, directory, role):
                return self.trial(role, p)
        result = confirmation(self.c, {"params": params}, Backend(), self.root)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected"], "candidate")
        self.assertLess(result["stock_speedup"], 1)
        self.assertGreater(result["primary_speedup"], 1)
        self.assertEqual(result["primary_baseline"], "source_baseline")

    def test_no_gain_or_no_effect_never_selects_stock_or_claims_tuning(self):
        params = {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        for identical in (False, True):
            class Backend:
                def evaluate(inner, p, directory, role):
                    trial = self.trial(role, p, time=8 if identical and role == "candidate" else 10)
                    if identical and role == "candidate":
                        trial["selection"]["effective_tiling"] = self.trial("source_baseline")["selection"]["effective_tiling"]
                    return trial
            result = confirmation(self.c, {"params": params}, Backend(), self.root/str(identical))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["selected"], "source_baseline")

    def test_search_and_source_baseline_command_use_v3_as_primary(self):
        params = {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        c = deepcopy(self.c)
        c["search"].update(budget=1, population=2, generations=1)
        c["space"] = {k: [v] for k, v in params.items()}
        class Backend:
            def __init__(inner, config, root): pass
            def evaluate(inner, p, directory, role):
                return self.trial(role, p)
        with patch("paretopilot.search.AscendBackend", Backend), patch("paretopilot.search.inspect_environment", return_value={"ready": True}):
            run(c, self.root/"search")
            baseline_result = run(c, self.root/"baseline-only", baseline_only=True)
        result = read_json(self.root/"search/summary.json")
        self.assertEqual(result["primary_baseline"], "source_baseline")
        self.assertEqual(result["selected"], "candidate")
        agent = read_json(self.root/"search/agent/000.input.json")
        self.assertEqual(agent["baseline"]["operator_identity"]["Op Type"], "BatchMatMulV3")
        self.assertEqual(agent["stock_baseline"]["operator_identity"]["Op Type"], "BatchMatMulV2")
        self.assertEqual(baseline_result["baseline_us"], 10)
        self.assertEqual(baseline_result["stock_reference_us"], 7)
        self.assertIn("Primary optimization baseline: default BatchMatMulV3", (self.root/"search/report.md").read_text(encoding="utf-8"))
        self.assertEqual(script.verify_result(self.root/"search", True)["selected"], "candidate")

    def test_default_script_covers_all_three_source_workloads(self):
        args = script.parser().parse_args(["--cann-path", "/opt/cann", "--soc", "Ascend910B2", "--operator", "matmul", "--matmul-case", "all", "--scope", "smoke", "--deps", "check"])
        self.assertEqual(args.matmul_implementation, "batch-v3")
        configs = script.prepare_configs(args, self.root)
        self.assertEqual(len(configs), 6)
        for path in configs.values():
            self.assertTrue(official.batch_enabled(read_json(path)))
        for name, path in configs.items():
            if name.endswith("-smoke"):
                self.assertEqual(read_json(path)["space"], {k: [v] for k, v in official.BATCH_DEFAULT_PARAMS.items()})
        session = script.Session(self.root, args)
        stages = []
        session.run = lambda name, command: stages.append(name)
        with patch.object(script, "verify_result", return_value={"status": "passed"}):
            script.execute_suite(session, args, "python3", configs)
        self.assertEqual(stages[-3:], ["matmul-smoke", "matmul-shared-rhs-smoke", "matmul-batched-rhs-smoke"])
        self.assertFalse(any(name.endswith("-baseline") for name in stages))

    def test_generated_batch_adapter_and_architecture_are_checked(self):
        dynamic = self.root/"build/tbe/dynamic"
        dynamic.mkdir(parents=True)
        (dynamic/"mat_mul_v3.py").write_text("parent adapter")
        with self.assertRaises(TuningError):
            official.generated_implementations(self.root/"build", batch=True)
        adapter = dynamic/"batch_mat_mul_v3.py"
        adapter.write_text('ascendc_src_file = "batch_mat_mul_v3_apt.cpp"')
        with self.assertRaises(TuningError):
            official.generated_implementations(self.root/"build", batch=True)
        adapter.write_text('ascendc_src_file = "batch_mat_mul_v3.cpp"')
        self.assertEqual(len(official.generated_implementations(self.root/"build", batch=True)), 2)

    def test_batch_build_generates_and_installs_both_adapters_for_910b(self):
        relative = Path("built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_comm_legacy.so")
        legacy = self.root/"system-opp"/relative
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"fixture legacy library")
        project = self.root/"project"
        def materialize(path, params, **kwargs):
            self.assertTrue(kwargs["batch"])
            path.mkdir()
            return {"operator": "BatchMatMulV3", "commit": official.COMMIT}
        commands = []
        def command(argv, cwd, log, timeout, env):
            commands.append(argv)
            if "ascendc_impl_gen" in argv:
                dynamic = project/"build/tbe/dynamic"
                dynamic.mkdir(parents=True)
                (dynamic/"mat_mul_v3.py").write_text("parent adapter")
                (dynamic/"batch_mat_mul_v3.py").write_text('ascendc_src_file = "batch_mat_mul_v3.cpp"')
            if "package" in argv:
                out = project/"build_out"
                out.mkdir()
                (out/"custom.run").write_text("package fixture")
                (project/"build/libparetopilot_onnx.so").write_bytes(b"parser fixture")
            if argv[0] == "bash":
                dynamic = self.root/"opp/vendors/paretopilot_nn/op_impl/ai_core/tbe/paretopilot_nn_impl/dynamic"
                dynamic.mkdir(parents=True)
                for path in (project/"build/tbe/dynamic").glob("*.py"):
                    shutil.copyfile(path, dynamic/path.name)
        with patch.object(official, "materialize", side_effect=materialize), \
             patch.object(official, "run_command", side_effect=command), \
             patch.object(official.platform, "machine", return_value="aarch64"):
            result = official._build_official(self.c, official.BATCH_DEFAULT_PARAMS.copy(), self.root, {"ASCEND_OPP_PATH": str(self.root/"system-opp")})
        self.assertIn("-DASCEND_COMPUTE_UNIT=ascend910b", commands[0])
        self.assertIn("-DASCEND_OP_NAME=batch_mat_mul_v3", commands[0])
        self.assertIn("-DASCEND_COMPILE_OPS=mat_mul_v3;batch_mat_mul_v3;matmul.common", commands[0])
        self.assertEqual(result["op_type"], "BatchMatMulV3")
        self.assertEqual(len(result["python_implementations"]), 2)

    def test_invalid_batch_strategy_rejected(self):
        c = deepcopy(self.c)
        c["space"]["batch_strategy"] = [2]
        write_json(self.root/"bad.json", c)
        with self.assertRaises(TuningError):
            load_config(self.root/"bad.json")

    def test_source_package_reused_but_each_trial_compiles_and_measures(self):
        backend = object.__new__(AscendBackend)
        backend.c, backend.root, backend.case = self.c, self.root, self.root/"case"
        backend.source_builds = {}
        backend.case.mkdir()
        inputs, output = shapes(self.c["operator"])
        write_json(backend.case/"case.json", {"inputs": [{"name": n, "shape": s} for n, s in zip(("a", "b"), inputs)], "output_shape": output})
        backend.candidate_graph = lambda p: p.write_bytes(b"mock ONNX")
        params = official.BATCH_DEFAULT_PARAMS.copy()
        calls = []
        def build(c, p, directory, env):
            opp = directory/"opp"
            opp.mkdir()
            (opp/"kernel.so").write_bytes(b"immutable package")
            return {"opp_path": str(opp), "source_dir": str(directory/"project"),
                    "installed_sha256": tree_hash(opp), "operator": "BatchMatMulV3", "commit": official.COMMIT,
                    "parameter_binding": "atc_process_environment"}
        def command(argv, directory, name, env):
            calls.append(name)
            if name == "atc":
                self.assertEqual(env["PARETOPILOT_REQUEST_HASH"], digest(params))
                self.assertEqual(env["PARETOPILOT_CORE_LIMIT"], "0")
                self.assertTrue(Path(env["ASCEND_CACHE_PATH"]).is_relative_to(directory))
                (directory/"model.om").write_bytes(b"mock OM")
                record = self.trial("source_baseline")["selection"]["effective_tiling"][0]
                (directory/"atc.log").write_text(f"PARETOPILOT_PARAMS={digest(params)}\nPARETOPILOT_OFFICIAL={official.COMMIT}\nPARETOPILOT_BATCH_TILING="+json.dumps(record))
            elif name == "runtime":
                write_json(directory/"runtime.json", {"om_sha256": file_hash(directory/"model.om"), "correctness": {"passed": True}})
            elif name == "profiler_export":
                (directory/"profiler").mkdir()
                (directory/"profiler/op_summary.csv").write_text("Op Name,OP Type,Task Duration(us)\n"+"target,BatchMatMulV3,8\n"*20)
        backend.command = command
        with patch("paretopilot.backend.build_candidate", side_effect=build) as builder:
            for name, role in (("source", "source_baseline"), ("candidate", "candidate")):
                directory = self.root/name
                directory.mkdir()
                trial = backend.evaluate(params, directory, role)
                self.assertEqual(len(trial["samples_us"]), 20)
            self.assertEqual(builder.call_count, 1)
        self.assertEqual(calls.count("atc"), 2)
        self.assertEqual(calls.count("runtime"), 2)
        self.assertTrue((self.root/"candidate/reused_build.json").is_file())
        (self.root/"source/opp/kernel.so").write_bytes(b"tampered")
        directory = self.root/"tampered"
        directory.mkdir()
        with self.assertRaisesRegex(TuningError, "package changed"):
            backend.evaluate(params, directory, "candidate")


if __name__ == "__main__":
    unittest.main()
