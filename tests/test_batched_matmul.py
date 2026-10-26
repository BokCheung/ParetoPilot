from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.backend import AscendBackend
from paretopilot.config import load_config
from paretopilot.roofline import analyze
from paretopilot.search import run
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, file_hash, read_json, write_json
from paretopilot.workload import shapes
from scripts import test_910b as script

ROOT = Path(__file__).resolve().parents[1]


class BatchedMatmulTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, mode="shared_rhs"):
        return load_config(ROOT/f"examples/ascend910b_matmul_{mode}.json")

    def test_default_shapes_and_roofline_count_shared_weights_once(self):
        for mode, inputs, output in (
            ("shared_rhs", [[32, 100, 128], [128, 256]], [32, 100, 256]),
            ("batched_rhs", [[32, 1, 128], [32, 128, 100]], [32, 1, 100]),
        ):
            c = self.config(mode)
            self.assertEqual(shapes(c["operator"]), (inputs, output))
            r = analyze(c)
            op = c["operator"]
            self.assertEqual(r["flops"], 2*32*op["m"]*op["n"]*128)
            weights = 128*op["n"]*(32 if mode == "batched_rhs" else 1)
            self.assertEqual(r["minimum_bytes"], 2*(32*op["m"]*128+weights+32*op["m"]*op["n"]))
            self.assertEqual(r["suggested_priority"], [])
            self.assertEqual(enumerate_space(c), [])
            self.assertEqual(c["agent"], {"kind": "dummy"})
        c = self.config()
        c["operator"]["batch"] = 1
        self.assertEqual(shapes(c["operator"])[0][0], [1, 100, 128])

    def test_invalid_batch_or_candidate_configuration_rejected(self):
        variants = []
        for batch in (0, -1, True, 1.5, 2**31):
            c = self.config(); c["operator"]["batch"] = batch; variants.append(c)
        for mode in ("unknown", None):
            c = self.config(); c["operator"]["batch_mode"] = mode; variants.append(c)
        c = self.config(); del c["operator"]["batch"]; variants.append(c)
        c = self.config(); c["implementation"] = {"kind": "cann_matmul_v3"}; variants.append(c)
        c = self.config(); c["space"] = {"tile_m": [16]}; variants.append(c)
        for c in variants:
            path = self.root/"invalid.json"
            write_json(path, c)
            with self.subTest(operator=c["operator"]), self.assertRaises(TuningError):
                load_config(path)

    def test_search_rejected_before_creating_output_or_loading_agent(self):
        with patch("paretopilot.search.load_agent") as agent, self.assertRaisesRegex(TuningError, "baseline command"):
            run(self.config(), self.root/"search")
        agent.assert_not_called()
        self.assertFalse((self.root/"search").exists())

    def baseline(self, c):
        inputs, output = shapes(c["operator"])
        return {"status": "ok", "mode": "ascend", "role": "baseline", "duration_us": 7.5,
                "correctness": {"passed": True}, "samples_us": [7.5]*c["measurement"]["repeats"],
                "input_shapes": inputs, "output_shape": output,
                "om_sha256": "test-om-hash", "csv_files": ["op_summary.csv"], "roofline": analyze(c, 7.5)}

    def test_baseline_pipeline_reports_without_candidate_or_agent(self):
        c = self.config()
        output = self.root/"baseline"
        with patch("paretopilot.search.inspect_environment", return_value={"ready": True}), \
             patch("paretopilot.search.AscendBackend"), \
             patch("paretopilot.search.evaluate", return_value=self.baseline(c)) as evaluate, \
             patch("paretopilot.search.load_agent") as agent:
            result = run(c, output, baseline_only=True)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.args[-1], "baseline")
        agent.assert_not_called()
        self.assertEqual(result["baseline_us"], 7.5)
        self.assertEqual(read_json(output/"manifest.json")["legal_candidates"], 0)
        self.assertIn("Candidate tuning and speedup were not evaluated", (output/"report.html").read_text())
        self.assertTrue((output/"roofline.svg").is_file())
        self.assertFalse(script.verify_baseline_result(output)["tuning_supported"])

    def test_3d_suite_runs_both_baselines_without_candidate_stages(self):
        for scope in ("smoke", "full"):
            output = self.root/scope
            args = script.parser().parse_args(["--cann-path", "/opt/cann", "--soc", "Ascend910B2",
                "--operator", "matmul", "--matmul-case", "3d", "--scope", scope, "--deps", "check"])
            configs = script.prepare_configs(args, output)
            names = ["matmul-shared-rhs", "matmul-batched-rhs"]
            self.assertEqual(set(configs), set(names))
            session = script.Session(output, args)
            commands = []
            def stage(name, command):
                commands.append((name, command))
                if name.endswith("-baseline"):
                    c = read_json(configs[name.removesuffix("-baseline")])
                    write_json(output/name/"manifest.json", {"config": c})
                    write_json(output/name/"baseline_summary.json", self.baseline(c))
                    (output/name/"report.html").write_text("baseline report")
            session.run = stage
            script.execute_suite(session, args, sys.executable, configs)
            self.assertEqual(set(session.data["results"]), {name+"-baseline" for name in names})
            self.assertFalse(any(name.endswith(("-smoke", "-full")) for name, _ in commands))
            for name, argv in commands:
                if name.endswith("-baseline"):
                    self.assertIn("baseline", argv)
            self.assertEqual(script.prepare_python(session, args), sys.executable)

    def test_baseline_verification_rejects_missing_evidence(self):
        c = self.config()
        write_json(self.root/"manifest.json", {"config": c})
        (self.root/"report.html").write_text("report")
        for key, value in (("input_shapes", [[1, 1]]), ("output_shape", []), ("samples_us", [7.5]),
                           ("correctness", {"passed": True, "simulated": True}), ("om_sha256", None)):
            trial = self.baseline(c)
            trial[key] = value
            write_json(self.root/"baseline_summary.json", trial)
            with self.subTest(key=key), self.assertRaises(TuningError):
                script.verify_baseline_result(self.root)

    def test_backend_uses_exact_3d_atc_shapes_and_profile_samples(self):
        for mode in ("shared_rhs", "batched_rhs"):
            c = self.config(mode)
            directory = self.root/mode
            directory.mkdir()
            case = directory/"case"
            case.mkdir()
            inputs, output = shapes(c["operator"])
            write_json(case/"case.json", {"inputs": [{"name": n, "shape": s} for n, s in zip(("a", "b"), inputs)], "output_shape": output})
            (case/"baseline.onnx").write_bytes(b"mock-onnx")
            backend = object.__new__(AscendBackend)
            backend.c, backend.root, backend.case = c, directory, case
            def command(argv, target, name, env):
                self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", env)
                if name == "atc":
                    rhs = "128,256" if mode == "shared_rhs" else "32,128,100"
                    lhs = "32,100,128" if mode == "shared_rhs" else "32,1,128"
                    self.assertIn("--input_shape=a:"+lhs+";b:"+rhs, argv)
                    self.assertIn("--soc_version=Ascend910B2", argv)
                    (target/"model.om").write_bytes(b"mock-om")
                elif name == "runtime":
                    write_json(target/"runtime.json", {"om_sha256": file_hash(target/"model.om"), "correctness": {"passed": True}})
                elif name == "profiler_export":
                    (target/"profiler").mkdir()
                    (target/"profiler/op_summary.csv").write_text("Op Name,OP Type,Task Duration(us)\n"+"/MatMul,BatchMatMulV2,7.5\n"*20)
            backend.command = command
            with patch("paretopilot.backend.build_candidate") as build:
                result = backend.evaluate({}, directory, "baseline")
            build.assert_not_called()
            self.assertEqual(len(result["samples_us"]), 20)
            self.assertEqual(result["input_shapes"], inputs)
            self.assertEqual(result["output_shape"], output)

    def test_all_cases_keep_existing_2d_smoke_configuration(self):
        args = script.parser().parse_args(["--cann-path", "/opt/cann", "--soc", "Ascend910B2", "--matmul-case", "all"])
        configs = script.prepare_configs(args, self.root)
        self.assertEqual(set(configs), {"add", "add-smoke", "matmul", "matmul-smoke", "matmul-shared-rhs", "matmul-batched-rhs"})
        self.assertEqual(read_json(configs["matmul-smoke"])["space"], {"core_limit": [0], "tiling_policy": [0]})


if __name__ == "__main__":
    unittest.main()
