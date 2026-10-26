import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.config import load_config
from paretopilot.profiling import discover_matmul
from paretopilot.roofline import analyze
from paretopilot.routing import route, check_identity
from paretopilot.search import run
from paretopilot.util import TuningError, read_json, write_json
from scripts import test_910b as script

ROOT = Path(__file__).resolve().parents[1]


class OperatorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_matmul.json")

    def csv(self, tasks):
        path = self.root/"op_summary.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Op Name", "OP Type", "Device_id", "Stream ID", "Task Duration(us)"])
            for name, kind, count in tasks:
                writer.writerows([[name, kind, "0", "46", "7.5"]]*count)
        return path

    def baseline(self, kind="BatchMatMulV2"):
        return {"status": "ok", "mode": "ascend", "role": "baseline", "duration_us": 7.5,
                "measured_bytes": None, "samples_us": [7.5]*20,
                "correctness": {"passed": True}, "operator_identity": {"Op Name": "/MatMul", "Op Type": kind},
                "input_shapes": [[256, 256], [256, 256]], "output_shape": [256, 256],
                "om_sha256": "test-om", "csv_files": ["op_summary.csv"], "roofline": analyze(self.c, 7.5)}

    def test_discovers_actual_operator_without_config_type_assumption(self):
        for kind in ("BatchMatMulV2", "MatMulV3", "MatMulV2"):
            path = self.csv([("/layout", "Transpose", 20), ("/MatMul", kind, 20)])
            result = discover_matmul([path], 20)
            self.assertEqual(result["operator_identity"]["Op Type"], kind)
            self.assertEqual(result["operator_identity"]["Op Name"], "/MatMul")
            self.assertEqual(result["duration_us"], 7.5)

    def test_ambiguity_missing_samples_unknown_types_and_duplicate_exports_rejected(self):
        for tasks in ([ ("/MatMul", "BatchMatMulV2", 19) ],
                      [ ("/x", "BatchMatMulV2", 20), ("/y", "MatMulV3", 20) ],
                      [ ("/x", "BatchMatMulV2", 10), ("/y", "BatchMatMulV2", 10) ],
                      [ ("/x", "UnknownKernel", 20) ]):
            path = self.csv(tasks)
            with self.subTest(tasks=tasks), self.assertRaises(TuningError):
                discover_matmul([path], 20)
        path = self.csv([("/MatMul", "BatchMatMulV2", 20)])
        with self.assertRaises(TuningError):
            discover_matmul([path, path], 20)

    def test_route_never_substitutes_v3_for_v2_or_3d(self):
        for kind in ("BatchMatMulV2", "MatMulV2", None):
            self.assertFalse(route(self.c, self.baseline(kind))["tuning_supported"])
        matched = route(self.c, self.baseline("MatMulV3"))
        self.assertTrue(matched["tuning_supported"])
        self.assertEqual(matched["adapter"], "cann_matmul_v3")
        three_d = load_config(ROOT/"examples/ascend910b_matmul_batched_rhs.json")
        self.assertFalse(route(three_d, self.baseline("MatMulV3"))["tuning_supported"])

    def test_mismatching_candidate_rejected(self):
        with self.assertRaises(TuningError):
            check_identity(self.c, self.baseline(), self.baseline("MatMulV3"))
        check_identity(self.c, self.baseline(), self.baseline())

    def test_unavailable_adapter_preserves_baseline_and_never_loads_agent_or_source(self):
        out = self.root/"run"
        with patch("paretopilot.search.inspect_environment", return_value={"ready": True}) as doctor, \
             patch("paretopilot.search.AscendBackend"), \
             patch("paretopilot.search.evaluate", return_value=self.baseline()) as evaluate, \
             patch("paretopilot.search.load_agent") as agent:
            result = run(self.c, out)
        self.assertFalse(result["tuning_supported"])
        self.assertEqual(evaluate.call_count, 1)
        agent.assert_not_called()
        self.assertEqual(doctor.call_args.args[0]["implementation"]["kind"], "baseline_only")
        routing = read_json(out/"routing.json")
        self.assertIn("--baseline-dir", routing["inspection_argv"])
        self.assertIn("BatchMatMulV2", (out/"report.html").read_text())
        self.assertEqual(script.verify_result(out, True)["kind"], "tuning_unavailable")
        self.assertFalse((out/"source_baseline").exists())

    def test_suite_stops_after_one_baseline_when_adapter_is_unavailable(self):
        args = script.parser().parse_args(["--cann-path", "/opt/cann", "--soc", "Ascend910B2", "--operator", "matmul", "--deps", "check", "--matmul-implementation", "legacy"])
        configs = script.prepare_configs(args, self.root)
        session = script.Session(self.root, args)
        stages = []
        def stage(name, argv):
            stages.append(name)
            if name == "matmul-baseline":
                out = self.root/name
                write_json(out/"baseline_summary.json", self.baseline())
                write_json(out/"manifest.json", {"config": self.c})
                write_json(out/"routing.json", route(self.c, self.baseline()))
                (out/"report.html").write_text("baseline report")
        session.run = stage
        script.execute_suite(session, args, sys.executable, configs)
        self.assertEqual(stages.count("matmul-baseline"), 1)
        self.assertNotIn("matmul-smoke", stages)
        self.assertNotIn("matmul-full", stages)
        self.assertFalse(session.data["results"]["matmul-baseline"]["tuning_supported"])


if __name__ == "__main__":
    unittest.main()
