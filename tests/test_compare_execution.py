import csv
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paretopilot.compare_execution import compare, diagnose, direct_profile
from paretopilot.config import load_config
from paretopilot.direct_worker import profile_measurements
from paretopilot.util import TuningError, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class ExecutionCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        self.c["measurement"].update(repeats=3, warmup=2)

    def csv(self, directory=None, rows=None):
        directory = directory or self.root
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/"op_summary.csv"
        if rows is None:
            rows = [{"Op Name": "raw_entry", "OP Type": "unknown", "Task Type": "AI_CORE",
                     "Block Dim": "24", "Task Duration(us)": str(v), "Stream ID": "1"} for v in (7, 8, 9)]
            rows.append({**rows[0], "Task Type": "MEMCPY", "Task Duration(us)": "100"})
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def trial(self, us=250):
        return {"status": "ok", "duration_us": us, "samples_us": [us]*3,
                "selection": {"effective_tiling": [{"block_dim": 24, "tiling_key": 65536,
                                                       "base_m": 128, "cube_plan": {"raw": "aa", "step_n": 1}}]}}

    def test_direct_parser_isolates_core_tasks_without_guessing_op_name(self):
        path = self.csv()
        result = direct_profile([path], 3, 24)
        self.assertEqual(result["duration_us"], 8)
        self.assertEqual(result["operator_identity"]["Op Type"], "unknown")
        self.assertEqual(len(result["task_rows"]), 3)
        for paths, repeats, block in (([path, path], 3, 24), ([path], 4, 24), ([path], 3, 12)):
            with self.assertRaises(TuningError):
                direct_profile(paths, repeats, block)
        with path.open() as stream:
            rows = list(csv.DictReader(stream))[:3]
        rows[1]["Op Name"] = "other_core_kernel"
        with self.assertRaisesRegex(TuningError, "ambiguous"):
            direct_profile([self.csv(rows=rows)], 3, 24)
        for row in rows:
            del row["Task Type"]
        with self.assertRaisesRegex(TuningError, "Task Type"):
            direct_profile([self.csv(rows=rows)], 3, 24)

    def test_diagnosis_separates_event_overhead_and_real_task_regression(self):
        om, direct = self.trial(8), self.trial(250)
        event = diagnose(om, direct, {"duration_us": 9})
        self.assertTrue(event["effective_tiling_equal"])
        self.assertTrue(any("event_interval_exceeds" in x for x in event["findings"]))
        self.assertFalse(any("direct_kernel_task_slower" in x for x in event["findings"]))
        slow = diagnose(om, direct, {"duration_us": 240})
        self.assertTrue(any("direct_kernel_task_slower" in x for x in slow["findings"]))
        direct["selection"]["effective_tiling"][0]["cube_plan"]["raw"] = "bb"
        self.assertTrue(diagnose(om, direct, {"duration_us": 8})["effective_tiling_equal"])
        direct["selection"]["effective_tiling"][0]["cube_plan"]["step_n"] = 2
        self.assertFalse(diagnose(om, direct, {"duration_us": 8})["effective_tiling_equal"])

    def test_profiler_scope_stops_and_destroys_after_failure(self):
        calls = []
        prof = SimpleNamespace(create_config=lambda *a: calls.append("create") or 1,
            start=lambda h: calls.append("start") or 0, stop=lambda h: calls.append("stop") or 0,
            destroy_config=lambda h: calls.append("destroy") or 0)
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            with profile_measurements(prof, self.c):
                calls.append("launch")
                raise RuntimeError("launch failed")
        self.assertEqual(calls, ["create", "start", "launch", "stop", "destroy"])
        calls.clear()
        prof.start = lambda h: calls.append("start") or 7
        with self.assertRaisesRegex(TuningError, "start"):
            with profile_measurements(prof, self.c):
                self.fail("must not launch after profiler start failed")
        self.assertEqual(calls, ["create", "start", "destroy"])

    def test_comparison_uses_same_inputs_and_parameters_and_preserves_partial_failure(self):
        for suffix in ("", "_shared_rhs", "_batched_rhs"):
            c = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            c["measurement"].update(repeats=3, warmup=2)
            root = self.root/(suffix or "2d")
            backend = Mock()
            backend.case = root/"case"
            backend.build = {"opp_path": str(root/"opp")}
            direct, om = self.trial(250), self.trial(8)
            calls = []
            def evaluate(which, config, params, directory, role):
                directory.mkdir()
                calls.append((which, deepcopy(params), config["operator"]))
                return direct if which is backend else om
            def worker(argv, cwd, log, timeout, env):
                req = read_json(argv[-1])
                self.assertEqual(req["case"], str(backend.case))
                self.assertEqual(req["one_shot"]["params"], calls[0][1])
                self.assertEqual(req["config"]["measurement"]["repeats"], 3)
                write_json(root/"kernel_profile/direct_result.json", self.trial(260))
                return {"argv": argv}
            def export(argv, directory, name, env):
                self.csv(directory/"profiler")
            backend.command.side_effect = export
            with patch("paretopilot.compare_execution.KernelBackend", return_value=backend), \
                    patch("paretopilot.compare_execution.evaluate", side_effect=evaluate), \
                    patch("paretopilot.compare_execution.run_command", side_effect=worker), \
                    patch("paretopilot.compare_execution.validate_result") as validate:
                result = compare(c, root)
            self.assertEqual(result["metrics"]["kernel_task_p50_us"], 8)
            self.assertEqual(result["metrics"]["kernel_event_p50_us"], 250)
            self.assertEqual(calls[0][1:], calls[1][1:])
            self.assertEqual(calls[1][0], backend.om_backend)
            validate.assert_called_once()
            self.assertTrue((root/"comparison.md").is_file())
        root = self.root/"failure"
        with patch("paretopilot.compare_execution.KernelBackend", return_value=backend), \
                patch("paretopilot.compare_execution.evaluate", return_value={"status": "failed", "error": "bad output"}):
            with self.assertRaisesRegex(TuningError, "bad output"):
                compare(self.c, root)
        self.assertEqual(read_json(root/"comparison.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
