from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

from paretopilot.agent import DummyAgent, validate_proposal
from paretopilot.cli import main
from paretopilot.config import load_config
from paretopilot.profiling import read_profile
from paretopilot.roofline import analyze
from paretopilot.search import confirmation, run
from paretopilot.space import enumerate_space, legal
from paretopilot.util import TuningError, digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/simulation_matmul.json")

    def config_error(self, change):
        change(self.c)
        write_json(self.root/"config.json", self.c)
        with self.assertRaises(TuningError):
            load_config(self.root/"config.json")

    def test_rejects_bad_dimensions(self):
        self.config_error(lambda c: c["operator"].update(m=True))

    def test_rejects_fractional_core_count(self):
        self.config_error(lambda c: c["hardware"].update(max_blocks=3.5))

    def test_real_matmul_does_not_claim_double_buffer_control(self):
        self.c = load_config(ROOT/"examples/ascend910b_matmul.json")
        self.config_error(lambda c: c["space"].update(buffers=[1, 2]))

    def test_atc_identity_cannot_be_overridden(self):
        self.c = load_config(ROOT/"examples/ascend910b_matmul.json")
        self.config_error(lambda c: c["ascend"].update(atc_extra=["--model=other.onnx"]))

    def test_roofline_distinguishes_measured_bytes(self):
        r = analyze(self.c)
        self.assertEqual(r["flops"], 2*256**3)
        self.assertEqual(r["minimum_bytes"], 3*256*256*2)
        measured = analyze(self.c, 10, r["minimum_bytes"]*2)
        self.assertEqual(measured["arithmetic_intensity"], r["arithmetic_intensity"]/2)
        self.assertEqual(measured["traffic_basis"], "profiling")

    def test_dummy_never_changes_domain_from_roofline(self):
        context = {"original_space": self.c["space"], "roofline": {"suggested_priority": ["tile_k"]}}
        proposal = DummyAgent().propose(context)
        self.assertEqual(proposal["space"], self.c["space"])
        self.assertEqual(proposal["priority"], [])
        proposal["space"]["tile_k"].append(999)
        self.assertNotIn(999, self.c["space"]["tile_k"])

    def test_bad_plugin_domains_rejected(self):
        for value in (True, 32.0, {"tile":32}, 999):
            proposal = {"space": deepcopy(self.c["space"])}
            proposal["space"]["tile_k"] = [value]
            with self.subTest(value=value), self.assertRaises(TuningError):
                validate_proposal(proposal, self.c["space"])

    def profile(self, records):
        path = self.root/"op_summary.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, ["Op Name", "Op Type", "Task Duration(us)"])
            writer.writeheader()
            writer.writerows(records)
        return path

    def test_profile_median_and_exact_matching(self):
        records = [{"Op Name":"target", "Op Type":"MatMulV2", "Task Duration(us)":v} for v in (1, 5, 3)]
        records.append({"Op Name":"noise", "Op Type":"MatMulV2Extra", "Task Duration(us)":100})
        p = self.profile(records)
        self.assertEqual(read_profile([p], {"Op Type":"MatMulV2"}, 3)["duration_us"], 3)
        with self.assertRaises(TuningError):
            read_profile([p, p], {"Op Type":"MatMulV2"}, 3)

    def test_profile_rejects_nan_and_mixed_nodes(self):
        for values, names in (((1,"nan"), ("target","target")), ((1,2), ("first","second"))):
            p = self.profile([{"Op Name":n,"Op Type":"Add","Task Duration(us)":v} for n,v in zip(names,values)])
            with self.assertRaises(TuningError):
                read_profile([p], {"Op Type":"Add"}, 2)

    def test_integration_deterministic_budget_and_artifacts(self):
        results = []
        for name in ("first", "second"):
            result = run(self.c, self.root/name)
            self.assertEqual(result["evaluations"], self.c["search"]["budget"])
            summary = read_json(self.root/name/"summary.json")
            params = [t["params"] for t in summary["history"]]
            self.assertEqual(len({digest(p) for p in params}), len(params))
            self.assertTrue(all(legal(p,self.c) for p in params))
            self.assertTrue(all(t["correctness"]["simulated"] for t in summary["history"]))
            self.assertEqual([t["role"] for t in summary["confirmation"]["trials"]], ["baseline","candidate","candidate","baseline","baseline","candidate"])
            self.assertIn("synthetic", (self.root/name/"report.html").read_text())
            self.assertTrue((self.root/name/"roofline.svg").is_file())
            results.append(params)
        self.assertEqual(*results)

    def test_existing_output_is_not_overwritten(self):
        write_json(self.root/"old.json", {"keep":True})
        with self.assertRaises(TuningError):
            run(self.c, self.root)
        self.assertEqual(read_json(self.root/"old.json"), {"keep":True})

    def test_failed_candidates_count_budget_and_keep_baseline(self):
        with patch("paretopilot.backend.SimulationBackend.evaluate") as evaluate:
            def trial(params, directory, role):
                if role == "candidate":
                    raise TuningError("intentional compile failure")
                return {"status":"ok","duration_us":100,"measured_bytes":None,"mode":"simulation"}
            evaluate.side_effect = trial
            run(self.c, self.root/"failed")
        result = read_json(self.root/"failed/summary.json")
        self.assertEqual(result["selected"], "baseline")
        self.assertIsNone(result["selected_params"])
        self.assertEqual(len(result["history"]), self.c["search"]["budget"])
        self.assertTrue(all(t["status"]=="failed" for t in result["history"]))

    def test_confirmation_rejects_small_gain_and_noise(self):
        for b,c in (([100]*3,[99]*3), ([80,100,120],[70,90,110])):
            calls = iter([("baseline",b[0]),("candidate",c[0]),("candidate",c[1]),("baseline",b[1]),("baseline",b[2]),("candidate",c[2])])
            def evaluate(*args):
                role,value = next(calls)
                return {"role":role,"duration_us":value,"status":"ok"}
            with patch("paretopilot.search.evaluate", side_effect=evaluate):
                self.assertFalse(confirmation(self.c, {"params":{}}, None, self.root)["accepted"])

    def test_cli_smoke_and_baseline(self):
        for command in ("run", "baseline"):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main([command,"--config",str(ROOT/"examples/simulation_add.json"),"--output",str(self.root/command),"--smoke"])
            self.assertEqual(code,0)
        self.assertEqual(len(read_json(self.root/"run/history.json")),1)


if __name__ == "__main__":
    unittest.main()
