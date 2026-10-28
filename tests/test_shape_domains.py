from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.config import load_config
from paretopilot.backend import AscendBackend
from paretopilot import domains, tuning_cache, official
from paretopilot.search import run
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, read_json, write_json, file_hash
from paretopilot.workload import shapes

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = {"status": "ok", "runtime_soc": "Ascend910B2", "compile_soc": "Ascend910B2", "core_count": 24,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "L1_SIZE": 524288}


class ShapeDomainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        self.c["implementation"]["execution"] = "kernel"
        self.c["search"].update(result_cache_dir=str(self.root/"results"), budget=3, population=2, generations=3)
        self.c["search"]["runtime_signature"] = {"driver_version_lines": ["Version: fixture"]}

    def configured(self, c=None):
        return domains.generate(c or self.c, PLATFORM)[0]

    def source(self):
        return {"status": "ok", "duration_us": 10, "samples_us": [10]*20,
                "correctness": {"passed": True},
                "selection": {"installed_sha256": "verified-fixture", "cache_identity": {"version": "8.5.0"},
                              "platform_info": deepcopy(PLATFORM)}}

    def test_shape_changes_regenerate_domains_and_real_resources_filter_tiles(self):
        c = self.configured()
        self.assertIn(4, c["space"]["core_limit"])
        small = deepcopy(self.c)
        small["operator"].update(m=1, n=100, batch=32, batch_mode="batched_rhs")
        changed, evidence = domains.generate(small, PLATFORM)
        self.assertEqual(changed["space"]["base_m"], [0, 16])
        self.assertIn(112, changed["space"]["base_n"])
        self.assertTrue(any(p["base_n"] == 112 for p in enumerate_space(changed)))
        limited = {**PLATFORM, "core_count": 8, "L0C_SIZE": 4096}
        constrained = domains.generate(self.c, limited)[0]
        self.assertTrue(all(p["core_limit"] <= 8 and (not p["base_m"] or p["base_m"]*p["base_n"]*4 <= 4096)
                            for p in enumerate_space(constrained)))
        self.assertEqual(self.c["hardware"]["max_blocks"], 24)
        with self.assertRaises(TuningError):
            domains.generate(self.c, {**PLATFORM, "L1_SIZE": 0})

    def test_cache_exact_nearby_and_environment_mismatch(self):
        c, source = self.configured(), self.source()
        params = {"base_m": 32, "base_n": 64, "core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        winner = {"status": "ok", "params": params, "duration_us": 6}
        key = tuning_cache.save(c, source, winner, {"tuning_accepted": True}, self.root)
        record, match = tuning_cache.lookup(c, source)
        self.assertEqual(match["status"], "exact")
        self.assertEqual(record["key"], key)
        nearby = deepcopy(self.c)
        nearby["operator"]["m"] = 128
        self.assertEqual(tuning_cache.lookup(self.configured(nearby), source)[1]["status"], "nearby")
        tiny = deepcopy(self.c)
        tiny["operator"]["m"] = 1
        self.assertEqual(tuning_cache.lookup(self.configured(tiny), source)[1]["status"], "miss")
        altered = deepcopy(source)
        altered["selection"]["installed_sha256"] = "different build"
        self.assertEqual(tuning_cache.lookup(c, altered)[1]["status"], "miss")
        altered_config = deepcopy(c)
        altered_config["ascend"]["device"] = 1
        self.assertEqual(tuning_cache.lookup(altered_config, source)[1]["status"], "miss")
        (self.root/"results"/(key+".json")).write_text("corrupt", encoding="utf8")
        self.assertEqual(tuning_cache.lookup(c, source)[1]["status"], "miss")

    def test_om_probe_supplies_real_limits_and_manual_mode_does_not_probe(self):
        manifest = {"config": self.c}
        def probe(argv, cwd, log, timeout, env):
            request = read_json(argv[-1])
            write_json(Path(request["directory"])/"platform_init.json", PLATFORM)
            self.assertEqual(argv[1:3], ["-m", "paretopilot.domains"])
        with patch("paretopilot.domains.run_command", side_effect=probe) as command:
            generated = domains.prepare(self.c, {}, self.root, manifest)
        self.assertEqual(command.call_count, 1)
        self.assertEqual(generated["search"]["domain_limits"]["L0C_SIZE"], PLATFORM["L0C_SIZE"])
        self.assertEqual(read_json(self.root/"effective_config.json"), generated)
        manual = deepcopy(self.c)
        manual["search"]["domain_mode"] = "manual"
        with patch("paretopilot.domains.run_command") as command:
            self.assertEqual(domains.prepare(manual, {}, self.root, {}), manual)
            command.assert_not_called()

    def test_cache_requires_build_evidence_and_honors_fresh_search(self):
        c = self.configured()
        self.assertEqual(tuning_cache.lookup(c, {})[1]["status"], "unavailable")
        c["search"]["reuse_results"] = False
        self.assertEqual(tuning_cache.lookup(c, self.source())[1]["status"], "disabled")

    def test_shape_mode_discovers_stock_operator_and_detects_confirmation_changes(self):
        backend = object.__new__(AscendBackend)
        backend.c = self.c
        backend.case = self.root/"case"
        backend.case.mkdir()
        (backend.case/"baseline.onnx").write_bytes(b"fixture graph")
        write_json(backend.case/"case.json", {"inputs": [{"name": "a", "shape": [256, 256]},
                                                       {"name": "b", "shape": [256, 256]}], "output_shape": [256, 256]})
        op_type = "MatMulV2"
        def command(argv, directory, name, env):
            if name == "atc":
                (directory/"model.om").write_bytes(b"fixture OM")
            if name == "runtime":
                write_json(directory/"runtime.json", {"om_sha256": file_hash(directory/"model.om"), "correctness": {"passed": True}})
            if name == "profiler_export":
                p = directory/"profiler/op_summary.csv"
                p.parent.mkdir()
                p.write_text("Op Name,OP Type,Task Duration(us)\n"+f"/MatMul,{op_type},7.0\n"*20)
        backend.command = command
        first = self.root/"first-reference"
        first.mkdir()
        trial = backend.evaluate({}, first, "baseline")
        self.assertEqual(trial["operator_identity"]["Op Type"], "MatMulV2")
        op_type = "BatchMatMulV2"
        second = self.root/"second-reference"
        second.mkdir()
        with self.assertRaisesRegex(TuningError, "changed during confirmation"):
            backend.evaluate({}, second, "baseline")

    def test_exact_retest_must_be_correct_fast_stable_and_previously_confirmed(self):
        c, source = self.configured(), self.source()
        record = {"confirmed": True, "source_us": 10, "duration_us": 6}
        trial = {"status": "ok", "duration_us": 6, "samples_us": [6]*20, "correctness": {"passed": True}}
        self.assertTrue(tuning_cache.stable_retest(c, source, trial, record, {"status": "exact"}))
        for match, rec, t in (({"status": "nearby"}, record, trial),
                              ({"status": "exact"}, {**record, "confirmed": False}, trial),
                              ({"status": "exact"}, record, {"status": "failed"}),
                              ({"status": "exact"}, record, {**trial, "duration_us": 9.9}),
                              ({"status": "exact"}, record, {**trial, "samples_us": [1, 12]})):
            self.assertFalse(tuning_cache.stable_retest(c, source, t, rec, match))

    def test_event_cache_cannot_seed_or_short_circuit_task_profiler_search(self):
        c, source = self.configured(), self.source()
        source["timing_method"] = "device_event_us"
        params = {"base_m": 32, "base_n": 64, "core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}
        winner = {"status": "ok", "params": params, "duration_us": 6, "samples_us": [6]*20,
                  "correctness": {"passed": True}, "timing_method": "device_event_us"}
        key = tuning_cache.save(c, source, winner, {"tuning_accepted": True}, self.root)
        record = read_json(self.root/"results"/(key+".json"))
        source["timing_method"] = "kernel_task_duration_us"
        self.assertEqual(tuning_cache.lookup(c, source)[1]["status"], "miss")
        self.assertIsNone(tuning_cache.save(c, source, winner, {}, self.root))
        winner["timing_method"] = "kernel_task_duration_us"
        self.assertFalse(tuning_cache.stable_retest(c, source, winner, record, {"status": "exact"}))
        tuning_cache.save(c, source, winner, {"tuning_accepted": True}, self.root)
        new, match = tuning_cache.lookup(c, source)
        self.assertEqual(match["status"], "exact")
        self.assertTrue(tuning_cache.stable_retest(c, source, winner, new, match))

    def test_controller_retests_exact_hit_and_still_runs_fresh_om_confirmation(self):
        calls = []
        outer = self
        class Backend:
            candidate_duration = 6
            task_ranking_fixture = False
            def __init__(self, config, root):
                self.c = config
                self.om_backend = self.Om(self)
            class Om:
                def __init__(self, owner): self.owner = owner
                def evaluate(self, p, directory, role): return self.owner.measure(p, directory, role, "om")
            def evaluate(self, p, directory, role): return self.measure(p, directory, role, "kernel")
            def measure(self, p, directory, role, execution):
                calls.append((str(directory), role, execution, p))
                trial = outer.source()
                default = official.default_params(self.c)
                changed = p != default and role != "baseline"
                duration = self.candidate_duration if changed else 10
                if self.task_ranking_fixture and directory.parent.name == "trials":
                    duration = (20, 7, 9)[int(directory.name) % 3]
                inputs, output = shapes(self.c["operator"])
                record = {**p, "base_m": p.get("base_m") or (64 if changed else 128),
                          "base_n": p.get("base_n") or 128, "base_k": 64,
                          "used_core_num": 12 if changed else 24, "tiling_key": 65536,
                          "block_dim": 24, "multi_batch": 0, "load_mode": 0, "batch_count": 1}
                trial["selection"].update(commit=official.COMMIT, operator="BatchMatMulV3", effective_tiling=[record])
                trial.update(duration_us=duration, samples_us=[duration]*20, mode="ascend", execution=execution,
                    timing_method="kernel_task_duration_us" if execution == "kernel" else "task_duration_us",
                    measured_bytes=None, correctness={"passed": True}, input_shapes=inputs, output_shape=output,
                    operator_identity={"Op Type": "BatchMatMulV2" if role == "baseline" else "BatchMatMulV3"})
                trial["event_diagnostics"] = {"duration_us": 1000-duration*10, "timing_method": "device_event_us"}
                return trial
            def close(self): pass
        with patch("paretopilot.search.inspect_environment", return_value={"ready": True, "npu_smi": "Version: fixture"}), \
                patch("paretopilot.direct_backend.KernelBackend", Backend):
            run(self.c, self.root/"first")
            first = read_json(self.root/"first/summary.json")
            self.assertEqual(len(first["history"]), 3)
            self.assertEqual(first["agent"]["kind"], "dummy")
            calls.clear()
            run(self.c, self.root/"second")
            second = read_json(self.root/"second/summary.json")
            self.assertEqual(second["result_cache"]["status"], "exact")
            self.assertTrue(second["result_cache"]["search_skipped"])
            self.assertEqual(len(second["history"]), 1)
            confirmations = [v for v in calls if "confirmation" in v[0]]
            self.assertEqual(len(confirmations), 9)
            self.assertTrue(all(v[2] == "om" for v in confirmations))
            self.assertTrue((self.root/"second/domain.json").is_file())
            self.assertEqual(read_json(self.root/"second/agent/000.input.json")["original_space"],
                             read_json(self.root/"second/effective_config.json")["space"])
            fresh = deepcopy(self.c)
            fresh["search"]["reuse_results"] = False
            run(fresh, self.root/"fresh")
            self.assertEqual(len(read_json(self.root/"fresh/summary.json")["history"]), 3)
            Backend.candidate_duration = 9.9
            run(self.c, self.root/"regression")
            regression = read_json(self.root/"regression/summary.json")
            self.assertFalse(regression["result_cache"]["search_skipped"])
            self.assertEqual(len(regression["history"]), 3)
            calls.clear()
            run(self.c, self.root/"smoke", smoke=True)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(v[2] == "kernel" for v in calls))
            from scripts.test_910b import verify_result
            self.assertEqual(verify_result(self.root/"smoke", True)["status"], "passed")
            Backend.candidate_duration = 6
            Backend.task_ranking_fixture = True
            for suffix in ("", "_shared_rhs", "_batched_rhs"):
                case = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
                case["implementation"]["execution"] = "kernel"
                case["search"].update(budget=3, population=2, generations=3, result_cache=False)
                directory = self.root/("task-ranking"+suffix)
                calls.clear()
                run(case, directory)
                result = read_json(directory/"summary.json")
                self.assertEqual(result["winner"]["duration_us"], 7)
                self.assertEqual(result["winner"]["timing_method"], "kernel_task_duration_us")
                self.assertEqual(min(result["history"], key=lambda t: t["event_diagnostics"]["duration_us"])["duration_us"], 20)
                self.assertTrue(all(v[2] == "om" for v in calls if "confirmation" in v[0]))


if __name__ == "__main__":
    unittest.main()
