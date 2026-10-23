import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.util import TuningError, read_json, write_json
from scripts import inspect_cann_operator as script


class InspectCannOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def baseline(self):
        baseline = self.root/"baseline"
        baseline.mkdir()
        (baseline/"model.onnx").write_bytes(b"original model")
        write_json(baseline/"atc.command.json", {"cwd": str(baseline), "argv": ["atc", "--framework=5", "--model=model.onnx",
                   "--output=old-model", "--soc_version=Ascend910B2", "--input_shape=a:256,256;b:256,256", "--log=info"]})
        write_json(baseline/"trial.json", {"role": "baseline", "status": "ok", "duration_us": 7.98})
        return baseline

    def test_replay_preserves_shape_soc_and_input_without_overwriting_baseline(self):
        baseline = self.baseline()
        original = (baseline/"atc.command.json").read_bytes()
        command, model = script.replay_command(baseline, self.root/"diagnostic")
        self.assertIn("--soc_version=Ascend910B2", command)
        self.assertIn("--input_shape=a:256,256;b:256,256", command)
        self.assertIn("--model="+str(baseline/"model.onnx"), command)
        self.assertIn("--output="+str(self.root/"diagnostic/model"), command)
        self.assertEqual([v for v in command if v.startswith("--log=")], ["--log=debug"])
        self.assertEqual(model["size"], len(b"original model"))
        self.assertEqual((baseline/"atc.command.json").read_bytes(), original)
        write_json(baseline/"atc.command.json", {"argv": ["bash", "--framework=5", "--model=model.onnx"]})
        with self.assertRaisesRegex(TuningError, "recorded ONNX ATC"):
            script.replay_command(baseline, self.root/"diagnostic")

    def test_registration_exact_match_and_python_inspection_never_imports_module(self):
        config = self.root/"aic-ascend910b-ops-info.json"
        write_json(config, {"BatchMatMulV2": {"opFile": {"value": "batch_matmul_v2"}, "opInterface": {"value": "batch_matmul_v2"}},
                            "WeightQuantBatchMatmulV2": {"not_ours": True}})
        source = self.root/"batch_matmul_v2.py"
        source.write_text("raise RuntimeError('MUST NOT EXECUTE')\nfrom impl import helper\ndef build_op(x, impl_mode='high_performance'):\n    helper.auto_schedule(x)\n")
        warnings = []
        report = script.inspect_installation([self.root, self.root], warnings)
        self.assertFalse(warnings)
        self.assertEqual(len(report["registrations"]), 1)
        self.assertNotIn("not_ours", report["registrations"][0]["entry"])
        self.assertEqual(len(report["candidate_files"]), 1)
        details = report["candidate_files"][0]["python"]
        self.assertEqual(details["imports"][0]["statement"], "from impl import helper")
        self.assertIn("impl_mode='high_performance'", details["functions"][0]["arguments"])

    def test_ini_registration_preserves_field_names(self):
        config = self.root/"ops-info.ini"
        config.write_text("[BatchMatMulV2]\nopFile.value=batch_matmul_v2\n[Other]\nopFile.value=other\n")
        self.assertEqual(script.registration(config), {"opFile.value": "batch_matmul_v2"})

    def test_symlink_cycle_is_bounded_and_aliases_are_deduplicated(self):
        (self.root/"one.py").write_text("pass")
        try:
            (self.root/"cycle").symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        warnings = []
        self.assertEqual(list(script.installed_files([self.root], warnings)), [(self.root/"one.py").resolve()])

    def test_trace_does_not_treat_failed_searches_as_dependencies(self):
        trace = self.root/"trace.log"
        trace.write_text('10 openat(AT_FDCWD, "/missing.py", O_RDONLY) = -1 ENOENT\n'
                         '10 openat(AT_FDCWD, "/opp/entry.py", O_RDONLY) = 3</opp/entry.py>\n'
                         '11 execve("/bin/compiler", ["compiler"], 0x123) = 0\n'
                         '11 execve("/missing/compiler", [], 0x123) = -1 ENOENT\n')
        self.assertEqual(script.traced_files(trace), ["/bin/compiler", "/opp/entry.py"])

    def test_diagnostic_environment_is_private_and_keeps_system_opp(self):
        original = {"ASCEND_CUSTOM_OPP_PATH": "old-trial/vendor", "ASCEND_OPP_PATH": "system/opp",
                    "ASCEND_CACHE_PATH": "old-cache", "ASCEND_MODULE_LOG_LEVEL": "TBE=3"}
        env = script.diagnostic_environment(original, self.root)
        self.assertEqual(original["ASCEND_CACHE_PATH"], "old-cache")
        self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", env)
        self.assertNotIn("ASCEND_MODULE_LOG_LEVEL", env)
        self.assertEqual(env["ASCEND_OPP_PATH"], "system/opp")
        self.assertEqual(env["DUMP_GE_GRAPH"], "2")
        self.assertEqual(env["ASCEND_CACHE_PATH"], str(self.root/"cache"))

    def test_replay_failure_still_writes_evidence_and_does_not_claim_trace(self):
        baseline = self.baseline()
        cann = self.root/"cann"
        cann.mkdir()
        executable = cann/"atc"
        executable.write_bytes(b"ATC fixture")
        output = self.root/"diagnostic"
        def compile_command(argv, cwd, log, timeout, env):
            self.assertEqual(cwd, output)
            self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", env)
            log.write_text("BatchMatMulV2 compile failed\n")
            raise TuningError("ATC fixture failed")
        with patch.object(script.sys, "platform", "linux"), \
             patch.object(script, "resolve_cann", return_value=(cann, cann/"set_env.sh", {})), \
             patch.object(script, "source_environment", return_value={"PATH": str(cann), "EXAMPLE_SECRET": "not-logged"}), \
             patch.object(script.shutil, "which", side_effect=lambda name, **kwargs: str(executable) if name == "atc" else None), \
             patch.object(script, "run_command", side_effect=compile_command):
            code = script.main(["--cann-path", str(cann), "--baseline-dir", str(baseline), "--output", str(output), "--no-strace"])
        self.assertEqual(code, 2)
        report = read_json(output/"report.json")
        self.assertEqual(report["atc_status"], "failed")
        self.assertEqual(report["trace_status"], "disabled")
        self.assertNotIn("EXAMPLE_SECRET", json.dumps(report))
        self.assertIn("BatchMatMulV2 compile failed", (output/"log-matches.txt").read_text())
        self.assertTrue((output/"report.txt").is_file())

    def test_unavailable_ptrace_falls_back_to_one_atc_replay_with_explicit_status(self):
        baseline = self.baseline()
        cann = self.root/"cann"
        cann.mkdir()
        executable = cann/"atc"
        executable.write_bytes(b"ATC fixture")
        output = self.root/"diagnostic"
        calls = []
        def compile_command(argv, cwd, log, timeout, env):
            calls.append(argv)
            if argv[0] == "strace":
                log.write_text("ptrace: Operation not permitted")
                raise TuningError("ptrace denied")
            log.write_text("BatchMatMulV2 compiled\n")
            (cwd/"model.om").write_bytes(b"new diagnostic model")
            return {"argv": argv}
        with patch.object(script.sys, "platform", "linux"), \
             patch.object(script, "resolve_cann", return_value=(cann, cann/"set_env.sh", {})), \
             patch.object(script, "source_environment", return_value={"PATH": str(cann)}), \
             patch.object(script.shutil, "which", side_effect=lambda name, **kwargs: {"atc": str(executable), "strace": "strace"}.get(name)), \
             patch.object(script, "run_command", side_effect=compile_command):
            code = script.main(["--cann-path", str(cann), "--baseline-dir", str(baseline), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], str(executable))
        report = read_json(output/"report.json")
        self.assertEqual(report["atc_status"], "succeeded")
        self.assertEqual(report["trace_status"], "probe_failed")
        self.assertEqual(report["opened_candidate_files"], [])
        self.assertIn("ptrace denied", report["warnings"][0])


if __name__ == "__main__":
    unittest.main()
