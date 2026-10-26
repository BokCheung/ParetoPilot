import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from paretopilot.bmmv2_probe import comparison, verify_build_records, run_probe, bind_selected_kernels
from paretopilot.config import load_config
from paretopilot.roofline import analyze
from paretopilot.tbe_probe_hook import transform_source
from paretopilot.util import TuningError, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = b'''"""Installed wrapper fixture."""
from __future__ import annotations
class Tbe:
    @staticmethod
    def build(sch, config):
        return config["build_args"]
tbe = Tbe()
def _binary_constant_build(extra_params):
    config = {"name": "fixture_kernel", "build_args": {"enable_db_fold": True,
        "InjectSync": {"sync_opt_for_notail_db": 2}, "enable_cce_licm_safe_hoist": False}}
    return tbe.build("schedule", config)
'''


class Bmmv2ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/"batch_matmul_v2.py"
        self.source.write_bytes(SOURCE)
        self.env = {"PARETOPILOT_TBE_RECORD": str(self.root/"records.jsonl"),
                    "PARETOPILOT_TBE_POLICY": json.dumps({"enable_db_fold": False})}

    def execute(self, op="BatchMatMulV2"):
        namespace = {}
        with patch.dict(os.environ, self.env):
            exec(transform_source(SOURCE, str(self.source)), namespace)
            return namespace["_binary_constant_build"]({"op_type": op})

    def test_only_db_fold_changes_and_installed_file_is_untouched(self):
        original_hash = file_hash(self.source)
        result = self.execute()
        self.assertIs(result["enable_db_fold"], False)
        self.assertEqual(result["InjectSync"], {"sync_opt_for_notail_db": 2})
        self.assertIs(result["enable_cce_licm_safe_hoist"], False)
        evidence = verify_build_records(self.root/"records.jsonl", self.source, original_hash, False)
        self.assertEqual(evidence["build_calls"][0]["caller"], "_binary_constant_build")
        self.assertEqual(file_hash(self.source), original_hash)

    def test_related_operators_are_untouched_and_produce_no_build_evidence(self):
        self.assertIs(self.execute("CompressMatMul")["enable_db_fold"], True)
        self.assertFalse((self.root/"records.jsonl").exists())

    def test_default_probe_retains_true(self):
        self.env["PARETOPILOT_TBE_POLICY"] = json.dumps({"enable_db_fold": True})
        self.assertIs(self.execute()["enable_db_fold"], True)
        verify_build_records(self.root/"records.jsonl", self.source, file_hash(self.source), True)

    def test_evidence_requires_kernel_binding_to_om(self):
        self.execute()
        evidence = verify_build_records(self.root/"records.jsonl", self.source, file_hash(self.source), False)
        (self.root/"model.om").write_bytes(b"other_kernel")
        with self.assertRaises(TuningError):
            bind_selected_kernels(evidence, self.root)
        (self.root/"model.om").write_bytes(b"fixture_kernel")
        (self.root/"fixture_kernel.o").write_bytes(b"mock kernel code")
        bound = bind_selected_kernels(evidence, self.root)
        self.assertEqual(bound["om_kernel_names"], ["fixture_kernel"])
        self.assertEqual(bound["kernel_binary_sha256"]["fixture_kernel.o"], file_hash(self.root/"fixture_kernel.o"))

    def test_unknown_switch_and_changed_layout_fail(self):
        self.env["PARETOPILOT_TBE_POLICY"] = json.dumps({"InjectSync": False})
        with self.assertRaises(ValueError):
            self.execute()
        with self.assertRaises(ValueError):
            transform_source(b"def other(): return 1", str(self.source))

    def test_missing_or_incomplete_or_changed_sync_evidence_rejected(self):
        path = self.root/"records.jsonl"
        with self.assertRaises(TuningError):
            verify_build_records(path, self.source, file_hash(self.source), False)
        self.execute()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for variant in ([{"event": "hook_ready"}], records[:-1],
                        [{**r, "effective_build_args": {"enable_db_fold": False}} for r in records]):
            path.write_text("\n".join(map(json.dumps, variant)))
            with self.assertRaises(TuningError):
                verify_build_records(path, self.source, file_hash(self.source), False)

    def test_real_python_startup_hook_intercepts_only_target_loader(self):
        hook = self.root/"hook"
        hook.mkdir()
        (hook/"sitecustomize.py").write_text("from paretopilot.tbe_probe_hook import install\ninstall()\n")
        env = {**os.environ, **self.env, "PYTHONPATH": str(hook)+os.pathsep+str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1", "PARETOPILOT_TBE_SOURCE": str(self.source),
               "PARETOPILOT_TBE_SOURCE_SHA256": file_hash(self.source)}
        code = "import importlib.util,sys; s=importlib.util.spec_from_file_location('fixture',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m._binary_constant_build({'op_type':'BatchMatMulV2'})['enable_db_fold'] is False"
        result = subprocess.run([sys.executable, "-c", code, str(self.source)], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        verify_build_records(self.root/"records.jsonl", self.source, file_hash(self.source), False)

    def test_comparison_requires_gain_over_stock_and_default_and_confirmation(self):
        config = load_config(ROOT/"examples/ascend910b_matmul.json")
        config["measurement"]["confirmation_rounds"] = 2
        def trials(baseline, source, candidate):
            return [{"status": "ok", "role": role, "duration_us": t}
                    for _ in range(2) for role, t in zip(("baseline", "source_baseline", "candidate"), (baseline, source, candidate))]
        self.assertTrue(comparison(config, trials(10, 10, 8))["accepted"])
        self.assertFalse(comparison(config, trials(7, 10, 8))["accepted"])
        self.assertFalse(comparison(config, trials(10, 7, 8))["accepted"])
        self.assertFalse(comparison(config, trials(10, 10, 8)[:-1])["accepted"])

    def test_probe_pipeline_retains_dummy_and_orders_confirmation(self):
        config = load_config(ROOT/"examples/ascend910b_matmul_shared_rhs.json")
        config["measurement"]["confirmation_rounds"] = 2
        class Backend:
            def __init__(inner, config, root): pass
        calls = []
        def measure(backend, config, params, directory, role):
            calls.append((role, params))
            return {"status": "ok", "role": role, "duration_us": 8 if role == "candidate" else 10,
                    "roofline": analyze(config, 8 if role == "candidate" else 10),
                    "operator_identity": {"Op Type": "BatchMatMulV2"}}
        with patch("paretopilot.bmmv2_probe.inspect_environment", return_value={"ready": True}), \
             patch("paretopilot.bmmv2_probe.ProbeBackend", Backend), \
             patch("paretopilot.bmmv2_probe.installed_source", return_value=self.source), \
             patch("paretopilot.bmmv2_probe.evaluate", side_effect=measure):
            result = run_probe(config, self.root)
        self.assertFalse(result["manual_tiling_control"])
        self.assertEqual(result["agent"], {"kind": "dummy"})
        self.assertEqual([r for r, _ in calls[-6:]], ["baseline", "source_baseline", "candidate", "candidate", "source_baseline", "baseline"])
        self.assertEqual(read_json(self.root/"agent/output.json")["space"], {"enable_db_fold": [1, 0]})


if __name__ == "__main__":
    unittest.main()
