from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from paretopilot import official, official_cache as cache
from paretopilot.config import load_config
from paretopilot.search import run
from paretopilot.util import TuningError, digest, file_hash, read_json, tree_hash, write_json
from scripts.test_910b import verify_result

ROOT = Path(__file__).resolve().parents[1]


class SourceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        self.config["search"]["domain_mode"] = "manual"
        self.config["implementation"]["cache_dir"] = str(self.root/"cache")
        self.params = official.BATCH_DEFAULT_PARAMS.copy()

    def build(self, config, params, directory, env, runtime=False):
        self.assertTrue(runtime)
        opp = directory/"opp/vendor"
        opp.mkdir(parents=True)
        (opp/"kernel.so").write_bytes(b"package")
        legacy = directory/"opp/legacy.so"
        legacy.write_bytes(b"legacy")
        return {"opp_path": str(opp), "isolated_legacy_library": str(legacy),
                "installed_sha256": tree_hash(opp), "legacy_library_sha256": file_hash(legacy),
                "parameter_binding": "atc_process_environment"}

    def request(self, name, params=None):
        directory = self.root/name
        directory.mkdir()
        return cache.cached_build(self.config, params or self.params, directory, {})

    def test_different_params_shapes_and_runs_share_one_build(self):
        with patch.object(cache, "cache_identity", return_value={"source": "a"}), \
             patch.object(official, "_build_official", side_effect=self.build) as builder:
            first = self.request("first")
            self.config["operator"]["m"] = 100
            changed = {**self.params, "core_limit": 12, "batch_strategy": 1}
            second = self.request("second", changed)
            self.assertEqual(builder.call_count, 1)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["opp_path"], second["opp_path"])
            self.assertEqual(second["params_sha256"], digest(changed))
            self.assertNotEqual(first["params_sha256"], second["params_sha256"])

    def test_identity_tracks_toolchain_and_source_but_not_workload(self):
        legacy = self.root/"legacy.so"
        legacy.write_bytes(b"legacy")
        tool = self.root/"tool"
        tool.write_bytes(b"compiler")
        with patch.object(official, "verify_bundle", return_value={"archives": {"source": "hash"}}), \
             patch.object(official, "legacy_library", return_value=(legacy, None)), \
             patch.object(cache.shutil, "which", return_value=str(tool)):
            first = cache.cache_identity(self.config, {})
            changed = deepcopy(self.config)
            changed["operator"]["shape"] = [32, 100, 128]
            changed["space"]["core_limit"] = [12]
            self.assertEqual(first, cache.cache_identity(changed, {}))
            tool.write_bytes(b"new compiler")
            self.assertNotEqual(first, cache.cache_identity(changed, {}))
            tool.write_bytes(b"compiler")
            legacy.write_bytes(b"new CANN")
            self.assertNotEqual(first, cache.cache_identity(changed, {}))

    def test_corruption_builds_new_generation_and_preserves_old(self):
        with patch.object(cache, "cache_identity", return_value={"source": "a"}), \
             patch.object(official, "_build_official", side_effect=self.build) as builder:
            first = self.request("first")
            Path(first["isolated_legacy_library"]).write_bytes(b"corrupt")
            second = self.request("second")
            self.assertEqual(builder.call_count, 2)
            self.assertFalse(second["cache_hit"])
            self.assertNotEqual(first["opp_path"], second["opp_path"])
            self.assertTrue(Path(first["opp_path"]).is_dir())

    def test_failed_build_never_publishes_and_retry_succeeds(self):
        with patch.object(cache, "cache_identity", return_value={"source": "a"}), \
             patch.object(official, "_build_official", side_effect=TuningError("build failed")):
            with self.assertRaises(TuningError):
                self.request("failed")
        self.assertFalse(list((self.root/"cache").rglob("complete.json")))
        with patch.object(cache, "cache_identity", return_value={"source": "a"}), \
             patch.object(official, "_build_official", side_effect=self.build):
            self.assertFalse(self.request("retry")["cache_hit"])

    def test_concurrent_requests_publish_only_one_package(self):
        with patch.object(cache, "cache_identity", return_value={"source": "a"}), \
             patch.object(official, "_build_official", side_effect=self.build) as builder:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(self.request, ("one", "two")))
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(sorted(r["cache_hit"] for r in results), [False, True])

    def test_runtime_materialization_preserves_device_sources(self):
        project = self.root/"project"
        result = official.materialize(project, self.params, batch=True, runtime=True)
        parent = (project/official.HOST).with_name("paretopilot_params.h").read_text()
        self.assertEqual(parent, (project/official.BATCH_HOST).with_name("paretopilot_params.h").read_text())
        self.assertIn("std::getenv", parent)
        self.assertIn("PP_PARAMS_VALID", (project/official.HOST).read_text(encoding="utf-8"))
        self.assertIn(r"*end != '\0'", parent)
        self.assertNotIn(r"*end != '\\0'", parent)
        self.assertEqual(cache.runtime_environment(self.params)["PARETOPILOT_REQUEST_HASH"], digest(self.params))

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler required")
    def test_compiled_host_controls_change_between_processes(self):
        (self.root/"params.h").write_text(cache.runtime_header())
        (self.root/"main.cpp").write_text('#include "params.h"\n#include <cstdio>\nint main(){if(!PP_PARAMS_VALID)return 2;std::printf("%d %d %d %s",PP_CORE_LIMIT,PP_TILING_POLICY,PP_BATCH_STRATEGY,PP_PARAMS_HASH);}')
        executable = self.root/"check"
        subprocess.run(["g++", "-std=c++11", str(self.root/"main.cpp"), "-o", str(executable)], check=True, capture_output=True)
        for params in (self.params, {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}):
            env = {**os.environ, **cache.runtime_environment(params)}
            result = subprocess.run([str(executable)], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, f"{params['core_limit']} {params['tiling_policy']} {params['batch_strategy']} {digest(params)}")
        env["PARETOPILOT_CORE_LIMIT"] = "25"
        self.assertEqual(subprocess.run([str(executable)], env=env).returncode, 2)
        env["PARETOPILOT_CORE_LIMIT"] = "12junk"
        self.assertEqual(subprocess.run([str(executable)], env=env).returncode, 2)

    def test_fast_smoke_checks_default_and_explicit_tiles_without_gain_claim(self):
        from tests.test_official_batchmatmul import OfficialBatchMatmulTests
        helper = OfficialBatchMatmulTests()
        helper.c = self.config
        def trial(backend, config, params, directory, role):
            result = helper.trial(role, params)
            result["roofline"] = {"arithmetic_intensity": 1, "flops": 1}
            return result
        output = self.root/"smoke"
        with patch("paretopilot.search.inspect_environment", return_value={"ready": True}), \
             patch("paretopilot.search.AscendBackend"), \
             patch("paretopilot.search.evaluate", side_effect=trial) as evaluate, \
             patch("paretopilot.search.write_baseline_report"), \
             patch("paretopilot.search.load_agent") as agent:
            result = run(self.config, output, smoke=True)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(evaluate.call_args_list[0].args[-1], "source_baseline")
        self.assertEqual(evaluate.call_args.args[-1], "candidate")
        self.assertGreater(evaluate.call_args.args[2]["base_m"], 0)
        agent.assert_not_called()
        self.assertEqual(result["scope"], "source_smoke")
        self.assertNotIn("successful_candidates", result)
        self.assertFalse(result["tuning_accepted"])
        self.assertEqual(verify_result(output, True)["status"], "passed")
        with self.assertRaises(TuningError):
            verify_result(output, False)
        result["source_baseline"]["operator_identity"]["Op Type"] = "BatchMatMulV2"
        write_json(output/"summary.json", result)
        with self.assertRaises(TuningError):
            verify_result(output, True)
