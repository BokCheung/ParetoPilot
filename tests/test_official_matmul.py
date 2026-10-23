from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from paretopilot import official
from paretopilot.agent import DummyAgent
from paretopilot.backend import AscendBackend
from paretopilot.config import load_config
from paretopilot.cli import main as cli_main
from paretopilot.roofline import analyze
from paretopilot.search import confirmation, confirmation_roles, run
from paretopilot.space import enumerate_space
from paretopilot.util import TuningError, read_json, write_json
from scripts import test_910b

ROOT = Path(__file__).resolve().parents[1]


class OfficialMatmulTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = load_config(ROOT/"examples/ascend910b_matmul.json")
        self.c["measurement"]["confirmation_rounds"] = 2

    def test_pinned_archive_contains_exact_manifest_sources_and_licenses(self):
        manifest = official.verify_bundle()
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as z:
            self.assertEqual(set(manifest["upstream_files"]), {n for n in z.namelist() if not n.endswith('/')})
            for path, sha in manifest["upstream_files"].items():
                self.assertEqual(hashlib.sha256(z.read(path)).hexdigest(), sha, path)
            self.assertIn("CANN Open Software", z.read("LICENSE").decode())
            self.assertIn("conv/common/op_kernel/arch35/", z.namelist())
        with zipfile.ZipFile(official.VENDOR/"build-deps.zip") as z:
            self.assertIn("third_party/json/LICENSE.MIT", z.namelist())
            self.assertIn("third_party/makeself/COPYING", z.namelist())

    def test_materialization_preserves_official_kernel_and_limits_source_patches(self):
        project = self.root/"project"
        provenance = official.materialize(project, {"core_limit": 12, "tiling_policy": 1})
        changed = set()
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as z:
            for n in z.namelist():
                if not n.endswith('/') and (project/n).read_bytes() != z.read(n):
                    changed.add(n)
        self.assertEqual(changed, {official.HOST, "CMakeLists.txt", "common/CMakeLists.txt",
                                  "cmake/third_party/nlohmann_json.cmake", "cmake/third_party/makeself-fetch.cmake"})
        self.assertEqual(provenance["commit"], official.COMMIT)
        host = (project/official.HOST).read_text(encoding="utf-8")
        self.assertLess(host.index("compileInfo_.aicNum = PP_CORE_LIMIT"), host.index("MatmulV3BaseTiling::DoOpTiling"))
        self.assertIn("PP_TILING_POLICY == 0 && GetTilingFromRepo()", host)
        self.assertIn("PARETOPILOT_TILING=", host)
        self.assertNotIn("PP_TILE_M", host)
        self.assertTrue((project/"third_party/makeself/makeself.sh").is_file())

    def test_archive_tampering_fails_before_extraction(self):
        manifest = read_json(official.VENDOR/"manifest.json")
        manifest["archives"]["ops-nn-matmul.zip"] = "0"*64
        with patch.object(official, "read_json", return_value=manifest), self.assertRaisesRegex(TuningError, "checksum"):
            official.materialize(self.root/"project", official.DEFAULT_PARAMS)
        self.assertFalse((self.root/"project").exists())

    def test_invalid_core_and_policy_inputs_rejected(self):
        for params in ({"core_limit": -1, "tiling_policy": 0}, {"core_limit": 25, "tiling_policy": 0},
                       {"core_limit": True, "tiling_policy": 0}, {"core_limit": 12, "tiling_policy": 2},
                       {"core_limit": 12, "tile_m": 128}):
            with self.assertRaises(TuningError):
                official.validate_params(params)

    def test_official_domains_roofline_and_dummy_agree(self):
        pool = enumerate_space(self.c)
        self.assertEqual(len(pool), 12)
        self.assertIn(official.DEFAULT_PARAMS, pool)
        roof = analyze(self.c, 8)
        self.assertEqual(set(roof["suggested_priority"]), set(self.c["space"]))
        proposal = DummyAgent().propose({"original_space": self.c["space"], "roofline": roof})
        self.assertEqual(proposal["space"], self.c["space"])
        bad = deepcopy(self.c)
        bad["space"]["core_limit"] = [25]
        write_json(self.root/"bad.json", bad)
        with self.assertRaises(TuningError):
            load_config(self.root/"bad.json")

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "onnx required")
    def test_official_onnx_node_uses_unique_parser_origin(self):
        import onnx
        backend = AscendBackend.__new__(AscendBackend)
        backend.c, backend.case = self.c, self.root
        write_json(self.root/"case.json", {"inputs": [{"name": "a", "shape": [256, 256]}, {"name": "b", "shape": [256, 256]}], "output_shape": [256, 256]})
        backend.candidate_graph(self.root/"model.onnx")
        node = onnx.load(self.root/"model.onnx").graph.node[0]
        self.assertEqual(node.op_type, "ParetoPilotOfficialMatMul")
        self.assertEqual(list(node.input), ["a", "b"])

    def test_effective_tiling_requires_source_identity_and_enforces_core_limit(self):
        params = {"core_limit": 12, "tiling_policy": 1}
        record = {**params, "used_core_num": 12, "base_m": 128, "base_n": 256, "base_k": 64, "tiling_key": 0}
        log = self.root/"atc.log"
        def write(r, identity=True):
            log.write_text((f"PARETOPILOT_OFFICIAL={official.COMMIT}\n" if identity else "")+"PARETOPILOT_TILING="+json.dumps(r)+"\n")
        write(record)
        self.assertEqual(official.read_effective_tiling(log, params), [record])
        for r, identity in ((record, False), ({**record, "used_core_num": 24}, True), ({**record, "tiling_policy": 0}, True), ({}, True)):
            write(r, identity)
            with self.assertRaises(TuningError):
                official.read_effective_tiling(log, params)

    def test_official_build_is_offline_and_install_is_isolated(self):
        calls = []
        legacy_relative = Path("built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_comm_legacy.so")
        legacy = self.root/"system-opp"/legacy_relative
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy fixture")
        def command(argv, cwd, log, timeout, env):
            calls.append(argv)
            if "package" in argv:
                out = self.root/"project/build_out"
                out.mkdir(parents=True)
                (out/"custom.run").write_text("test fixture")
                build = self.root/"project/build"
                build.mkdir(exist_ok=True)
                (build/"libparetopilot_onnx.so").write_text("parser fixture")
            if argv[0] == "bash":
                self.assertEqual(argv[-1], "--install-path="+str(self.root/"opp"))
                (self.root/"opp/vendors/paretopilot_nn").mkdir(parents=True)
            return {"argv": argv}
        with patch.object(official, "run_command", side_effect=command), patch.object(official.platform, "machine", return_value="aarch64"):
            result = official.build_official(self.c, official.DEFAULT_PARAMS.copy(), self.root, {"ASCEND_OPP_PATH": str(self.root/"system-opp")})
        self.assertEqual(result["commit"], official.COMMIT)
        self.assertTrue(Path(result["opp_path"]).is_relative_to(self.root))
        self.assertTrue((Path(result["opp_path"])/"framework/onnx/libparetopilot_onnx.so").exists())
        self.assertEqual((self.root/"opp"/legacy_relative).read_bytes(), legacy.read_bytes())
        self.assertIn("-DASCEND_COMPUTE_UNIT=ascend910b", calls[0])
        self.assertFalse(any(a in ("git", "curl", "wget", "msopgen") for c in calls for a in c))

    def confirm(self, times, params=None):
        class Backend:
            def evaluate(inner, p, directory, role):
                return {"status": "ok", "duration_us": times[role], "measured_bytes": None}
        return confirmation(self.c, {"params": params or {"core_limit": 12, "tiling_policy": 1}}, Backend(), self.root)

    def test_confirmation_requires_gain_against_both_references(self):
        result = self.confirm({"baseline": 10, "source_baseline": 9, "candidate": 8})
        self.assertTrue(result["accepted"])
        self.assertTrue(result["tuning_accepted"])
        self.assertEqual(result["selected"], "candidate")
        self.assertEqual(result["source_speedup"], 9/8)
        self.assertEqual([t["role"] for t in result["trials"]], ["baseline", "source_baseline", "candidate", "candidate", "source_baseline", "baseline"])

    def test_source_improvement_does_not_replace_faster_stock(self):
        result = self.confirm({"baseline": 7, "source_baseline": 10, "candidate": 8})
        self.assertTrue(result["tuning_accepted"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected"], "baseline")

    def test_identical_default_params_never_claim_tuning_gain(self):
        result = self.confirm({"baseline": 12, "source_baseline": 10, "candidate": 9}, official.DEFAULT_PARAMS.copy())
        self.assertFalse(result["tuning_accepted"])
        self.assertEqual(result["selected"], "source_baseline")

    def test_source_default_can_win_when_search_is_slower(self):
        result = self.confirm({"baseline": 12, "source_baseline": 8, "candidate": 10})
        self.assertFalse(result["tuning_accepted"])
        self.assertEqual(result["selected"], "source_baseline")

    def test_matmul_only_smoke_uses_original_auto_parameters(self):
        args = test_910b.parser().parse_args(["--cann-path", "/cann", "--soc", "Ascend910B2", "--operator", "matmul", "--deps", "check"])
        configs = test_910b.prepare_configs(args, self.root)
        self.assertEqual(set(configs), {"matmul", "matmul-smoke"})
        config = read_json(configs["matmul-smoke"])
        self.assertEqual(config["space"], {k: [v] for k, v in official.DEFAULT_PARAMS.items()})
        self.assertEqual(test_910b.operator_kinds(args), ("matmul",))

    def test_direct_cli_smoke_also_uses_original_auto_parameters(self):
        with patch("paretopilot.cli.sys.platform", "linux"), patch("paretopilot.cli.run", return_value={"successful_candidates": 1}) as run_mock, redirect_stdout(io.StringIO()):
            code = cli_main(["run", "--config", str(ROOT/"examples/ascend910b_matmul.json"), "--output", str(self.root/"run"), "--smoke"])
        self.assertEqual(code, 0)
        self.assertEqual(run_mock.call_args.args[0]["space"], {k: [v] for k, v in official.DEFAULT_PARAMS.items()})

    def test_missing_legacy_library_fails_before_expensive_build(self):
        with patch.object(official, "run_command") as command, self.assertRaisesRegex(TuningError, "libophost_comm_legacy"):
            official.build_official(self.c, official.DEFAULT_PARAMS.copy(), self.root, {})
        command.assert_not_called()
        self.assertFalse((self.root/"project").exists())

    def test_search_uses_source_baseline_for_agent_and_preserves_stock(self):
        c = deepcopy(self.c)
        c["search"].update(budget=1, population=2, generations=1)
        c["space"] = {"core_limit": [12], "tiling_policy": [1]}
        class Backend:
            def __init__(inner, config, root): pass
            def evaluate(inner, params, directory, role):
                return {"status": "ok", "mode": "ascend", "duration_us": {"baseline": 12, "source_baseline": 10, "candidate": 8}[role],
                        "measured_bytes": None, "correctness": {"passed": True},
                        "selection": {"commit": official.COMMIT, "effective_tiling": [{"used_core_num": 12}]}}
        with patch("paretopilot.search.AscendBackend", Backend), patch("paretopilot.search.inspect_environment", return_value={"ready": True}):
            run(c, self.root/"run")
        result = read_json(self.root/"run/summary.json")
        context = read_json(self.root/"run/agent/000.input.json")
        self.assertEqual(context["baseline"]["role"], "source_baseline")
        self.assertEqual(context["stock_baseline"]["role"], "baseline")
        self.assertEqual(result["selected"], "candidate")
        accepted = test_910b.verify_result(self.root/"run", True)
        self.assertEqual(accepted["task_metrics"]["source_baseline_us"], 10)
        self.assertIn("Official source defaults", (self.root/"run/report.md").read_text())


if __name__ == "__main__":
    unittest.main()
