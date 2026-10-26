from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
        self.assertEqual(changed, {official.HOST, official.INFER, official.IMPL_CMAKE, "CMakeLists.txt", "common/CMakeLists.txt",
                                  "cmake/third_party/nlohmann_json.cmake", "cmake/third_party/makeself-fetch.cmake"})
        self.assertEqual(provenance["commit"], official.COMMIT)
        host = (project/official.HOST).read_text(encoding="utf-8")
        self.assertLess(host.index("compileInfo_.aicNum = PP_CORE_LIMIT"), host.index("MatmulV3BaseTiling::DoOpTiling"))
        self.assertIn("PP_TILING_POLICY == 0 && GetTilingFromRepo()", host)
        self.assertIn("PARETOPILOT_TILING=", host)
        self.assertNotIn("PP_TILE_M", host)
        self.assertTrue((project/"third_party/makeself/makeself.sh").is_file())

    def test_source_default_and_candidate_share_dtype_registration_and_original_shape_inference(self):
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as z:
            original = z.read(official.INFER)
        shape_source = original.decode().replace("\r\n", "\n").split("} // namespace\nnamespace ops {")[0]
        generated = []
        for name, params in (("source_baseline", official.DEFAULT_PARAMS),
                             ("candidate", {"core_limit": 12, "tiling_policy": 1})):
            with self.subTest(role=name):
                project = self.root/name
                provenance = official.materialize(project, params)
                source = (project/official.INFER).read_text(encoding="utf-8")
                generated.append(source)
                # Keep all upstream shape logic; extend its existing registration,
                # since a second registration could replace the shape callback.
                self.assertTrue(source.startswith(shape_source))
                self.assertEqual(source.count("IMPL_OP_INFERSHAPE(MatMulV3)"), 1)
                self.assertRegex(source, r"IMPL_OP_INFERSHAPE\(MatMulV3\)\s*"
                                 r"\.InferShape\(InferShapeForMatMulV3\)\s*"
                                 r"\.InferDataType\(InferDataTypeForMatMulV3\);")
                self.assertIn("return context->SetOutputDataType(0, context->GetInputDataType(0));", source)
                patches = [p for p in provenance["patches"] if p["path"] == official.INFER]
                self.assertEqual(patches[0]["before_sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(patches[-1]["after_sha256"], hashlib.sha256((project/official.INFER).read_bytes()).hexdigest())
        self.assertEqual(generated[0], generated[1])

    @unittest.skipUnless(shutil.which("cmake"), "CMake required for install regression")
    def test_cmake_installs_python_adapters_generated_after_configuration(self):
        patched = self.root/"patched"
        official.materialize(patched, official.DEFAULT_PARAMS)
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as z:
            original = z.read(official.IMPL_CMAKE).decode()

        def command(argv, cwd, env=None):
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)
            return result.stdout

        for name, cmake in (("original", original), ("fixed", (patched/official.IMPL_CMAKE).read_text(encoding="utf-8"))):
            with self.subTest(version=name):
                source = self.root/name
                source.mkdir()
                (source/"gen_ops_info.cmake").write_text(cmake, encoding="utf-8")
                (source/"CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.16)
project(PackageRegression NONE)
set(ENABLE_PACKAGE ON)
include(gen_ops_info.cmake)
add_ops_impl_target(TARGET adapters
    OPS_INFO_DIR "${CMAKE_BINARY_DIR}/autogen"
    IMPL_DIR "${CMAKE_BINARY_DIR}/tbe/ascendc"
    OUT_DIR "${CMAKE_BINARY_DIR}/tbe"
    INSTALL_DIR "op_impl/ai_core/tbe/paretopilot_nn_impl/dynamic")
''', encoding="utf-8")
                build, vendor = source/"build", source/"paretopilot_nn"
                generator = ["-G", "Ninja"] if shutil.which("ninja") else []
                command(["cmake", "-S", str(source), "-B", str(build), *generator], source)
                # Simulate only the CANN generator. CMake configuration and
                # installation are real, and there is no second configure.
                dynamic = build/"tbe/dynamic"
                dynamic.mkdir(parents=True)
                for module in ("mat_mul_v3", "batch_mat_mul_v3"):
                    (dynamic/(module+".py")).write_text("VALUE = 'generated after configure'\n")
                (dynamic/"not-an-adapter.txt").write_text("exclude")
                command(["cmake", "--install", str(build), "--prefix", str(vendor)], source)
                installed = vendor/"op_impl/ai_core/tbe/paretopilot_nn_impl/dynamic"
                if name == "original":
                    self.assertFalse((installed/"mat_mul_v3.py").exists())
                    continue
                hashes = official.verify_installed_implementations(official.generated_implementations(build), vendor)
                self.assertEqual(len(hashes), 2)
                self.assertFalse((installed/"not-an-adapter.txt").exists())
                output = command([sys.executable, "-c",
                                  "from paretopilot_nn_impl.dynamic import mat_mul_v3; print(mat_mul_v3.VALUE)"],
                                 source, {**os.environ, "PYTHONPATH": str(vendor/"op_impl/ai_core/tbe")})
                self.assertEqual(output.strip(), "generated after configure")

    def test_python_adapter_checks_reject_missing_and_stale_installations(self):
        build, vendor = self.root/"build", self.root/"paretopilot_nn"
        with self.assertRaisesRegex(TuningError, "was not generated"):
            official.generated_implementations(build)
        generated = build/"tbe/dynamic/mat_mul_v3.py"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(b"new adapter")
        files = official.generated_implementations(build)
        with self.assertRaisesRegex(TuningError, "OPP is missing Python adapter"):
            official.verify_installed_implementations(files, vendor)
        installed = vendor/"op_impl/ai_core/tbe/paretopilot_nn_impl/dynamic/mat_mul_v3.py"
        installed.parent.mkdir(parents=True)
        installed.write_bytes(b"old adapter")
        with self.assertRaisesRegex(TuningError, "differs from generated"):
            official.verify_installed_implementations(files, vendor)
        installed.write_bytes(generated.read_bytes())
        self.assertEqual(official.verify_installed_implementations(files, vendor),
                         {"paretopilot_nn_impl/dynamic/mat_mul_v3.py": hashlib.sha256(generated.read_bytes()).hexdigest()})

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
            if "ascendc_impl_gen" in argv:
                dynamic = self.root/"project/build/tbe/dynamic"
                dynamic.mkdir(parents=True)
                (dynamic/"mat_mul_v3.py").write_text("generated adapter fixture")
            if "package" in argv:
                out = self.root/"project/build_out"
                out.mkdir(parents=True)
                (out/"custom.run").write_text("test fixture")
                build = self.root/"project/build"
                build.mkdir(exist_ok=True)
                (build/"libparetopilot_onnx.so").write_text("parser fixture")
            if argv[0] == "bash":
                self.assertEqual(argv[-1], "--install-path="+str(self.root/"opp"))
                dynamic = self.root/"opp/vendors/paretopilot_nn/op_impl/ai_core/tbe/paretopilot_nn_impl/dynamic"
                dynamic.mkdir(parents=True)
                shutil.copyfile(self.root/"project/build/tbe/dynamic/mat_mul_v3.py", dynamic/"mat_mul_v3.py")
            return {"argv": argv}
        with patch.object(official, "run_command", side_effect=command), patch.object(official.platform, "machine", return_value="aarch64"):
            result = official.build_official(self.c, official.DEFAULT_PARAMS.copy(), self.root, {"ASCEND_OPP_PATH": str(self.root/"system-opp")})
        self.assertEqual(result["commit"], official.COMMIT)
        self.assertIn("paretopilot_nn_impl/dynamic/mat_mul_v3.py", result["python_implementations"])
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
        args = test_910b.parser().parse_args(["--cann-path", "/cann", "--soc", "Ascend910B2", "--operator", "matmul", "--deps", "check", "--matmul-implementation", "legacy"])
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
                return {"status": "ok", "mode": "ascend", "operator_identity": {"Op Type": "MatMulV3"}, "duration_us": {"baseline": 12, "source_baseline": 10, "candidate": 8}[role],
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
