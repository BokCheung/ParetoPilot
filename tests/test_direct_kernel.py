from contextlib import ExitStack
from copy import deepcopy
import ctypes as C
import json
import io
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import zipfile

from paretopilot import official
from paretopilot.config import load_config
from paretopilot.direct_backend import KernelBackend, validate_result
from paretopilot.direct_build import fp16_config, select_kernel, build_kernels, kernel_build_environment
from paretopilot.direct_worker import Runtime, decode_tiling, P, capture_native, recent_error, initialize_compile_platform
from paretopilot.search import confirmation
from paretopilot.util import TuningError, digest, file_hash, read_json, tree_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


class FakeNative:
    """CPU memory-backed ABI fixture, not evidence of NPU correctness/performance."""
    def __init__(self, reference, key=65536):
        self.reference, self.key = reference, key
        self.allocations, self.calls, self.arguments = {}, [], []
        self.count = 100
        self.bad_output = False
        self.tiling_failure = False
        self.error_message = b""
        self.overwrite_plan = False

    def __getattr__(self, name):
        def function(*args):
            self.calls.append(name)
            if name in ("aclrtCreateStream", "aclrtCreateEvent", "aclrtKernelArgsInit", "aclrtBinaryLoadFromFile", "aclrtBinaryGetFunction"):
                self.count += 1
                output = args[0] if name in ("aclrtCreateStream", "aclrtCreateEvent") else args[-1]
                output._obj.value = self.count
                if name == "aclrtKernelArgsInit":
                    self.arguments = []
            elif name == "aclrtMalloc":
                buffer = C.create_string_buffer(args[1])
                self.allocations[C.addressof(buffer)] = buffer
                args[0]._obj.value = C.addressof(buffer)
            elif name == "aclrtFree":
                del self.allocations[args[0].value]
            elif name == "aclrtMemcpy":
                C.memmove(args[0], args[2], args[3])
            elif name == "aclrtMemset":
                C.memset(args[0], args[2], args[3])
            elif name == "aclrtKernelArgsAppend":
                self.arguments.append(args[1]._obj.value)
            elif name == "aclrtLaunchKernelWithConfig":
                assert len(self.arguments) == 7
                assert self.arguments[2:4] == [None, None]
                if not self.bad_output:
                    C.memmove(self.arguments[4], self.reference.ctypes.data, self.reference.nbytes)
            elif name == "pp_launch_measured":
                self.aclrtRecordEvent(args[4], args[2])
                self.aclrtLaunchKernelWithConfig(args[0], args[1], args[2], None, args[3], None)
                self.aclrtRecordEvent(args[5], args[2])
                self.aclrtSynchronizeStream(args[2])
                args[6]._obj.value = 0.008
            elif name == "aclrtEventElapsedTime":
                args[0]._obj.value = 0.008
            elif name == "OpTilingForCompile":
                if self.tiling_failure:
                    return 0
                params = json.loads(args[1])["paretopilot_params"]
                record = {**params, "base_m": params.get("base_m") or 16, "base_n": params.get("base_n") or 16, "base_k": 16,
                          "used_core_num": 12, "block_dim": 12, "tiling_key": self.key,
                          "batch_count": 1, "multi_batch": 0, "load_mode": 0}
                assert os.environ["PARETOPILOT_CORE_LIMIT"] == str(params["core_limit"])
                assert os.environ["PARETOPILOT_BASE_M"] == str(params.get("base_m", 0))
                assert os.environ["PARETOPILOT_BASE_N"] == str(params.get("base_n", 0))
                assert os.environ["PARETOPILOT_REQUEST_HASH"] == digest(params)
                assert json.loads(args[3])[-2:] == [None, None]
                assert json.loads(args[5])[2]["dtype"] == "int64"
                plan = {k: record[k] for k in ("base_m", "base_n", "base_k", "used_core_num")}
                plan.update(raw="01020304", single_core_m=16, single_core_n=256, single_core_k=256,
                            step_m=1, step_n=16, step_ka=1, step_kb=1, depth_a1=1, depth_b1=16)
                message = (f"PARETOPILOT_PARAMS={digest(params)}\nPARETOPILOT_OFFICIAL={official.COMMIT}\n"
                           "PARETOPILOT_API_CUBE_PLAN="+json.dumps(plan)+"\n"
                           "PARETOPILOT_FINAL_CUBE_PLAN="+json.dumps({**plan, "step_n": 1} if self.overwrite_plan else plan)+"\n"
                           "PARETOPILOT_TILE_REQUEST="+json.dumps({"base_m": params.get("base_m", 0), "base_n": params.get("base_n", 0), "base_k_auto": True})+"\n"
                           "PARETOPILOT_BATCH_TILING="+json.dumps(record)+"\n")
                os.write(2, message.encode())
                info = {"tiling_key": self.key, "block_dim": 12, "workspaces": [32],
                        "clear_atomic": True, "tiling_data": "01020304"}
                C.memmove(args[6], json.dumps(info).encode()+b"\0", len(json.dumps(info))+1)
                return 1
            elif name == "aclGetRecentErrMsg":
                return self.error_message
            elif name == "aclrtGetSocName":
                return b"Ascend910B2"
            return 0
        setattr(self, name, function)
        return function


class DirectKernelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT/"examples/ascend910b_batchmatmul.json")
        self.params = official.BATCH_DEFAULT_PARAMS.copy()

    def catalog(self):
        binary = self.root/"kernel.o"
        binary.write_bytes(b"fixture ELF")
        metadata = self.root/"kernel.json"
        write_json(metadata, {"binFileName": "kernel", "compileInfo": {}, "kernelList": [
            {"tilingKey": 65536, "kernelName": "official_kernel_65536", "kernelType": "MIX_AIC"}]})
        return [{"binary": str(binary), "binary_sha256": file_hash(binary),
                 "metadata": str(metadata), "metadata_sha256": file_hash(metadata)}]

    def test_official_binary_recipe_is_restricted_to_fp16_nd(self):
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as archive:
            original = json.loads(archive.read("matmul/batch_mat_mul_v3/op_host/config/ascend910b/batch_mat_mul_v3_binary.json"))
        selected = fp16_config(original)
        self.assertEqual(len(selected["op_list"]), 1)
        self.assertEqual(selected["op_list"][0]["bin_filename"], "BatchMatMulV3_ND_ND_ND_ND_FP16_FP16_FP16_FP16")
        self.assertGreater(len(original["op_list"]), 1)
        with self.assertRaises(TuningError):
            fp16_config({**original, "op_list": []})

    def test_capture_includes_native_stdout_and_stderr_and_handles_missing_recent_error(self):
        path = self.root/"native.log"
        with capture_native(path):
            os.write(1, b"native stdout\n")
            os.write(2, b"native stderr\n")
        self.assertIn(b"native stdout", path.read_bytes())
        self.assertIn(b"native stderr", path.read_bytes())
        self.assertEqual(recent_error(SimpleNamespace()), "")

    def platform_fixture(self):
        return SimpleNamespace(__file__="fixture/platform.py",
            set_current_compile_soc_info=Mock(return_value="success"),
            get_soc_spec=Mock(side_effect=lambda key: {"SOC_VERSION": "Ascend910B2", "CORE_NUM": 24,
                "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "L1_SIZE": 524288}[key]))

    def test_compile_platform_uses_actual_soc_and_rejects_incomplete_initialization(self):
        native = SimpleNamespace(aclrtGetSocName=Mock(return_value=b"Ascend910B2"))
        platform = self.platform_fixture()
        with patch.dict(sys.modules, {"tbe.common.platform": platform}):
            result = initialize_compile_platform(native, self.config, self.root)
            self.assertEqual(result["status"], "ok")
            platform.set_current_compile_soc_info.assert_called_once_with("Ascend910B2", core_type="AiCore")
            for bad in ("error", None):
                platform.set_current_compile_soc_info.return_value = bad
                with self.assertRaisesRegex(TuningError, "initialization returned"):
                    initialize_compile_platform(native, self.config, self.root)
                self.assertEqual(read_json(self.root/"platform_init.json")["status"], "failed")
            platform.set_current_compile_soc_info.return_value = "success"
            platform.get_soc_spec.side_effect = lambda key: {"SOC_VERSION": "Ascend310", "CORE_NUM": 8}[key]
            with self.assertRaisesRegex(TuningError, "does not match"):
                initialize_compile_platform(native, self.config, self.root)
            native.aclrtGetSocName.return_value = b"Ascend910B1"
            platform.set_current_compile_soc_info.reset_mock()
            with self.assertRaisesRegex(TuningError, "differs from configured"):
                initialize_compile_platform(native, self.config, self.root)
            platform.set_current_compile_soc_info.assert_not_called()

    def test_key_lookup_rejects_missing_duplicate_and_modified_artifact(self):
        catalogs = self.catalog()
        self.assertEqual(select_kernel(catalogs, 65536)["entry"], "official_kernel_65536")
        with self.assertRaisesRegex(TuningError, "found 0"):
            select_kernel(catalogs, 999)
        with self.assertRaisesRegex(TuningError, "found 2"):
            select_kernel(catalogs+catalogs, 65536)
        Path(catalogs[0]["binary"]).write_bytes(b"changed")
        with self.assertRaisesRegex(TuningError, "changed"):
            select_kernel(catalogs, 65536)

    def test_tiling_contract_checks_atomic_flag_bytes_blocks_and_workspaces(self):
        info = {"tiling_data": "0100", "tiling_key": 65536, "block_dim": 12,
                "workspaces": [32], "clear_atomic": False}
        self.assertEqual(decode_tiling(info), (b"\1\0", 12, 65536, 32))
        self.assertEqual(decode_tiling({**info, "clear_atomic": True}), (b"\1\0", 12, 65536, 32))
        for key, value in (("tiling_data", "xyz"), ("block_dim", 0), ("block_dim", 25),
                           ("workspaces", [-1]), ("workspaces", [32, 64]), ("clear_atomic", "true")):
            with self.assertRaises(TuningError, msg=key):
                decode_tiling({**info, key: value})

    def test_kernel_build_targets_only_batch_and_requires_catalog(self):
        project = self.root/"project"
        config_path = project/"matmul/batch_mat_mul_v3/op_host/config/ascend910b/batch_mat_mul_v3_binary.json"
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as archive:
            write_json(config_path, json.loads(archive.read(config_path.relative_to(project).as_posix())))
        env_script = project/"scripts/kernel/binary_script/build_env.sh"
        env_script.parent.mkdir(parents=True)
        env_script.write_text('SOC_MAP=([Ascend910B]="Ascend910B1")')
        out = project/"build/binary/ascend910b/bin/ascend910b/batch_mat_mul_v3"
        def command(argv, cwd, log, timeout, env):
            if "--build" in argv:
                self.assertEqual(env["ASCENDC_PER_COMPILE_JOB_THREAD"], "4")
                out.mkdir(parents=True)
                (out/"kernel.o").write_bytes(b"fixture")
                write_json(out/"kernel.json", {"binFileName": "kernel", "binFileSuffix": ".o", "kernelList": [{"tilingKey": 65536, "kernelName": "entry"}]})
            if argv[0] == "g++":
                Path(argv[-1]).write_bytes(b"fixture bridge")
            return {"argv": argv}
        cann = self.root/"cann"
        (cann/"include/acl").mkdir(parents=True)
        (cann/"include/acl/acl.h").write_text("fixture header")
        (cann/"lib64").mkdir()
        (cann/"lib64/libascendcl.so").write_bytes(b"fixture runtime")
        with patch("paretopilot.direct_build.run_command", side_effect=command) as build:
            result = build_kernels(self.config, {"source_dir": str(project)}, self.root, {"ASCEND_HOME_PATH": str(cann)})
        self.assertEqual(build.call_count, 3)
        self.assertIn("config_compile_ascend910b_batch_mat_mul_v3", build.call_args_list[1].args[0])
        self.assertEqual(len(result["kernel_catalogs"]), 1)
        self.assertEqual(result["binary_soc"], "Ascend910B2")
        self.assertIn('"Ascend910B2"', env_script.read_text())
        self.assertNotIn(b'\r', env_script.read_bytes())

    def test_upstream_wrapper_receives_three_arguments_without_inherited_thread_variable(self):
        config = deepcopy(self.config)
        config["implementation"]["build_jobs"] = 3
        original = {**os.environ, "ASCENDC_PER_COMPILE_JOB_THREAD": ""}
        env = kernel_build_environment(config, original)
        self.assertEqual(env["ASCENDC_PER_COMPILE_JOB_THREAD"], "3")
        self.assertEqual(original["ASCENDC_PER_COMPILE_JOB_THREAD"], "")
        if os.name == "nt":
            return
        # Run the actual upstream wrapper, stubbing only task generation and
        # compilation. The previous empty variable reduced child argc to two.
        with zipfile.ZipFile(official.VENDOR/"ops-nn-matmul.zip") as archive:
            for name in ("build_binary_single_op.sh", "get_threadnum_with_op.sh"):
                raw = archive.read("scripts/kernel/binary_script/"+name)
                (self.root/name).write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        (self.root/"build_binary_single_op_gen_task.sh").write_text('printf "ascendc_per_compile_job=0\\n"\n', newline="\n")
        (self.root/"build_binary_single_op_exe_task.sh").write_text(
            'test "$#" = 3 || exit 20\ntest "$2" = 3 || exit 21\nprintf "argc=%s threads=%s\\n" "$#" "$2"\n', newline="\n")
        tools = self.root/"tools"
        tools.mkdir()
        gawk = tools/"gawk"
        gawk.write_text('#!/bin/sh\ncat\n', newline="\n")
        gawk.chmod(0o755)
        env["PATH"] = str(tools)+os.pathsep+env["PATH"]
        result = subprocess.run(["bash", "build_binary_single_op.sh", "BatchMatMulV3", "ascend910b", str(self.root/"out")],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=15, check=True)
        self.assertIn("argc=3 threads=3", result.stdout)

    def test_persistent_launcher_changes_parameters_without_atc_for_all_shapes(self):
        import numpy as np
        for suffix in ("", "_shared_rhs", "_batched_rhs"):
            config = load_config(ROOT/f"examples/ascend910b_batchmatmul{suffix}.json")
            config["measurement"].update(repeats=3, warmup=1)
            case = self.root/(suffix or "2d")
            case.mkdir()
            from paretopilot.workload import shapes
            inputs, output = shapes(config["operator"])
            for i, shape in enumerate(inputs):
                np.save(case/f"input_{i}.npy", np.zeros(shape, dtype=np.float16))
            reference = np.ones(output, dtype=np.float16)
            np.save(case/"reference.npy", reference)
            opp = case/"opp/op_impl/ai_core/tbe/op_tiling"
            opp.mkdir(parents=True)
            (opp/"libcust_opmaster.so").write_bytes(b"fixture Host")
            build = {"opp_path": str(case/"opp"), "commit": official.COMMIT, "operator": "BatchMatMulV3",
                     "installed_sha256": tree_hash(case/"opp"), "kernel_catalogs": self.catalog()}
            launcher = case/"launcher.so"
            launcher.write_bytes(b"fixture launcher")
            build.update(launcher_library=str(launcher), launcher_sha256=file_hash(launcher))
            fake = FakeNative(reference)
            platform = self.platform_fixture()
            with patch("paretopilot.direct_worker.C.CDLL", return_value=fake), patch.dict(os.environ), \
                    patch.dict(sys.modules, {"tbe.common.platform": platform}):
                runtime = Runtime({"config": config, "build": build, "case": str(case)})
                try:
                    for index, params in enumerate((self.params, {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1},
                            {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1, "base_m": 16, "base_n": 16})):
                        directory = case/str(index)
                        directory.mkdir()
                        result = runtime.evaluate({"params": params, "directory": str(directory)})
                        validate_result(result, config, params, build)
                        self.assertEqual(len(result["samples_us"]), 3)
                        self.assertAlmostEqual(result["duration_us"], 8, places=4)
                    self.assertEqual(fake.calls.count("aclInit"), 1)
                    platform.set_current_compile_soc_info.assert_called_once()
                    self.assertLess(fake.calls.index("aclrtSetDevice"), fake.calls.index("aclrtGetSocName"))
                    self.assertLess(fake.calls.index("aclrtGetSocName"), fake.calls.index("OpTilingForCompile"))
                    self.assertEqual(fake.calls.count("aclrtBinaryLoadFromFile"), 1)
                    self.assertEqual(fake.calls.count("OpTilingForCompile"), 3)
                    fake.overwrite_plan = True
                    overwritten = case/"overwritten"
                    overwritten.mkdir()
                    launches = fake.calls.count("aclrtLaunchKernelWithConfig")
                    with self.assertRaisesRegex(TuningError, "was overwritten"):
                        runtime.evaluate({"params": params, "directory": str(overwritten)})
                    self.assertEqual(fake.calls.count("aclrtLaunchKernelWithConfig"), launches)
                    fake.overwrite_plan = False
                    fake.bad_output = True
                    bad = case/"bad"
                    bad.mkdir()
                    with self.assertRaisesRegex(TuningError, "reference comparison"):
                        runtime.evaluate({"params": self.params, "directory": str(bad)})
                    fake.tiling_failure = True
                    fake.error_message = b"fixture CANN compile info error"
                    failed = case/"failed-host"
                    failed.mkdir()
                    with self.assertRaisesRegex(TuningError, "returned 0"):
                        runtime.host_tiling(self.params, failed)
                    diagnostics = read_json(failed/"host_tiling_diagnostics.json")
                    self.assertEqual(diagnostics["return_code"], 0)
                    self.assertEqual(diagnostics["recent_error"], "fixture CANN compile info error")
                    self.assertEqual(diagnostics["output_bytes"], 0)
                    self.assertIn("OpTilingForCompile returned 0", (failed/"host_tiling.log").read_text())
                    self.assertIn("fixture CANN compile info error", (failed/"host_tiling.log").read_text())
                finally:
                    runtime.close()
            self.assertFalse(fake.allocations)

    def test_final_confirmation_uses_om_backend_and_never_direct_launch(self):
        from tests.test_official_batchmatmul import OfficialBatchMatmulTests
        helper = OfficialBatchMatmulTests()
        helper.c = self.config
        winner = {"params": {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1}}
        om = object()
        backend = SimpleNamespace(om_backend=om)
        def evaluate(actual, config, params, directory, role):
            self.assertIs(actual, om)
            return helper.trial(role, params)
        with patch("paretopilot.search.evaluate", side_effect=evaluate) as execution:
            result = confirmation(self.config, winner, backend, self.root)
        self.assertEqual(execution.call_count, 9)
        self.assertEqual(len(result["trials"]), 9)

    def test_controller_routes_stock_to_om_and_reuses_live_worker_for_candidates(self):
        from tests.test_official_batchmatmul import OfficialBatchMatmulTests
        helper = OfficialBatchMatmulTests()
        helper.c = self.config
        opp = self.root/"opp"
        opp.mkdir()
        (opp/"host.so").write_bytes(b"immutable Host fixture")
        backend = object.__new__(KernelBackend)
        backend.c = self.config
        backend.build = {"installed_sha256": tree_hash(opp), "opp_path": str(opp)}
        backend.process = SimpleNamespace(stdin=io.StringIO())
        backend.om_backend = SimpleNamespace(evaluate=Mock(return_value={"stock": True}), source_builds={})
        backend.start = Mock()
        self.assertEqual(backend.evaluate({}, self.root, "baseline"), {"stock": True})
        for i, params in enumerate((self.params, {"core_limit": 12, "tiling_policy": 1, "batch_strategy": 1})):
            result = helper.trial("candidate", params)
            result.update(execution="kernel", timing_method="device_event_us")
            result["selection"].update(params=params, params_sha256=digest(params), installed_sha256=tree_hash(opp))
            directory = self.root/str(i)
            directory.mkdir()
            backend.wait_file = Mock(return_value=result)
            backend.evaluate(params, directory, "candidate")
            self.assertIn(digest(params), backend.om_backend.source_builds)
            bad = deepcopy(result)
            bad["duration_us"] += 1
            with self.assertRaisesRegex(TuningError, "P50"):
                validate_result(bad, self.config, params, backend.build)
        backend.start.assert_not_called()
        self.assertEqual(len(backend.process.stdin.getvalue().splitlines()), 2)
        backend.om_backend.evaluate.assert_called_once()

    def test_configuration_rejects_non_batch_and_uncached_execution(self):
        config = deepcopy(self.config)
        config["implementation"].update(execution="kernel", build_cache=False)
        write_json(self.root/"bad.json", config)
        with self.assertRaisesRegex(TuningError, "build_cache"):
            load_config(self.root/"bad.json")


if __name__ == "__main__":
    unittest.main()
