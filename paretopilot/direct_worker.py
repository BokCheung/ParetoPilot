"""Persistent CANN 8.5 kernel launcher. No ATC, OM, or torch in this process."""
from contextlib import ExitStack, contextmanager
import ctypes as C
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from .direct_build import select_kernel
from .official import read_effective_tiling, validate_params
from .official_cache import runtime_environment
from .util import TuningError, digest, file_hash, read_json, write_json
from .worker import check_outputs

P = C.c_void_p
U = C.c_uint32
S = C.c_size_t


def bind(lib, name, args, result=C.c_int):
    try:
        fn = getattr(lib, name)
    except AttributeError as exc:
        raise TuningError(f"CANN 8.5 kernel execution requires exported symbol {name}") from exc
    fn.argtypes, fn.restype = args, result
    return fn


@contextmanager
def capture_native(path):
    """Capture both native streams, including buffered C/C++ stdout."""
    sys.stdout.flush()
    sys.stderr.flush()
    def flush_native():
        if os.name == "nt":
            return
        try:
            bind(C.CDLL(None), "fflush", [P])(None)
        except (OSError, AttributeError, TuningError):
            pass
    flush_native()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    with Path(path).open("wb") as stream:
        for fd in saved:
            os.dup2(stream.fileno(), fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            flush_native()
            for fd, original in saved.items():
                os.dup2(original, fd)
                os.close(original)


def recent_error(lib):
    try:
        value = bind(lib, "aclGetRecentErrMsg", [], C.c_char_p)()
        return value.decode("utf-8", errors="replace")[-6000:] if isinstance(value, bytes) else ""
    except (OSError, AttributeError, TuningError):
        return ""


def decode_tiling(info):
    try:
        data = bytes.fromhex(info["tiling_data"])
        block, key = info["block_dim"], info["tiling_key"]
        workspace = info["workspaces"]
        if (not data or len(data) > 1024*1024 or type(block) is not int or not 1 <= block <= 24
                or type(key) is not int or not 0 <= key < 2**64
                or not isinstance(workspace, list) or len(workspace) != 1
                or type(workspace[0]) is not int or not 0 <= workspace[0] <= 2**34
                or type(info.get("clear_atomic")) is not bool):
            raise ValueError("Unsupported tiling/workspace/atomic-clean contract")
        return data, block, key, workspace[0]
    except (KeyError, ValueError, TypeError) as exc:
        raise TuningError(f"Invalid direct Host tiling result: {exc}") from exc


def initialize_compile_platform(lib, config, directory):
    """ACL device selection does not configure TBE's compilation platform."""
    record = {"status": "failed", "device": config["ascend"]["device"],
              "configured_soc": config["ascend"]["soc_version"],
              "api": "tbe.common.platform.set_current_compile_soc_info"}
    path = Path(directory)/"platform_init.json"
    try:
        with capture_native(Path(directory)/"platform_init.log"):
            name = bind(lib, "aclrtGetSocName", [], C.c_char_p)()
            if not name:
                raise TuningError("aclrtGetSocName returned no device SoC")
            record["runtime_soc"] = name.decode("utf-8")
            if record["runtime_soc"] != record["configured_soc"]:
                raise TuningError(f"Device SoC {record['runtime_soc']} differs from configured {record['configured_soc']}")
            platform = importlib.import_module("tbe.common.platform")
            record["module"] = str(getattr(platform, "__file__", ""))
            record["return_value"] = platform.set_current_compile_soc_info(record["runtime_soc"], core_type="AiCore")
            if record["return_value"] != "success":
                raise TuningError(f"TBE platform initialization returned {record['return_value']!r}")
            record["compile_soc"] = platform.get_soc_spec("SOC_VERSION")
            record["core_count"] = int(platform.get_soc_spec("CORE_NUM"))
            if record["compile_soc"] != record["runtime_soc"] or record["core_count"] <= 0:
                raise TuningError("TBE compile platform does not match the selected device")
            if config["search"].get("domain_mode") == "shape":
                from .domains import MEMORY_KEYS
                for key in MEMORY_KEYS:
                    record[key] = int(platform.get_soc_spec(key))
                    if record[key] <= 0:
                        raise TuningError(f"Invalid CANN platform resource {key}")
            record["status"] = "ok"
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", recent_error=recent_error(lib))
        raise TuningError(f"Compile platform initialization failed: {exc}; see {path} and platform_init.log") from exc
    finally:
        write_json(path, record)
    return record


class Runtime:
    def __init__(self, request):
        import numpy as np
        self.np = np
        self.c, self.build = request["config"], request["build"]
        self.case = Path(request["case"])
        self.cleanup = ExitStack()
        self.lib = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
        signatures = {
            "aclInit": [C.c_char_p], "aclFinalize": [], "aclrtSetDevice": [C.c_int],
            "aclrtResetDevice": [C.c_int], "aclrtCreateStream": [C.POINTER(P)],
            "aclrtDestroyStream": [P], "aclrtSynchronizeStream": [P],
            "aclrtMalloc": [C.POINTER(P), S, C.c_int], "aclrtFree": [P],
            "aclrtMemcpy": [P, S, P, S, C.c_int], "aclrtMemset": [P, S, C.c_int, S],
            "aclrtCreateEvent": [C.POINTER(P)], "aclrtDestroyEvent": [P],
            "aclrtRecordEvent": [P, P], "aclrtEventElapsedTime": [C.POINTER(C.c_float), P, P],
            "aclrtBinaryLoadFromFile": [C.c_char_p, P, C.POINTER(P)],
            "aclrtBinaryUnLoad": [P], "aclrtBinaryGetFunction": [P, C.c_char_p, C.POINTER(P)],
            "aclrtKernelArgsInit": [P, C.POINTER(P)],
            "aclrtKernelArgsAppend": [P, P, S, C.POINTER(P)],
            "aclrtKernelArgsFinalize": [P], "aclrtLaunchKernelWithConfig": [P, U, P, P, P, P],
        }
        for name, args in signatures.items():
            bind(self.lib, name, args)
        if file_hash(self.build["launcher_library"]) != self.build["launcher_sha256"]:
            raise TuningError("Native kernel timing bridge changed")
        self.launcher = C.CDLL(self.build["launcher_library"])
        self.measure = bind(self.launcher, "pp_launch_measured", [P, U, P, P, P, P, C.POINTER(C.c_float)])
        self.call("aclInit", None)
        self.cleanup.callback(self.call, "aclFinalize")
        device = self.c["ascend"]["device"]
        self.call("aclrtSetDevice", device)
        self.cleanup.callback(self.call, "aclrtResetDevice", device)
        self.platform_info = initialize_compile_platform(
            self.lib, self.c, Path(request.get("ready", self.case/"ready.json")).parent)
        self.stream = self.handle("aclrtCreateStream", "aclrtDestroyStream")
        self.events = [self.handle("aclrtCreateEvent", "aclrtDestroyEvent") for _ in range(2)]
        self.inputs = [np.ascontiguousarray(np.load(self.case/f"input_{i}.npy", allow_pickle=False)) for i in range(2)]
        self.reference = np.load(self.case/"reference.npy", allow_pickle=False)
        if any(a.dtype != np.float16 for a in [*self.inputs, self.reference]):
            raise TuningError("Direct BatchMatMulV3 requires FP16 tensors")
        self.pointers = []
        for array in self.inputs:
            ptr = self.allocate(array.nbytes)
            self.call("aclrtMemcpy", ptr, array.nbytes, P(array.ctypes.data), array.nbytes, 1)
            self.pointers.append(ptr)
        self.output = self.allocate(self.reference.nbytes)
        self.binaries, self.functions = {}, {}
        self.register = C.CDLL("libregister.so", mode=C.RTLD_GLOBAL)
        self.tiling = bind(self.register, "OpTilingForCompile",
                           [C.c_char_p]*6+[P, S, C.POINTER(C.c_uint64), C.c_char_p])
        loader = bind(self.register, "TbeLoadSoAndSaveToRegistry", [C.c_char_p])
        libraries = sorted({p.resolve() for p in (Path(self.build["opp_path"])/"op_impl/ai_core/tbe/op_tiling").rglob("*.so")})
        if not libraries:
            raise TuningError("Private OPP contains no Host tiling shared library")
        for path in libraries:
            if loader(os.fsencode(path)) != 0:
                raise TuningError(f"Cannot register private Host tiling library: {path}")
        self.tiling_libraries = [str(path) for path in libraries]

    def call(self, name, *args):
        ret = getattr(self.lib, name)(*args)
        if ret:
            raise TuningError(f"{name} failed with ACL code {ret}")

    def handle(self, create, destroy):
        ptr = P()
        self.call(create, C.byref(ptr))
        self.cleanup.callback(self.call, destroy, ptr)
        return ptr

    def allocate(self, size, cleanup=None):
        ptr = P()
        self.call("aclrtMalloc", C.byref(ptr), max(size, 32), 0)
        (cleanup or self.cleanup).callback(self.call, "aclrtFree", ptr)
        return ptr

    def host_tiling(self, params, directory):
        os.environ.update(runtime_environment(params))
        def desc(array):
            return {"shape": list(array.shape), "ori_shape": list(array.shape),
                    "format": "ND", "ori_format": "ND", "dtype": "float16"}
        compile_info = read_json(self.build["kernel_catalogs"][0]["metadata"]).get("compileInfo", {})
        if isinstance(compile_info, str):
            compile_info = json.loads(compile_info)
        compile_info = {**compile_info, "device_id": str(self.c["ascend"]["device"]),
                        "_cube_vector_core_type": "AiCore", "paretopilot_params": params}
        attrs = [{"name": name, "dtype": kind, "value": value} for name, kind, value in
                 (("adj_x1", "bool", False), ("adj_x2", "bool", False),
                  ("offset_x", "int64", 0), ("enable_hf32", "bool", False))]
        args = [b"BatchMatMulV3", json.dumps(compile_info).encode(), digest(compile_info).encode(),
                json.dumps([*(desc(a) for a in self.inputs), None, None]).encode(),
                json.dumps([desc(self.reference)]).encode(), json.dumps(attrs).encode()]
        write_json(directory/"host_tiling_request.json", {"compile_info": compile_info,
                   "inputs": json.loads(args[3]), "outputs": json.loads(args[4]), "attrs": attrs})
        buffer, elapsed = C.create_string_buffer(4*1024*1024), (C.c_uint64*2)()
        with capture_native(directory/"host_tiling.log"):
            print("Calling OpTilingForCompile: BatchMatMulV3; params="+digest(params), flush=True)
            ret = self.tiling(*args, buffer, len(buffer), elapsed,
                              b'{"op_name":"paretopilot_BatchMatMulV3"}')
            error = recent_error(self.lib) if ret != 1 else ""
            print(f"OpTilingForCompile returned {ret}; output_bytes={len(buffer.value)}", flush=True)
            if error:
                print("CANN recent error: "+error, flush=True)
        diagnostics = {"api": "OpTilingForCompile", "return_code": ret, "expected_success": 1,
                       "recent_error": error, "output_bytes": len(buffer.value),
                       "output_preview": buffer.value[:2000].decode("utf-8", errors="replace"),
                       "tiling_libraries": self.tiling_libraries, "register_library": str(self.register._name),
                       "compile_platform": self.platform_info,
                       "pid": os.getpid(), "request": str(directory/"host_tiling_request.json")}
        write_json(directory/"host_tiling_diagnostics.json", diagnostics)
        if ret != 1:
            raise TuningError(f"Official Host tiling returned {ret}; {error or 'CANN supplied no recent error'}; "
                              f"see {directory/'host_tiling_diagnostics.json'} and host_tiling.log")
        log = (directory/"host_tiling.log").read_text(encoding="utf-8", errors="replace")
        if "PARETOPILOT_PARAMS="+digest(params) not in log:
            raise TuningError("Direct Host tiling parameter signature missing or stale")
        effective = read_effective_tiling(directory/"host_tiling.log", params, batch=True)
        info = json.loads(buffer.value)
        write_json(directory/"host_tiling.json", info)
        return info, effective

    def function(self, kernel):
        path, entry = kernel["binary"], kernel["entry"]
        if path not in self.binaries:
            ptr = P()
            self.call("aclrtBinaryLoadFromFile", os.fsencode(path), None, C.byref(ptr))
            self.binaries[path] = ptr
            self.cleanup.callback(self.call, "aclrtBinaryUnLoad", ptr)
        if (path, entry) not in self.functions:
            ptr = P()
            self.call("aclrtBinaryGetFunction", self.binaries[path], entry.encode(), C.byref(ptr))
            self.functions[path, entry] = ptr
        return self.functions[path, entry]

    def evaluate(self, request):
        started = time.monotonic()
        directory, params = Path(request["directory"]), request["params"]
        validate_params(params, batch=True)
        info, effective = self.host_tiling(params, directory)
        data, block, key, workspace_size = decode_tiling(info)
        if any(r["tiling_key"] != key or r["block_dim"] != block for r in effective):
            raise TuningError("Host tiling bytes and observed key/blockDim disagree")
        kernel = select_kernel(self.build["kernel_catalogs"], key)
        function = self.function(kernel)
        m = self.c["measurement"]
        samples = []
        with ExitStack() as cleanup:
            workspace = self.allocate(workspace_size, cleanup)
            tiling = self.allocate(len(data), cleanup)
            raw = C.create_string_buffer(data)
            self.call("aclrtMemcpy", tiling, len(data), raw, len(data), 1)
            self.call("aclrtMemset", workspace, max(workspace_size, 32), 0, max(workspace_size, 32))

            def launch(timed=False):
                # The JSON bridge may default clear_atomic to true even for
                # overwrite-only kernels. Honor it conservatively for every
                # launch, including warmup, outside the timed interval.
                self.call("aclrtMemset", self.output, self.reference.nbytes,
                          0 if info["clear_atomic"] else 0xA5, self.reference.nbytes)
                if info["clear_atomic"]:
                    self.call("aclrtMemset", workspace, max(workspace_size, 32), 0, max(workspace_size, 32))
                # System-managed args are prepared for each launch, not retained
                # after submission. Preparation is outside the timed interval.
                args, param = P(), P()
                self.call("aclrtKernelArgsInit", function, C.byref(args))
                for pointer in (*self.pointers, P(), P(), self.output, workspace, tiling):
                    self.call("aclrtKernelArgsAppend", args, C.byref(pointer), C.sizeof(P), C.byref(param))
                self.call("aclrtKernelArgsFinalize", args)
                if timed:
                    ms = C.c_float()
                    ret = self.measure(function, block, self.stream, args, *self.events, C.byref(ms))
                    if ret:
                        raise TuningError(f"Native kernel event measurement failed with ACL code {ret}")
                    us = ms.value*1000
                    if not math.isfinite(us) or us <= 0:
                        raise TuningError("Invalid kernel event timing")
                    samples.append(us)
                else:
                    self.call("aclrtLaunchKernelWithConfig", function, block, self.stream, None, args, None)
                    self.call("aclrtSynchronizeStream", self.stream)

            def compare():
                actual = self.np.empty_like(self.reference)
                self.call("aclrtMemcpy", P(actual.ctypes.data), actual.nbytes, self.output, actual.nbytes, 2)
                result = check_outputs(actual, self.reference, m)
                self.np.save(directory/"output.npy", actual)
                write_json(directory/"correctness.json", result)
                if not result["passed"]:
                    raise TuningError("Direct kernel output failed reference comparison")
                return result

            launch()
            compare()
            for _ in range(m["warmup"]):
                launch()
            for _ in range(m["repeats"]):
                launch(timed=True)
                compare()
            correctness = compare()
        write_json(directory/"kernel_selection.json", kernel)
        return {"status": "ok", "mode": "ascend", "execution": "kernel",
                "duration_us": statistics.median(samples), "samples_us": samples, "measured_bytes": None,
                "timing_method": "device_event_us", "correctness": correctness,
                "input_shapes": [list(a.shape) for a in self.inputs], "output_shape": list(self.reference.shape),
                "operator_identity": {"Op Type": "BatchMatMulV3", "Op Name": kernel["entry"], "evidence": "compiled_kernel_metadata"},
                "selection": {**self.build, "params": params, "params_sha256": digest(params),
                              "platform_info": self.platform_info,
                              "effective_tiling": effective, "kernel": kernel, "tiling_sha256": file_hash(directory/"host_tiling.json")},
                "seconds": time.monotonic()-started}

    def close(self):
        self.cleanup.close()


def main():
    bootstrap = read_json(sys.argv[1])
    runtime = None
    try:
        runtime = object.__new__(Runtime)
        runtime.__init__(bootstrap)
        write_json(bootstrap["ready"], {"status": "ready"})
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("action") == "close":
                break
            try:
                result = runtime.evaluate(request)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(Path(request["directory"])/"direct_result.json", result)
    except Exception as exc:
        write_json(bootstrap["ready"], {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if runtime is not None and hasattr(runtime, "cleanup"):
            runtime.close()


if __name__ == "__main__":
    main()
