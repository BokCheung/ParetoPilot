"""Kernel search backend with a persistent launcher and independent OM confirmation."""
import atexit
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .backend import AscendBackend, environment
from .build import build_candidate
from .official import BATCH_DEFAULT_PARAMS, COMMIT, validate_params
from .kernel_profiling import direct_profile
from .profiling import inspect_profiles
from .util import TuningError, digest, file_hash, read_json, tree_hash, write_json
from .workload import shapes


def validate_result(result, config, params, build):
    """Validate raw worker Event evidence; never use this as profiler evidence."""
    inputs, output = shapes(config["operator"])
    selection = result.get("selection") or {}
    samples = result.get("samples_us", [])
    if (result.get("status") != "ok" or result.get("mode") != "ascend"
            or result.get("execution") != "kernel" or result.get("timing_method") != "device_event_us"
            or result.get("correctness", {}).get("passed") is not True
            or result.get("correctness", {}).get("simulated", False)
            or result.get("input_shapes") != inputs or result.get("output_shape") != output
            or result.get("operator_identity", {}).get("Op Type") != "BatchMatMulV3"
            or len(samples) != config["measurement"]["repeats"]
            or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in samples)
            or selection.get("commit") != COMMIT or selection.get("params_sha256") != digest(params)
            or selection.get("params") != params or not selection.get("effective_tiling")
            or selection.get("installed_sha256") != build["installed_sha256"]):
        raise TuningError("Invalid direct kernel correctness/timing/shape/source evidence")
    import statistics
    if result.get("duration_us") != statistics.median(samples):
        raise TuningError("Direct duration_us does not match sample P50")
    return result


def validate_task_result(result, config, params, build):
    profile = result.get("profile", {})
    event = result.get("event_diagnostics", {})
    if (result.get("timing_method") != "kernel_task_duration_us"
            or result.get("duration_us") != profile.get("duration_us")
            or result.get("samples_us") != profile.get("samples_us")
            or not profile.get("csv_files") or event.get("under_profiler") is not True):
        raise TuningError("Kernel search requires target-task profiler P50; Event fallback is forbidden")
    raw = {**result, "duration_us": event.get("duration_us"), "samples_us": event.get("samples_us", []),
           "timing_method": "device_event_us"}
    validate_result(raw, config, params, build)
    samples = result.get("samples_us", [])
    import statistics
    if (len(samples) != config["measurement"]["repeats"]
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)
            or result["duration_us"] != statistics.median(samples)):
        raise TuningError("Invalid target-task profiler samples/P50")
    return result


class KernelBackend(AscendBackend):
    def __init__(self, config, root, profile_search=True):
        super().__init__(config, root)
        self.profile_search = profile_search
        self.process, self.log, self.build = None, None, None
        self.om_backend = object.__new__(AscendBackend)
        self.om_backend.c = deepcopy(config)
        self.om_backend.c["implementation"]["execution"] = "om"
        self.om_backend.root, self.om_backend.case = self.root, self.case
        self.om_backend.source_builds = {}
        atexit.register(self.close)

    def wait_file(self, path):
        started = time.monotonic()
        while not path.is_file():
            if self.process.poll() is not None:
                raise TuningError(f"Direct launcher exited {self.process.returncode}; see {self.root/'direct-worker.log'}")
            if time.monotonic()-started > self.c["measurement"]["timeout_s"]:
                self.close()
                raise TuningError(f"Direct launcher timed out waiting for {path}; see direct-worker.log")
            time.sleep(0.05)
        return read_json(path)

    def start(self):
        prepare = self.root/"direct_prepare"
        prepare.mkdir()
        self.build = build_candidate(self.c, BATCH_DEFAULT_PARAMS.copy(), prepare, environment())
        if self.build.get("parameter_binding") != "atc_process_environment" or not self.build.get("kernel_catalogs"):
            raise TuningError("Kernel backend requires a cached runtime-parameter OPP with compiled device binaries")
        ready = prepare/"ready.json"
        request = prepare/"bootstrap.json"
        write_json(request, {"config": self.c, "build": self.build, "case": str(self.case), "ready": str(ready)})
        env = environment(self.build["opp_path"])
        # Set before loading CANN. Console-native output is captured locally;
        # never depend on the test machine's default ~/ascend/log location.
        env.update(ASCEND_SLOG_PRINT_TO_STDOUT="1", ASCEND_GLOBAL_LOG_LEVEL="1", PYTHONUNBUFFERED="1")
        self.log = (self.root/"direct-worker.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen([sys.executable, "-m", "paretopilot.direct_worker", str(request)],
                                        cwd=self.root, env=env,
                                        stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", start_new_session=os.name != "nt")
        status = self.wait_file(ready)
        if status.get("status") != "ready":
            raise TuningError(f"Direct launcher failed: {status.get('error')}; see direct-worker.log")

    def evaluate(self, params, directory, role):
        if role == "baseline":
            return self.om_backend.evaluate(params, directory, role)
        validate_params(params, batch=True)
        if self.process is None:
            self.start()
        if tree_hash(self.build["opp_path"]) != self.build["installed_sha256"]:
            raise TuningError("Direct source OPP changed during search")
        directory = Path(directory)
        request = {"params": params, "directory": str(directory)}
        if self.profile_search:
            request["profile_directory"] = str(directory/"profiler")
        self.process.stdin.write(json.dumps(request)+"\n")
        self.process.stdin.flush()
        result = self.wait_file(Path(directory)/"direct_result.json")
        if result.get("status") != "ok":
            raise TuningError(result.get("error", "Direct kernel failed"))
        validate_result(result, self.c, params, self.build)
        if self.profile_search:
            # Worker finalizes this capture before publishing direct_result.
            # No ATC or process restart is needed per search candidate.
            self.command([self.c["ascend"].get("msprof", "msprof"), "--export=on", "--output="+str(directory/"profiler")],
                         directory, "profiler_export", environment(self.build["opp_path"]))
            paths = sorted((directory/"profiler").rglob("op_summary*.csv"))
            if not paths:
                raise TuningError("Kernel search produced no op_summary CSV; Event fallback is forbidden; inspect profiler_export.log")
            write_json(directory/"profile_inventory.json", inspect_profiles(paths))
            profile = direct_profile(paths, self.c["measurement"]["repeats"],
                                     result["selection"]["effective_tiling"][0]["block_dim"])
            write_json(directory/"profile_metrics.json", profile)
            result = {**result, "event_diagnostics": {"duration_us": result["duration_us"],
                      "samples_us": result["samples_us"], "timing_method": "device_event_us", "under_profiler": True},
                      "duration_us": profile["duration_us"], "samples_us": profile["samples_us"],
                      "timing_method": "kernel_task_duration_us", "metric": profile["metric"], "profile": profile}
            validate_task_result(result, self.c, params, self.build)
            write_json(directory/"profile_result.json", result)
        # Reuse this immutable runtime-parameter Host package during OM
        # confirmation, while compiling a fresh OM for every confirmation trial.
        self.om_backend.source_builds[digest(params)] = {**deepcopy(self.build), "params": params.copy(), "params_sha256": digest(params)}
        write_json(Path(directory)/"selection.json", result["selection"])
        return result

    def close(self):
        process = self.process
        self.process = None
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.stdin.write('{"action":"close"}\n')
                        process.stdin.flush()
                        process.wait(timeout=5)
                    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                        if os.name != "nt":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait(timeout=10)
            finally:
                if process.stdin:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
        if self.log is not None:
            self.log.close()
            self.log = None
