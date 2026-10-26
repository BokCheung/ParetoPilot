"""Read-only checks; runtime imports are isolated from the controller process."""
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .official import enabled as official_enabled, verify_bundle, legacy_library
from .util import TuningError
from .workload import baseline_only


def inspect_environment(config):
    checks = []
    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    if config["backend"] == "simulation":
        return {"ready": all(x["passed"] for x in checks), "backend": "simulation", "checks": checks}
    add("platform", sys.platform.startswith("linux"), platform.platform())
    a = config["ascend"]
    commands = [a.get("atc", "atc"), a.get("msprof", "msprof"), "bash", "cmake", "g++", "make"]
    if baseline_only(config):
        commands = commands[:3]
    elif not official_enabled(config):
        commands.append(a.get("msopgen", "msopgen"))
    else:
        try:
            add("official_source", True, verify_bundle()["commit"])
        except (TuningError, OSError) as exc:
            add("official_source", False, str(exc))
        try:
            add("official_legacy_library", True, str(legacy_library(os.environ)[0]))
        except (TuningError, OSError) as exc:
            add("official_legacy_library", False, str(exc))
    for command in commands:
        path = shutil.which(command)
        add(command, path is not None, path or "Not on PATH; source CANN set_env.sh and install development tools")
    import_timeout = config["measurement"]["timeout_s"]
    for module in ("numpy", "torch", "onnx", "acl"):
        try:
            result = subprocess.run([sys.executable, "-c", f"import {module}; print(getattr({module}, '__version__', 'import OK'))"],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=import_timeout)
            add(module, result.returncode == 0, (result.stdout+result.stderr).strip()[-1500:])
        except subprocess.TimeoutExpired:
            add(module, False, f"Import timed out after {import_timeout} seconds using {sys.executable}. "
                "This does not establish a missing dependency. Increase measurement.timeout_s "
                "(scripts/test_910b.py: --timeout), or inspect the import in this environment.")
        except OSError as exc:
            add(module, False, str(exc))
    version_files = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            version_files.extend(Path(value).glob("version.info"))
            version_files.extend(Path(value).glob("*/ascend_toolkit_install.info"))
    versions = {str(p): p.read_text(errors="replace")[:4096] for p in set(version_files)}
    if versions:
        add("cann_version", any(a["cann_version"] in value for value in versions.values()), versions)
    else:
        add("cann_version", False, "Cannot verify CANN version; source set_env.sh and set ASCEND_HOME_PATH to the 8.5.0 toolkit root")
    info = "npu-smi not found"
    if shutil.which("npu-smi"):
        try:
            r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            info = (r.stdout+r.stderr).strip()[-8000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            info = str(exc)
    return {"ready": all(x["passed"] for x in checks), "backend": "ascend", "checks": checks,
            "import_timeout_s": import_timeout,
            "configured_soc": a["soc_version"], "configured_cann": a["cann_version"], "npu_smi": info,
            "note": "Check the configured SoC against npu-smi. Software checks do not prove a kernel compiles or runs."}
