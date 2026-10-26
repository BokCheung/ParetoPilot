"""Immutable cross-run BatchMatMulV3 package cache and runtime Host controls."""
from contextlib import contextmanager
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

from . import official
from .util import TuningError, digest, file_hash, read_json, tree_hash, write_json


def runtime_environment(params):
    official.validate_params(params, batch=True)
    return {**{"PARETOPILOT_"+k.upper(): str(v) for k, v in params.items()},
            "PARETOPILOT_REQUEST_HASH": digest(params)}


def runtime_header():
    return r'''#pragma once
#include <cerrno>
#include <cstdlib>
#include <cstring>
namespace paretopilot {
struct RuntimeParams {
    int core_limit = 0;
    int tiling_policy = 0;
    int batch_strategy = 0;
    const char *hash = nullptr;
    bool valid = false;
    static bool Read(const char *name, int max, int &out) {
        const char *text = std::getenv(name);
        if (text == nullptr || *text == '\0') { return false; }
        char *end = nullptr;
        errno = 0;
        long value = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || value < 0 || value > max) { return false; }
        out = static_cast<int>(value);
        return true;
    }
    RuntimeParams() {
        hash = std::getenv("PARETOPILOT_REQUEST_HASH");
        bool hashValid = hash != nullptr && std::strlen(hash) == 64;
        if (hashValid) {
            for (int i = 0; i < 64; ++i) {
                if (!((hash[i] >= '0' && hash[i] <= '9') || (hash[i] >= 'a' && hash[i] <= 'f'))) { hashValid = false; }
            }
        }
        valid = hashValid && Read("PARETOPILOT_CORE_LIMIT", 24, core_limit)
            && Read("PARETOPILOT_TILING_POLICY", 1, tiling_policy)
            && Read("PARETOPILOT_BATCH_STRATEGY", 1, batch_strategy);
    }
};
inline const RuntimeParams &Params() { static const RuntimeParams params; return params; }
}
#define PP_CORE_LIMIT (paretopilot::Params().core_limit)
#define PP_TILING_POLICY (paretopilot::Params().tiling_policy)
#define PP_BATCH_STRATEGY (paretopilot::Params().batch_strategy)
#define PP_PARAMS_HASH (paretopilot::Params().hash)
#define PP_PARAMS_VALID (paretopilot::Params().valid)
''' + f'#define PP_SOURCE_COMMIT "{official.COMMIT}"\n'


def cache_identity(config, env):
    manifest = official.verify_bundle()
    legacy, _ = official.legacy_library(env)
    paths = [Path(__file__), Path(official.__file__), *sorted((Path(__file__).parent/"official_adapter").glob("*"))]
    tools = {}
    for name in ("cmake", "g++"):
        path = shutil.which(name, path=env.get("PATH"))
        if not path:
            raise TuningError(f"Build-cache identity requires {name} on PATH")
        tools[name] = {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
    roots = {key: str(Path(env[key]).resolve()) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH") if env.get(key)}
    versions = {}
    for root in set(roots.values()):
        for path in [Path(root)/"version.info", *Path(root).glob("*/ascend_toolkit_install.info")]:
            if path.is_file():
                versions[str(path)] = file_hash(path)
    return {"format": 1, "kind": official.BATCH_KIND, "archives": manifest["archives"],
            "soc": config["ascend"]["soc_version"], "cann": config["ascend"]["cann_version"],
            "machine": platform.machine(), "python": sys.executable, "python_version": sys.version,
            "environment_paths": roots, "versions": versions, "legacy_sha256": file_hash(legacy),
            "adapter_files": {str(p): file_hash(p) for p in paths if p.is_file()}, "tools": tools}


@contextmanager
def build_lock(path, timeout):
    """Kernel-held lock: automatically released after a killed builder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        started = time.monotonic()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic()-started >= timeout:
                    raise TuningError(f"Timed out waiting for source cache lock: {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def valid_snapshot(result, entry):
    try:
        opp = Path(result["opp_path"])
        legacy = Path(result["isolated_legacy_library"])
        return (result.get("parameter_binding") == "atc_process_environment"
                and opp.is_dir() and legacy.is_file()
                and opp.resolve().is_relative_to(entry.resolve())
                and legacy.resolve().is_relative_to(entry.resolve())
                and tree_hash(opp) == result["installed_sha256"]
                and file_hash(legacy) == result["legacy_library_sha256"])
    except (OSError, KeyError, TypeError, AttributeError, TuningError):
        return False


def cached_build(config, params, directory, env):
    official.validate_params(params, batch=True)
    identity = cache_identity(config, env)
    key = digest(identity)
    configured = config["implementation"].get("cache_dir")
    root = Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1]/".cache/official-builds"
    entry = root/key
    entry.mkdir(parents=True, exist_ok=True)
    cache_hit = False
    with build_lock(root/(key+".lock"), config["measurement"]["timeout_s"]):
        result = None
        if (entry/"complete.json").is_file():
            try:
                existing = read_json(entry/"complete.json")
                if isinstance(existing, dict) and existing.get("cache_identity") == identity and valid_snapshot(existing, entry):
                    result, cache_hit = existing, True
            except TuningError:
                pass
        if result is None:
            generation = Path(tempfile.mkdtemp(prefix="build-", dir=entry))
            print(f"Source cache miss: building one runtime-parameter OPP at {generation}", flush=True)
            try:
                result = official._build_official(config, official.BATCH_DEFAULT_PARAMS.copy(), generation, env, runtime=True)
                if not valid_snapshot(result, entry):
                    raise TuningError("Built source package failed cache integrity validation")
                result.update(cache_identity=identity, cache_key=key)
                write_json(entry/"complete.json", result)
            finally:
                # Keep the absolute build workspace stable; generated files may
                # contain these paths. Never move a configured CMake project.
                for path in generation.glob("*.log"):
                    target = directory/"cache-build-logs"/path.name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(path, target)
        else:
            print(f"Source cache hit: {key[:12]}; no configure/build/package/install", flush=True)
        result = {**result, "params": params.copy(), "params_sha256": digest(params),
                  "cache_hit": cache_hit, "cache_key": key}
        write_json(directory/"source_manifest.json", result)
        write_json(directory/"build_manifest.json", result)
        write_json(directory/"params.json", params)
        return result
