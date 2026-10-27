"""Offline FP16/ND device binaries for the pinned 910B BatchMatMulV3."""
from pathlib import Path
import re

from .util import TuningError, read_json, write_json, file_hash, run_command


def enabled(config):
    return config.get("implementation", {}).get("execution", "om") == "kernel"


def kernel_build_environment(config, env):
    # build.sh normally exports this; our direct CMake target bypasses it.
    # Without it, the unquoted upstream $thread_num disappears from argv.
    jobs = config["implementation"].get("build_jobs", 4)
    if type(jobs) is not int or jobs < 1:
        raise TuningError("Kernel build_jobs must be a positive integer")
    return {**env, "ASCENDC_PER_COMPILE_JOB_THREAD": str(jobs)}


def fp16_config(config):
    """Limit upstream binary generation to our external ND/FP16 contract."""
    selected = []
    for item in config["op_list"]:
        inputs, outputs = item["inputs"], item["outputs"]
        if (all(x["dtype"] == "float16" and x["format"] == "ND" for x in inputs[:3])
                and all(x["dtype"] == "float16" and x["format"] == "ND" for x in outputs)
                and all(a["value"] is False for a in item["attrs"] if a["name"] in ("adj_x1", "adj_x2"))):
            selected.append(item)
    if len(selected) != 1:
        raise TuningError(f"Expected one official FP16/ND binary configuration, found {len(selected)}")
    return {**config, "op_list": selected}


def build_kernels(config, result, directory, env):
    build_env = kernel_build_environment(config, env)
    result["binary_build_threads"] = int(build_env["ASCENDC_PER_COMPILE_JOB_THREAD"])
    write_json(directory/"kernel-build-environment.json",
               {"ASCENDC_PER_COMPILE_JOB_THREAD": build_env["ASCENDC_PER_COMPILE_JOB_THREAD"]})
    project = Path(result["source_dir"])
    binary_config = project/"matmul/batch_mat_mul_v3/op_host/config/ascend910b/batch_mat_mul_v3_binary.json"
    write_json(binary_config, fp16_config(read_json(binary_config)))
    result["binary_recipe_sha256"] = file_hash(binary_config)
    # Upstream maps the generic compute unit to B1. Use the requested exact
    # 910B SKU for OPC while retaining ascend910b's source/config directories.
    soc = config["ascend"]["soc_version"]
    if re.fullmatch(r"Ascend910B[0-9]+", soc) is None:
        raise TuningError("Direct binary compilation requires an exact Ascend910B SKU, e.g. Ascend910B2")
    script = project/"scripts/kernel/binary_script/build_env.sh"
    text = script.read_text(encoding="utf-8")
    marker = '[Ascend910B]="Ascend910B1"'
    if text.count(marker) != 1:
        raise TuningError("Official binary SoC map changed; refusing an unverified target")
    script.write_text(text.replace(marker, '[Ascend910B]="'+soc+'"'), encoding="utf-8", newline="\n")
    result["binary_soc"] = soc
    build = project/"build"
    commands = [
        ("kernel-configure", ["cmake", "-S", str(project), "-B", str(build), "-DENABLE_BINARY=ON"]),
        # Do not build the all-operator `binary` or `package` targets here.
        ("kernel-build", ["cmake", "--build", str(build), "--target",
                          "config_compile_ascend910b_batch_mat_mul_v3", "--parallel",
                          str(config["implementation"].get("build_jobs", 4))]),
    ]
    for name, command in commands:
        record = run_command(command, project, directory/(name+".log"), config["measurement"]["timeout_s"], build_env)
        write_json(directory/(name+".command.json"), record)
    root = build/"binary/ascend910b/bin/ascend910b/batch_mat_mul_v3"
    catalogs = []
    for path in sorted(root.rglob("*.json")):
        meta = read_json(path)
        if not isinstance(meta, dict) or "binFileName" not in meta:
            continue
        binary = path.parent/(meta["binFileName"]+meta.get("binFileSuffix", ".o"))
        if not binary.is_file() or not meta.get("kernelList"):
            raise TuningError(f"Missing binary or explicit tiling-key kernelList: {path}")
        if meta.get("optionalInputMode", "gen_placeholder") != "gen_placeholder":
            raise TuningError(f"Unsupported optional-input ABI: {path}")
        catalogs.append({"metadata": str(path), "metadata_sha256": file_hash(path),
                         "binary": str(binary), "binary_sha256": file_hash(binary)})
    if len(catalogs) != 1:
        raise TuningError(f"Expected one FP16/ND kernel catalog at {root}, found {len(catalogs)}; see kernel-build.log")
    roots = [Path(env[k]) for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if env.get(k)]
    roots = [p for root in roots for p in (root, root/"aarch64-linux", root/"x86_64-linux")]
    include = next((p/"include" for p in roots if (p/"include/acl/acl.h").is_file()), None)
    lib = next((p/"lib64" for p in roots if (p/"lib64/libascendcl.so").is_file()), None)
    if include is None or lib is None:
        raise TuningError("Kernel timing bridge requires CANN include/acl/acl.h and lib64/libascendcl.so")
    launcher = directory/"libparetopilot_launch.so"
    command = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
               str(Path(__file__).parent/"official_adapter/direct_launcher.cpp"),
               "-I"+str(include), "-L"+str(lib), "-Wl,-rpath,"+str(lib), "-lascendcl", "-o", str(launcher)]
    record = run_command(command, project, directory/"launcher-build.log", config["measurement"]["timeout_s"], env)
    write_json(directory/"launcher-build.command.json", record)
    result.update(launcher_library=str(launcher), launcher_sha256=file_hash(launcher))
    result["kernel_catalogs"] = catalogs
    return result


def select_kernel(catalogs, tiling_key):
    matches = []
    for entry in catalogs:
        for name, hash_key in (("metadata", "metadata_sha256"), ("binary", "binary_sha256")):
            if file_hash(entry[name]) != entry[hash_key]:
                raise TuningError(f"Direct kernel cache changed: {entry[name]}")
        meta = read_json(entry["metadata"])
        for kernel in meta["kernelList"]:
            if int(kernel["tilingKey"]) == tiling_key:
                name = kernel.get("kernelName")
                if not isinstance(name, str) or not name:
                    raise TuningError("Kernel metadata has no explicit entry name")
                matches.append({**entry, **kernel, "entry": name})
    if len(matches) != 1:
        raise TuningError(f"Expected one compiled kernel for tiling key {tiling_key}, found {len(matches)}; no fallback to a different key")
    return matches[0]
