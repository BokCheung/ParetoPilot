"""Pinned experimental AddV2 from CANN ops-math 8.5.0, not stock Add."""
import json
import os
import platform
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

from .util import TuningError, digest, file_hash, read_json, run_command, tree_hash, write_json

COMMIT = "f1a42c5381bbd33981877ad87751d242663c2ce6"
DEFAULT_PARAMS = {"core_limit": 0, "tile_elems": 0}
SOURCE = "experimental/math/add_v2"
VENDOR = Path(__file__).parent/"vendor/cann_8_5_0"


def validate_params(params):
    if (set(params) != set(DEFAULT_PARAMS) or any(type(v) is not int for v in params.values())
            or not 0 <= params["core_limit"] <= 48 or not 0 <= params["tile_elems"] <= 32768
            or params["tile_elems"] % 16):
        raise TuningError("AddV2 requires core_limit=0..48 and tile_elems=0 (upstream) or a multiple of 16 <=32768")


def runtime_environment(params):
    validate_params(params)
    return {"PARETOPILOT_CORE_LIMIT": str(params["core_limit"]), "PARETOPILOT_TILE_ELEMS": str(params["tile_elems"]),
            "PARETOPILOT_REQUEST_HASH": digest(params)}


def verify_bundle():
    manifest = read_json(VENDOR/"add_manifest.json")
    if manifest.get("commit") != COMMIT or set(manifest.get("archives", {})) != {"ops-math-add-v2.zip"}:
        raise TuningError("AddV2 source manifest mismatch")
    for name, expected in manifest["archives"].items():
        if file_hash(VENDOR/name) != expected:
            raise TuningError("AddV2 official archive checksum mismatch")
    return manifest


def materialize(project):
    from .official import extract, normalize_shell_scripts, verify_bundle as verify_deps
    verify_bundle()
    verify_deps()  # Existing pinned offline json/makeself dependency bundle.
    project = Path(project)
    project.mkdir(parents=True)
    extract(VENDOR/"ops-math-add-v2.zip", project)
    extract(VENDOR/"build-deps.zip", project)
    device_hash = tree_hash(project/SOURCE/"op_kernel")
    changes = normalize_shell_scripts(project)
    def replace(path, old, new):
        file = project/path
        source = file.read_text(encoding="utf8")
        if source.count(old) != 1:
            raise TuningError(f"Pinned AddV2 patch anchor changed: {path}: {old[:60]}")
        before = file_hash(file)
        file.write_text(source.replace(old, new), encoding="utf8", newline="\n")
        changes.append({"path": path, "before_sha256": before, "after_sha256": file_hash(file)})
    for dep in ("eigen", "abseil-cpp", "ascend_protobuf"):
        replace("CMakeLists.txt", f"include(cmake/third_party/{dep}.cmake)", f"# Unneeded for isolated AddV2 graph parser: {dep}")
    replace("CMakeLists.txt", "  add_subdirectory(experimental)", "  add_subdirectory(experimental/math/add_v2)")
    common = project/"common/CMakeLists.txt"
    text = common.read_text(encoding="utf8")
    old = text[text.index("file(GLOB CURRENT_DIRS"):]
    replace("common/CMakeLists.txt", old, "# ParetoPilot provides an Operator-based parser; no generic protobuf framework.\n")
    replace("cmake/third_party/json.cmake", 'else()\n    set(REQ_URL', 'else()\n    message(FATAL_ERROR "Bundled JSON headers missing; network disabled")\n    set(REQ_URL')
    replace("cmake/third_party/makeself-fetch.cmake", '    set(MAKESELF_URL ', '    message(FATAL_ERROR "Bundled makeself missing; network disabled")\n    set(MAKESELF_URL ')
    host = SOURCE+"/op_host/add_v2_tiling.cpp"
    header = r'''#pragma once
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstdio>
inline int PpAddParam(const char *name, int maximum) {
    const char *s = std::getenv(name); if (!s) return -1;
    char *end = nullptr; errno = 0; long value = std::strtol(s, &end, 10);
    return errno || end == s || *end || value < 0 || value > maximum ? -1 : static_cast<int>(value);
}
'''
    (project/SOURCE/"op_host/paretopilot_add_params.h").write_text(header, encoding="utf8")
    replace(host, '#include "log/log.h"', '#include "log/log.h"\n#include "paretopilot_add_params.h"')
    replace(host, "    // 2. shapes & dtype", '''    const int ppCore = PpAddParam("PARETOPILOT_CORE_LIMIT", 48);
    const int ppTile = PpAddParam("PARETOPILOT_TILE_ELEMS", 32768);
    const char *ppHash = std::getenv("PARETOPILOT_REQUEST_HASH");
    if (ppCore < 0 || ppTile < 0 || ppTile % 16 || !ppHash || std::strlen(ppHash) != 64 || ppCore > coreNum) return ge::GRAPH_FAILED;
    const auto ppPhysicalCores = coreNum;
    if (ppCore) coreNum = ppCore;
    // 2. shapes & dtype''')
    replace(host, "    // 每个 tile 包含的元素数（至少 1）", '''    // Constrain the official tile input BEFORE partition/tail dependencies.
    if (ppTile) {
        const uint64_t requestedBlocks = static_cast<uint64_t>(ppTile) * inputBytes / BLOCK_SIZE;
        if (!requestedBlocks || requestedBlocks > tileBlockNum) return ge::GRAPH_FAILED;
        tileBlockNum = static_cast<uint32_t>(requestedBlocks);
    }
    // 每个 tile 包含的元素数（至少 1）''')
    replace(host, "    return ge::GRAPH_SUCCESS;\n}\n\n\nstatic ge::graphStatus TilingParseForAddV2", '''    std::fprintf(stderr, "PARETOPILOT_PARAMS=%s\\nPARETOPILOT_OFFICIAL='''+COMMIT+'''\\n", ppHash);
    std::fprintf(stderr, "PARETOPILOT_ADD_TILING={\\"core_limit\\":%d,\\"tile_elems\\":%d,\\"physical_cores\\":%lld,\\"block_dim\\":%u,\\"used_core_num\\":%u,\\"tiling_key\\":%llu,\\"tile_data_num\\":%u,\\"small_core_data_num\\":%u,\\"big_core_data_num\\":%u,\\"small_tiles\\":%u,\\"big_tiles\\":%u,\\"tail_block_num\\":%u,\\"ub_size\\":%llu}\\n",
        ppCore, ppTile, static_cast<long long>(ppPhysicalCores), finalCoreNum, finalCoreNum,
        static_cast<unsigned long long>(tilingKey), tileDataNum, smallCoreDataNum, bigCoreDataNum,
        finalSmallTileNum, finalBigTileNum, tailBlockNum, static_cast<unsigned long long>(ubSize));
    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParseForAddV2''')
    infer = SOURCE+"/op_host/add_v2_infershape.cpp"
    replace(infer, "IMPL_OP_INFERSHAPE(AddV2).InferShape(InferShapeAddV2);", "static ge::graphStatus PpInferType(gert::InferDataTypeContext* c) { return c->SetOutputDataType(0, c->GetInputDataType(0)); }\nIMPL_OP_INFERSHAPE(AddV2).InferShape(InferShapeAddV2).InferDataType(PpInferType);")
    parser = '#include "register/register.h"\nnamespace domi { Status PpParseAdd(const ge::Operator&, ge::Operator&) { return SUCCESS; }\nREGISTER_CUSTOM_OP("AddV2").FrameworkType(ONNX).OriginOpType("ai.onnx::13::ParetoPilotOfficialAddV2").ParseParamsByOperatorFn(PpParseAdd).ImplyType(ImplyType::TVM); }\n'
    (project/"parser.cpp").write_text(parser, encoding="utf8")
    shutil.copyfile(Path(__file__).parent/"official_adapter/parser.cmake", project/"parser.cmake")
    with (project/"CMakeLists.txt").open("a", encoding="utf8") as out:
        out.write("\ninclude(parser.cmake)\n")
    if tree_hash(project/SOURCE/"op_kernel") != device_hash:
        raise TuningError("Official AddV2 Device source changed")
    return {**verify_bundle(), "operator": "AddV2", "commit": COMMIT, "patches": changes, "device_source_sha256": device_hash}


def read_effective_tiling(path, params):
    validate_params(params)
    text = Path(path).read_text(encoding="utf8", errors="replace")
    if "PARETOPILOT_OFFICIAL="+COMMIT not in text or "PARETOPILOT_PARAMS="+digest(params) not in text:
        raise TuningError("Pinned AddV2 Host source/parameter signature missing")
    records = [json.loads(s) for s in re.findall(r"PARETOPILOT_ADD_TILING=(\{[^\r\n]*\})", text)]
    if not records:
        raise TuningError("AddV2 effective tiling missing")
    for r in records:
        if (any(r.get(k) != v for k, v in params.items()) or not 1 <= r.get("block_dim", 0) <= r.get("physical_cores", 0) <= 48
                or r.get("tiling_key") != 3 or r.get("tile_data_num", 0) <= 0
                or (params["core_limit"] and r["block_dim"] > params["core_limit"])
                or (params["tile_elems"] and r["tile_data_num"] != params["tile_elems"])):
            raise TuningError("AddV2 effective tiling disagrees with requested controls")
    return records


def validate_tiling_bytes(info, records, numel):
    data = bytes.fromhex(info["tiling_data"])
    if len(data) != 64:
        raise TuningError("AddV2 tiling ABI requires exactly eight int64 fields")
    small, big, big_tiles, small_tiles, tile, small_tail, big_tail, extra = struct.unpack("<8q", data)
    blocks = info["block_dim"]
    if (info["tiling_key"] != 3 or not 0 <= extra < blocks or tile <= 0 or tile % 16
            or min(small, big, big_tiles, small_tiles, small_tail, big_tail) <= 0
            or small*(blocks-extra)+big*extra != numel
            or small != (small_tiles-1)*tile+small_tail or big != (big_tiles-1)*tile+big_tail
            or max(small_tail, big_tail) > tile):
        raise TuningError("AddV2 binary tiling partition/loop/tail contract failed")
    observed = {"small_core_data_num": small, "big_core_data_num": big, "big_tiles": big_tiles,
                "small_tiles": small_tiles, "tile_data_num": tile, "tail_block_num": extra}
    if any(any(r.get(k) != v for k, v in observed.items()) or tile*12 > r.get("ub_size", 0) for r in records):
        raise TuningError("AddV2 raw tiling differs from Host evidence or exceeds UB")


def _build(config, directory, env):
    from .official import verify_installed_implementations
    if re.fullmatch(r"Ascend910B[0-9]+", config["ascend"]["soc_version"]) is None:
        raise TuningError("Official AddV2 compilation requires an exact 910B SKU, e.g. Ascend910B2")
    project = directory/"project"
    provenance = materialize(project)
    write_json(directory/"source_manifest.json", provenance)
    build_dir = project/"build"
    jobs = str(config["implementation"].get("build_jobs", 4))
    timeout = config["measurement"]["timeout_s"]
    script = project/"scripts/kernel/binary_script/build_env.sh"
    text = script.read_text(encoding="utf8")
    if text.count('[Ascend910B]="Ascend910B1"') != 1:
        raise TuningError("AddV2 OPC SoC mapping changed")
    script.write_text(text.replace('[Ascend910B]="Ascend910B1"', '[Ascend910B]="'+config["ascend"]["soc_version"]+'"'), encoding="utf8", newline="\n")
    commands = [("configure", ["cmake", "-S", str(project), "-B", str(build_dir), "-DENABLE_CUSTOM=ON", "-DENABLE_PACKAGE=ON", "-DENABLE_EXPERIMENTAL=ON", "-DENABLE_BINARY=OFF", "-DASCEND_COMPUTE_UNIT=ascend910b", "-DASCEND_OP_NAME=add_v2", "-DVENDOR_NAME=paretopilot", "-DCMAKE_INSTALL_PREFIX="+str(project/"build_out")]),
                ("generate", ["cmake", "--build", str(build_dir), "--target", "ascendc_impl_gen", "--parallel", jobs]),
                ("binary-recipe", ["cmake", "--build", str(build_dir), "--target", "gen_bin_scripts", "--parallel", jobs]),
                ("configure-fp16", []),
                ("kernel-build", ["cmake", "--build", str(build_dir), "--target", "config_compile_ascend910b_add_v2", "--parallel", jobs]),
                ("build", ["cmake", "--build", str(build_dir), "--target", "package", "paretopilot_onnx", "--parallel", jobs])]
    env = {**env, **runtime_environment(DEFAULT_PARAMS), "ASCENDC_PER_COMPILE_JOB_THREAD": jobs}
    for name, argv in commands:
        if name == "configure-fp16":
            argv = commands[0][1]
        record = run_command(argv, project, directory/(name+".log"), timeout, env)
        write_json(directory/(name+".command.json"), record)
        if name == "binary-recipe":
            # Preserve all compiler-generated keys/attributes; restrict only
            # the external tensor contract, never invent a simplified key.
            from .direct_build import fp16_config
            generated = build_dir/"binary/ascend910b/gen/add_v2/add_v2_binary.json"
            recipe = fp16_config(read_json(generated))
            write_json(project/SOURCE/"op_host/config/ascend910b/add_v2_binary.json", recipe)
    packages = sorted({*project.glob("build_out/*.run"), *build_dir.glob("*.run"), *build_dir.glob("**/cann-ops-math*.run")})
    if len(packages) != 1:
        raise TuningError(f"Expected one AddV2 OPP package, found {packages}; inspect build.log")
    install = directory/"opp"
    install.mkdir()
    run_command(["bash", str(packages[0]), "--install-path="+str(install)], project, directory/"install.log", timeout, env)
    vendors = list((install/"vendors").iterdir())
    if len(vendors) != 1:
        raise TuningError("AddV2 requires exactly one isolated vendor")
    adapters = list((build_dir/"tbe/dynamic").glob("*.py"))
    if not any(p.name == "add_v2.py" for p in adapters):
        raise TuningError("AddV2 compiler adapter was not generated")
    verify_installed_implementations(adapters, vendors[0])
    parser = vendors[0]/"framework/onnx"
    parser.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build_dir/"libparetopilot_onnx.so", parser/"libparetopilot_onnx.so")
    result = {**provenance, "source_dir": str(project), "opp_path": str(vendors[0]), "package": str(packages[0]),
              "installed_sha256": tree_hash(vendors[0]), "parameter_binding": "atc_process_environment"}
    from .direct_build import collect_catalogs, build_launcher
    catalogs = collect_catalogs(build_dir/"binary/ascend910b/bin/ascend910b/add_v2")
    result["kernel_catalogs"] = catalogs
    return build_launcher(config, result, directory, env)


def build(config, params, directory, env):
    from .official_cache import build_lock
    validate_params(params)
    identity = {"format": 1, "manifest": verify_bundle(), "soc": config["ascend"]["soc_version"],
                "cann": config["ascend"]["cann_version"], "paths": {k: env.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH")},
                "machine": platform.machine(), "python": sys.executable, "python_version": sys.version,
                "dependencies": file_hash(VENDOR/"build-deps.zip"),
                "direct_builder": file_hash(Path(__file__).with_name("direct_build.py")),
                "adapter": file_hash(__file__), "parser": file_hash(Path(__file__).parent/"official_adapter/parser.cmake"),
                "launcher": file_hash(Path(__file__).parent/"official_adapter/direct_launcher.cpp")}
    identity["tools"] = {}
    for name in ("cmake", "g++"):
        tool = shutil.which(name, path=env.get("PATH"))
        if not tool:
            raise TuningError(f"AddV2 build requires {name}")
        identity["tools"][name] = {"path": tool, "sha256": file_hash(tool)}
    identity["versions"] = {str(p): file_hash(p) for root in {v for v in identity["paths"].values() if v}
                            for p in [Path(root)/"version.info", *Path(root).glob("*/ascend_toolkit_install.info")] if p.is_file()}
    key = digest(identity)
    root = Path(config["implementation"].get("cache_dir", Path(__file__).resolve().parents[1]/".cache/official-add-builds"))
    entry = root/key
    entry.mkdir(parents=True, exist_ok=True)
    with build_lock(root/(key+".lock"), config["measurement"]["timeout_s"]):
        result, hit = None, False
        if (entry/"complete.json").is_file():
            try:
                old = read_json(entry/"complete.json")
                if old.get("cache_identity") == identity and tree_hash(old["opp_path"]) == old["installed_sha256"]:
                    from .direct_build import select_kernel
                    select_kernel(old["kernel_catalogs"], 3)
                    if file_hash(old["launcher_library"]) == old["launcher_sha256"]:
                        result, hit = old, True
            except (TuningError, OSError, KeyError, TypeError):
                result = None
        if result is None:
            generation = Path(tempfile.mkdtemp(prefix="build-", dir=entry))
            print(f"AddV2 source cache miss: {generation}", flush=True)
            try:
                result = _build(config, generation, env)
            finally:
                for log in generation.glob("*.log"):
                    target = directory/"cache-build-logs"/log.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(log, target)
            result.update(cache_identity=identity, cache_key=key)
            write_json(entry/"complete.json", result)
        else:
            print(f"AddV2 source cache hit: {key[:12]}", flush=True)
        result = {**result, "params": params.copy(), "params_sha256": digest(params), "cache_hit": hit}
        write_json(directory/"build_manifest.json", result)
        return result
