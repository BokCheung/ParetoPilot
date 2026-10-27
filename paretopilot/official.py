"""Pinned CANN MatMulV3 sources, offline build adapter, and tiling controls.

The device implementation is upstream code. Patches cover build/graph
integration and validated controls for the original Host tiling process.
"""
import json
from pathlib import Path
import platform
import re
import shutil
import stat
import zipfile

from .util import TuningError, digest, file_hash, read_json, run_command, tree_hash, write_json

COMMIT = "9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef"
KIND = "cann_matmul_v3"
BATCH_KIND = "cann_batch_matmul_v3"
DEFAULT_PARAMS = {"core_limit": 0, "tiling_policy": 0}
BATCH_DEFAULT_PARAMS = {**DEFAULT_PARAMS, "batch_strategy": 0}
TILE_DEFAULT_PARAMS = {"base_m": 0, "base_n": 0}
VENDOR = Path(__file__).parent/"vendor/cann_8_5_0"
HOST = "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp"
INFER = "matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp"
IMPL_CMAKE = "cmake/gen_ops_info.cmake"
BATCH_HOST = "matmul/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.cpp"
BATCH_INFER = "matmul/batch_mat_mul_v3/op_host/batch_mat_mul_v3_infershape.cpp"


def enabled(config):
    return config.get("implementation", {}).get("kind") in (KIND, BATCH_KIND)


def batch_enabled(config):
    return config.get("implementation", {}).get("kind") == BATCH_KIND


def default_params(config):
    params = (BATCH_DEFAULT_PARAMS if batch_enabled(config) else DEFAULT_PARAMS).copy()
    if batch_enabled(config) and "base_m" in config["space"]:
        params.update(TILE_DEFAULT_PARAMS)
    return params


def legacy_library(env):
    """Upstream's custom-OPP loader resolves this library relative to its OPP root."""
    roots = []
    if env.get("ASCEND_OPP_PATH"):
        roots.append(Path(env["ASCEND_OPP_PATH"]))
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if env.get(key):
            root = Path(env[key])
            roots.extend([root/"opp", *root.glob("*-linux/opp")])
    arch = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
    for root in roots:
        path = root/f"built-in/op_impl/ai_core/tbe/op_host/lib/linux/{arch}/libophost_comm_legacy.so"
        if path.is_file():
            return path, path.relative_to(root)
    raise TuningError("Official MatMul requires CANN 8.5.0 ops libophost_comm_legacy.so under ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/op_host/lib/linux/<arch>; check the installed 910B ops package")


def validate_params(params, batch=False):
    expected = set(BATCH_DEFAULT_PARAMS if batch else DEFAULT_PARAMS)
    if (set(params) not in (expected, expected | set(TILE_DEFAULT_PARAMS) if batch else expected) or any(type(v) is not int for v in params.values())
            or not 0 <= params["core_limit"] <= 24 or params["tiling_policy"] not in (0, 1)
            or (batch and params["batch_strategy"] not in (0, 1))):
        raise TuningError("Official MatMul requires core_limit=0..24 and tiling_policy=0 (upstream) or 1 (heuristic)")
    if batch and "base_m" in params:
        m, n = params["base_m"], params["base_n"]
        if ((m == 0) != (n == 0) or any(v < 0 or v > 256 or v % 16 for v in (m, n))
                or (m and (params["tiling_policy"] != 1 or params["batch_strategy"] != 1))):
            raise TuningError("base_m/base_n must both be 0 (auto) or 16-aligned values <=256; explicit tiles require tiling_policy=1 and batch_strategy=1")


def verify_bundle():
    manifest = read_json(VENDOR/"manifest.json")
    if manifest["commit"] != COMMIT:
        raise TuningError("Official MatMul source commit mismatch")
    for name, expected in manifest["archives"].items():
        if file_hash(VENDOR/name) != expected:
            raise TuningError(f"Official source archive checksum mismatch: {name}")
    if set(manifest["archives"]) != {"ops-nn-matmul.zip", "build-deps.zip"}:
        raise TuningError("Incomplete official source bundle")
    return manifest


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = destination/item.filename
            if not path.resolve().is_relative_to(destination.resolve()):
                raise TuningError("Invalid source archive path")
            source.extract(item, destination)
            if not item.is_dir() and item.filename.endswith(".sh"):
                path.chmod(path.stat().st_mode | stat.S_IXUSR)


def normalize_shell_scripts(project):
    """Adapt extracted Shell scripts without decoding or changing archives."""
    changes = []
    root = Path(project)
    for path in sorted(root.rglob("*.sh")):
        original = path.read_bytes()
        normalized = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized != original:
            before = file_hash(path)
            path.write_bytes(normalized)
            changes.append({"path": path.relative_to(root).as_posix(), "before_sha256": before,
                            "after_sha256": file_hash(path), "reason": "shell_line_endings_lf"})
    return changes


def materialize(project, params, *, batch=False, runtime=False):
    """Extract verified originals and record every adaptation with before/after hashes."""
    validate_params(params, batch=batch)
    if runtime and not batch:
        raise TuningError("Runtime Host controls currently require BatchMatMulV3")
    manifest = verify_bundle()
    project = Path(project)
    project.mkdir(parents=True, exist_ok=False)
    for name in manifest["archives"]:
        extract(VENDOR/name, project)
    changes = normalize_shell_scripts(project)

    def replace(path, old, new):
        file = project/path
        source = file.read_text(encoding="utf-8")
        if source.count(old) != 1:
            raise TuningError(f"Pinned upstream patch anchor changed: {path}: {old[:80]}")
        before = file_hash(file)
        file.write_text(source.replace(old, new), encoding="utf-8", newline="\n")
        changes.append({"path": path, "before_sha256": before, "after_sha256": file_hash(file)})

    # ATC uses a small ParseParamsByOperator parser; no protobuf compiler, Eigen,
    # or upstream generic ONNX parser is needed by this FP16 ND MatMul build.
    for dep in ("abseil-cpp", "ascend_protobuf", "eigen"):
        replace("CMakeLists.txt", f"include(cmake/third_party/{dep}.cmake)",
                f"# ParetoPilot: {dep} is not needed by the selected ATC path.")
    replace("common/CMakeLists.txt", "add_subdirectory(src/framework)",
            "# ParetoPilot supplies an Operator-based ONNX parser without protobuf.")
    replace("cmake/third_party/nlohmann_json.cmake", "else()\n    include(ExternalProject)",
            'else()\n    message(FATAL_ERROR "Bundled nlohmann/json headers missing; network downloads disabled")\n    include(ExternalProject)')
    replace("cmake/third_party/makeself-fetch.cmake", '    set(MAKESELF_URL ',
            '    message(FATAL_ERROR "Bundled makeself missing; network downloads disabled")\n    set(MAKESELF_URL ')

    # These Python adapters do not exist at configure time. Upstream's GLOB
    # freezes an empty file list on a fresh build, silently omitting dynamic/.
    # Enumerate at install/package time, after ascendc_impl_gen has run.
    replace(IMPL_CMAKE, """  file(GLOB dynamic_impl ${OPIMPL_OUT_DIR}/dynamic/*.py)
  if(ENABLE_PACKAGE)
    install(
      FILES ${dynamic_impl}
      DESTINATION ${OPIMPL_INSTALL_DIR}
      OPTIONAL
    )
  endif()""", """  if(ENABLE_PACKAGE)
    install(
      DIRECTORY "${OPIMPL_OUT_DIR}/dynamic/"
      DESTINATION ${OPIMPL_INSTALL_DIR}
      FILES_MATCHING PATTERN "*.py"
    )
  endif()""")

    # The pinned MatMulV3 registers only shape inference. Its custom OPP needs
    # dtype inference for ATC; y has x1's dtype in the upstream OpDef. Extend the
    # same registration (as upstream GemmV2 does), preserving shape inference.
    replace(INFER, "} // namespace\nnamespace ops {", """// ParetoPilot: supply the graph-mode dtype rule for the isolated custom OPP.
static ge::graphStatus InferDataTypeForMatMulV3(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

} // namespace
namespace ops {""")
    replace(INFER, "IMPL_OP_INFERSHAPE(MatMulV3).InferShape(InferShapeForMatMulV3);",
            "IMPL_OP_INFERSHAPE(MatMulV3)\n        .InferShape(InferShapeForMatMulV3)\n        .InferDataType(InferDataTypeForMatMulV3);")

    replace(HOST, '#include "matmul_v3_base_tiling.h"',
            '#include "matmul_v3_base_tiling.h"\n#include "paretopilot_params.h"\n#include <cstdio>')
    replace(HOST, "    if (compileInfo_.aicNum == 0) {", """    // ParetoPilot: change the input core budget BEFORE upstream derives tiling.
    if (PP_CORE_LIMIT > 0) {
        if (static_cast<uint64_t>(PP_CORE_LIMIT) > compileInfo_.aicNum) {
            OP_LOGE(context_->GetNodeName(), "ParetoPilot core_limit exceeds physical AI Core count");
            return ge::GRAPH_FAILED;
        }
        compileInfo_.aicNum = PP_CORE_LIMIT;
    }
    if (compileInfo_.aicNum == 0) {""")
    replace(HOST, "    if (GetTilingFromRepo()) {", "    if (PP_TILING_POLICY == 0 && GetTilingFromRepo()) {")
    # Explicit tiles enter the official API before any dependent dimensions
    # are calculated. The automatic path remains byte-for-byte at these anchors.
    replace(HOST, "    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {", """    if (PP_EXPLICIT_TILES) {
        if (std::string(context_->GetNodeType()) != "BatchMatMulV3" ||
            PP_TILING_POLICY != 1 || !compileInfo_.supportL0c2out ||
            args_.aType != ge::DT_FLOAT16 || args_.bType != ge::DT_FLOAT16 ||
            args_.aFormat != ge::FORMAT_ND || args_.bFormat != ge::FORMAT_ND ||
            args_.outFormat != ge::FORMAT_ND || args_.isATrans || args_.isBTrans || args_.hasBias ||
            args_.kValue > 65535UL || args_.nValue > 65535UL ||
            PP_BASE_M > ops::CeilAlign(args_.mValue, 16UL) ||
            PP_BASE_N > ops::CeilAlign(args_.nValue, 16UL) ||
            static_cast<uint64_t>(PP_BASE_M) * PP_BASE_N * 4UL > compileInfo_.l0CSize ||
            mm_.SetFixSplit(PP_BASE_M, PP_BASE_N, -1) == -1) {
            OP_LOGE(args_.opName, "Unsupported ParetoPilot explicit baseM/baseN; baseK must remain automatic");
            return ge::GRAPH_FAILED;
        }
    }
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {""")
    replace(HOST, "    SetParamsV310();", """    if (PP_EXPLICIT_TILES) {
        const auto &ppApi = tilingData_.matmulTiling;
        if (ppApi.baseM != PP_BASE_M || ppApi.baseN != PP_BASE_N || ppApi.baseK == 0) {
            OP_LOGE(args_.opName, "Official API did not honor explicit baseM/baseN");
            return ge::GRAPH_FAILED;
        }
        runInfo_.baseM = ppApi.baseM;
        runInfo_.baseN = ppApi.baseN;
        runInfo_.baseK = ppApi.baseK;
        runInfo_.usedCoreNum = ppApi.usedCoreNum;
        // Keep the API's complete TCubeTiling; no KB/basic/split-K rewrite.
        runInfo_.needUpdate = false;
        DoNd2NzVectorTiling();
        SetNd2NzInfo();
        return ge::GRAPH_SUCCESS;
    }
    SetParamsV310();""")
    replace(HOST, "    workspaces[0] = workspaceSize_;", """    workspaces[0] = workspaceSize_;
    std::fprintf(stderr, "PARETOPILOT_PARAMS=%s\\nPARETOPILOT_OFFICIAL=%s\\n", PP_PARAMS_HASH, PP_SOURCE_COMMIT);
    std::fprintf(stderr, "PARETOPILOT_TILING={\\\"core_limit\\\":%d,\\\"tiling_policy\\\":%d,\\\"used_core_num\\\":%u,\\\"base_m\\\":%u,\\\"base_n\\\":%u,\\\"base_k\\\":%u,\\\"tiling_key\\\":%llu}\\n",
        PP_CORE_LIMIT, PP_TILING_POLICY, tilingData_.matmulTiling.usedCoreNum,
        tilingData_.matmulTiling.baseM, tilingData_.matmulTiling.baseN, tilingData_.matmulTiling.baseK,
        static_cast<unsigned long long>(tilingKey_));""")
    header = "#pragma once\n" + ''.join(f"#define PP_{k.upper()} {v}\n" for k, v in sorted(params.items()))
    header += ''.join(f"#define PP_{k.upper()} {v}\n" for k, v in TILE_DEFAULT_PARAMS.items() if k not in params)
    header += '#define PP_EXPLICIT_TILES (PP_BASE_M > 0 && PP_BASE_N > 0)\n'
    header += f'#define PP_PARAMS_HASH "{digest(params)}"\n#define PP_SOURCE_COMMIT "{COMMIT}"\n'
    if runtime:
        from .official_cache import runtime_header
        header = runtime_header()
        replace(HOST, "    // ParetoPilot: change the input core budget BEFORE upstream derives tiling.", """    if (!PP_PARAMS_VALID) {
        OP_LOGE(context_->GetNodeName(), "Missing or invalid ParetoPilot runtime Host parameters");
        return ge::GRAPH_FAILED;
    }
    // ParetoPilot: change the input core budget BEFORE upstream derives tiling.""")
    (project/HOST).with_name("paretopilot_params.h").write_text(header, encoding="utf-8")
    if batch:
        replace(BATCH_INFER, "} // namespace", """static ge::graphStatus InferDataTypeForBatchMatMulV3(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}
} // namespace""")
        replace(BATCH_INFER, ".InferShapeRange(InferShapeRangeForBatchMatMulV3);",
                ".InferShapeRange(InferShapeRangeForBatchMatMulV3)\n    .InferDataType(InferDataTypeForBatchMatMulV3);")
        replace(BATCH_HOST, '#include "batch_mat_mul_v3_base_tiling.h"',
                '#include "batch_mat_mul_v3_base_tiling.h"\n#include "paretopilot_params.h"\n#include <cstdio>')
        replace(BATCH_HOST, "    auto ret = MatmulV3BaseTiling::DoLibApiTiling();", """    if (PP_EXPLICIT_TILES && PP_BATCH_STRATEGY != 1) {
        OP_LOGE(args_.opName, "Explicit tiles require common batch_strategy=1");
        return ge::GRAPH_FAILED;
    }
    auto ret = MatmulV3BaseTiling::DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) { return ret; }""")
        replace(BATCH_HOST, "void BatchMatmulV3BaseTiling::DoCommonTiling()\n{\n    TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum);",
                "void BatchMatmulV3BaseTiling::DoCommonTiling()\n{\n    if (!PP_EXPLICIT_TILES) { TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum); }")
        replace(BATCH_HOST, "    DoCommonTiling();", """    DoCommonTiling();
    if (PP_EXPLICIT_TILES) {
        const auto &pp = bmmTilingData_.matmulTiling.matmulTiling;
        const uint64_t ppL1 = static_cast<uint64_t>(pp.depthA1) * pp.baseM * pp.baseK * aDtypeSize_ +
                              static_cast<uint64_t>(pp.depthB1) * pp.baseN * pp.baseK * bDtypeSize_;
        if (pp.baseM != PP_BASE_M || pp.baseN != PP_BASE_N || pp.baseK == 0 ||
            ppL1 > compileInfo_.l1Size + 256UL) {
            OP_LOGE(args_.opName, "Explicit tile dependency/capacity validation failed");
            return ge::GRAPH_FAILED;
        }
    }""")
        # Keep the default algorithm intact. The alternative keeps upstream
        # common tiling and excludes L1-full-load/multibatch specialization.
        replace(BATCH_HOST, "    DoL1FullLoadTiling();", "    if (PP_BATCH_STRATEGY == 0) { DoL1FullLoadTiling(); }")
        replace(BATCH_HOST, "    if (compileInfo_.supportL0c2out && tilingSelect_ != TilingCalcSelect::COMMON &&",
                "    if (PP_BATCH_STRATEGY == 0 && compileInfo_.supportL0c2out && tilingSelect_ != TilingCalcSelect::COMMON &&")
        replace(BATCH_HOST, "    workspaces[0] = workspaceSize_;", """    workspaces[0] = workspaceSize_;
    const auto &ppTiling = bmmTilingData_.matmulTiling.matmulTiling;
    std::fprintf(stderr, "PARETOPILOT_TILE_REQUEST={\\\"base_m\\\":%d,\\\"base_n\\\":%d,\\\"base_k_auto\\\":true}\\n", PP_BASE_M, PP_BASE_N);
    std::fprintf(stderr, "PARETOPILOT_PARAMS=%s\\nPARETOPILOT_OFFICIAL=%s\\n", PP_PARAMS_HASH, PP_SOURCE_COMMIT);
    std::fprintf(stderr, "PARETOPILOT_BATCH_TILING={\\\"core_limit\\\":%d,\\\"tiling_policy\\\":%d,\\\"batch_strategy\\\":%d,\\\"used_core_num\\\":%u,\\\"base_m\\\":%u,\\\"base_n\\\":%u,\\\"base_k\\\":%u,\\\"tiling_key\\\":%llu,\\\"block_dim\\\":%llu,\\\"batch_count\\\":%llu,\\\"multi_batch\\\":%d,\\\"load_mode\\\":%d}\\n",
        PP_CORE_LIMIT, PP_TILING_POLICY, PP_BATCH_STRATEGY, ppTiling.usedCoreNum,
        ppTiling.baseM, ppTiling.baseN, ppTiling.baseK,
        static_cast<unsigned long long>(tilingKey_), static_cast<unsigned long long>(compileInfo_.aicNum),
        static_cast<unsigned long long>(batchInfo_.batchC),
        static_cast<int>(tilingEnable_.tilingEnableMultiBatch), static_cast<int>(tilingEnable_.tilingEnableLoadMode));""")
        (project/BATCH_HOST).with_name("paretopilot_params.h").write_text(header, encoding="utf-8")
    adapter = Path(__file__).parent/"official_adapter"
    for name in ("parser.cpp", "parser.cmake"):
        shutil.copyfile(adapter/("batch_parser.cpp" if batch and name == "parser.cpp" else name), project/name)
    replace("CMakeLists.txt", "install(DIRECTORY ${OPS_ADV_ACT}/",
            "# ParetoPilot ATC parser integration.\ninclude(parser.cmake)\n\ninstall(DIRECTORY ${OPS_ADV_ACT}/")
    return {"repository": manifest["repository"], "commit": COMMIT, "operator": "BatchMatMulV3" if batch else "MatMulV3",
            "archives": manifest["archives"], "params": params, "params_sha256": digest(params),
            "patches": changes, "generated_source_sha256": tree_hash(project),
            "device_dependency_sha256": tree_hash(project/"matmul/mat_mul_v3/op_kernel"),
            "device_source_sha256": tree_hash(project/("matmul/batch_mat_mul_v3/op_kernel" if batch else "matmul/mat_mul_v3/op_kernel"))}


def generated_implementations(build, batch=False):
    dynamic = Path(build)/"tbe/dynamic"
    if not (dynamic/"mat_mul_v3.py").is_file():
        raise TuningError(f"Official MatMul Python adapter was not generated: {dynamic/'mat_mul_v3.py'}; see generate.log")
    if batch:
        adapter = dynamic/"batch_mat_mul_v3.py"
        text = adapter.read_text(encoding="utf-8") if adapter.is_file() else ""
        if "batch_mat_mul_v3.cpp" not in text or "batch_mat_mul_v3_apt.cpp" in text:
            raise TuningError("910B BatchMatMulV3 adapter missing or selects an _apt.cpp entry; inspect generate.log")
    return sorted(dynamic.glob("*.py"))


def verify_installed_implementations(generated, vendor):
    """Reject incomplete/stale OPP packages before ATC tries to import them."""
    tbe = Path(vendor)/"op_impl/ai_core/tbe"
    dynamic = tbe/(Path(vendor).name+"_impl")/"dynamic"
    hashes = {}
    for source in generated:
        installed = dynamic/source.name
        if not installed.is_file():
            raise TuningError(f"Official MatMul OPP is missing Python adapter: {installed}; see build.log and install.log")
        expected = file_hash(source)
        if file_hash(installed) != expected:
            raise TuningError(f"Official MatMul installed Python adapter differs from generated source: {installed}")
        hashes[installed.relative_to(tbe).as_posix()] = expected
    return hashes


def build_official(config, params, directory, env):
    if batch_enabled(config) and config["implementation"].get("build_cache", True):
        from .official_cache import cached_build
        return cached_build(config, params, directory, env)
    return _build_official(config, params, directory, env)


def _build_official(config, params, directory, env, *, runtime=False):
    batch = batch_enabled(config)
    legacy, legacy_relative = legacy_library(env)
    project = directory/"project"
    provenance = materialize(project, params, batch=batch, runtime=runtime)
    write_json(directory/"source_manifest.json", provenance)
    write_json(directory/"params.json", params)
    timeout = config["measurement"]["timeout_s"]
    build = project/"build"
    jobs = str(config["implementation"].get("build_jobs", 4))
    commands = [
        ("configure", ["cmake", "-S", str(project), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                       "-DENABLE_CUSTOM=ON", "-DENABLE_PACKAGE=ON", "-DENABLE_BINARY=OFF",
                       "-DASCEND_COMPUTE_UNIT=ascend910b", "-DASCEND_OP_NAME="+("batch_mat_mul_v3" if batch else "mat_mul_v3"),
                       "-DASCEND_COMPILE_OPS=mat_mul_v3;batch_mat_mul_v3;matmul.common", "-DVENDOR_NAME=paretopilot"]),
        ("generate", ["cmake", "--build", str(build), "--target", "ascendc_impl_gen", "--parallel", jobs]),
        ("build", ["cmake", "--build", str(build), "--target", "package", "paretopilot_onnx", "--parallel", jobs]),
    ]
    for name, command in commands:
        record = run_command(command, project, directory/f"{name}.log", timeout, env)
        write_json(directory/f"{name}.command.json", record)
        if name == "generate":
            implementations = generated_implementations(build, batch=batch)
    packages = sorted((project/"build_out").glob("*.run"))
    if len(packages) != 1:
        raise TuningError(f"Expected one official custom OPP package, found {packages}; see build.log")
    install = directory/"opp"
    install.mkdir()
    run_command(["bash", str(packages[0]), "--install-path="+str(install)], project, directory/"install.log", timeout, env)
    # Upstream LegacyCommonMgr walks from vendors/... back to this isolated OPP.
    # Copy only its runtime dependency; never install into or alter the system OPP.
    legacy_copy = install/legacy_relative
    legacy_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, legacy_copy)
    vendors = [p for p in (install/"vendors").iterdir() if p.is_dir()]
    if len(vendors) != 1:
        raise TuningError("Expected one isolated official MatMul OPP vendor")
    implementation_hashes = verify_installed_implementations(implementations, vendors[0])
    parser = vendors[0]/"framework/onnx"
    parser.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build/"libparetopilot_onnx.so", parser/"libparetopilot_onnx.so")
    result = {**provenance, "implementation": BATCH_KIND if batch else KIND, "op_type": provenance["operator"], "opp_path": str(vendors[0]),
              "source_dir": str(project), "package": str(packages[0]), "installed_sha256": tree_hash(vendors[0]),
              "python_implementations": implementation_hashes,
              "parameter_binding": "atc_process_environment" if runtime else "compiled_header",
              "isolated_legacy_library": str(legacy_copy),
              "legacy_library": str(legacy), "legacy_library_sha256": file_hash(legacy_copy)}
    if config["implementation"].get("execution") == "kernel":
        if not batch or not runtime:
            raise TuningError("Kernel execution requires the runtime-parameter BatchMatMulV3 build cache")
        from .direct_build import build_kernels
        result = build_kernels(config, result, directory, env)
    write_json(directory/"build_manifest.json", result)
    return result


def read_effective_tiling(log, params, batch=False):
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    if f"PARETOPILOT_OFFICIAL={COMMIT}" not in text:
        raise TuningError("ATC did not execute the pinned official MatMul Host tiler")
    try:
        prefix = "PARETOPILOT_BATCH_TILING" if batch else "PARETOPILOT_TILING"
        records = [json.loads(m) for m in re.findall(prefix+r"=(\{[^\r\n]*\})", text)]
    except ValueError as exc:
        raise TuningError("Invalid official MatMul effective tiling JSON") from exc
    if not records:
        raise TuningError("Official MatMul effective tiling record missing in ATC log")
    for record in records:
        if not isinstance(record, dict) or not {"core_limit", "tiling_policy", "used_core_num", "base_m", "base_n", "base_k", "tiling_key", *params} <= record.keys():
            raise TuningError("Incomplete official MatMul effective tiling record")
        if any(record[k] != v for k, v in params.items() if k not in TILE_DEFAULT_PARAMS) or any(record[k] <= 0 for k in ("used_core_num", "base_m", "base_n", "base_k")):
            raise TuningError("Official MatMul effective tiling does not match requested parameters")
        if "base_m" in params:
            requests = [json.loads(m) for m in re.findall(r"PARETOPILOT_TILE_REQUEST=(\{[^\r\n]*\})", text)]
            if (not requests or any(r.get("base_m") != params["base_m"] or r.get("base_n") != params["base_n"]
                                   or r.get("base_k_auto") is not True for r in requests)
                    or (params["base_m"] and (record["base_m"] != params["base_m"] or record["base_n"] != params["base_n"]))):
                raise TuningError("Explicit baseM/baseN request was missing, stale or overwritten")
        if params["core_limit"] and record["used_core_num"] > params["core_limit"]:
            raise TuningError("Official MatMul exceeded the requested core budget")
        if batch:
            if not {"batch_strategy", "block_dim", "batch_count", "multi_batch", "load_mode"} <= record.keys():
                raise TuningError("Incomplete BatchMatMulV3 effective tiling record")
            if record["block_dim"] <= 0 or record["batch_count"] <= 0 or (params["core_limit"] and record["block_dim"] > params["core_limit"]):
                raise TuningError("Invalid BatchMatMulV3 block/batch count")
            if params["batch_strategy"] == 1 and (record["multi_batch"] != 0 or record["load_mode"] != 0):
                raise TuningError("BatchMatMulV3 common strategy was not applied")
    return records
