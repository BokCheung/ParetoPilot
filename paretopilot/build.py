"""Generate the CANN 8.5 project locally, insert bundled kernels, build and isolate OPP."""
from pathlib import Path
import shutil
from .util import TuningError, digest, read_json, run_command, tree_hash, write_json


def op_names(config):
    return ("PpMatmul", "pp_matmul") if config["operator"]["kind"] == "matmul" else ("PpAdd", "pp_add")


def build_candidate(config, params, directory, env):
    directory = Path(directory)
    op, stem = op_names(config)
    timeout = config["measurement"]["timeout_s"]
    parameter_hash = digest(params)
    proto = [{"op": op, "language": "cpp", "input_desc": [{"name": name, "param_type": "required", "format": ["ND"], "type": ["float16"]} for name in ("a", "b")],
              "output_desc": [{"name": "c", "param_type": "required", "format": ["ND"], "type": ["float16"]}]}]
    write_json(directory/"operator.json", proto)
    write_json(directory/"params.json", params)
    project = directory/"project"
    run_command([config["ascend"].get("msopgen", "msopgen"), "gen", "-i", str(directory/"operator.json"), "-f", "onnx", "-c", "ai_core-"+config["ascend"]["soc_version"],
                 "-lan", "cpp", "-out", str(project)], directory, directory/"generate.log", timeout, env)
    templates = Path(__file__).parent/"kernels"
    header = "#pragma once\n" + ''.join(f"#define PP_{name.upper()} {value}\n" for name, value in sorted(params.items())) + f'#define PP_PARAMS_HASH "{parameter_hash}"\n'
    for folder in ("op_host", "op_kernel"):
        if not (project/folder).is_dir():
            raise TuningError(f"Unexpected msopgen project layout: missing {folder}")
        (project/folder/"paretopilot_params.h").write_text(header, encoding="utf-8")
    shutil.copyfile(templates/f"{stem}_host.cpp", project/"op_host"/f"{stem}.cpp")
    shutil.copyfile(templates/f"{stem}_tiling.h", project/"op_host"/f"{stem}_tiling.h")
    shutil.copyfile(templates/f"{stem}_kernel.cpp", project/"op_kernel"/f"{stem}.cpp")
    # Replace only the generated ONNX parser implementation; its generated CMake owns linking.
    parsers = list((project/"framework").rglob("*.cpp"))
    if len(parsers) != 1:
        raise TuningError(f"Expected one generated ONNX parser; found {parsers}")
    parsers[0].write_text('#include "register/register.h"\nnamespace domi {\n'
        f'Status Parse{op}(const ge::Operator &src, ge::Operator &dst) {{ return SUCCESS; }}\n'
        f'REGISTER_CUSTOM_OP("{op}").FrameworkType(ONNX).OriginOpType("ai.onnx::13::{op}")'
        f'.ParseParamsByOperatorFn(Parse{op}).ImplyType(ImplyType::TVM);\n}}\n', encoding="utf-8")
    presets = project/"CMakePresets.json"
    data = read_json(presets)
    def update(obj):
        if isinstance(obj, dict):
            if "vendor_name" in obj:
                if isinstance(obj["vendor_name"], dict):
                    obj["vendor_name"]["value"] = "paretopilot"
                else:
                    obj["vendor_name"] = "paretopilot"
            for value in obj.values():
                update(value)
        elif isinstance(obj, list):
            for value in obj:
                update(value)
    update(data)
    write_json(presets, data)
    source_hash = digest({folder: tree_hash(project/folder) for folder in ("op_host", "op_kernel", "framework")})
    write_json(directory/"source_manifest.json", {"params_sha256": parameter_hash, "generated_source_sha256": source_hash, "presets": data})
    run_command(["bash", "build.sh"], project, directory/"build.log", timeout, env)
    packages = list((project/"build_out").glob("*.run"))
    if len(packages) != 1:
        raise TuningError("Expected a single custom OPP .run package in project/build_out")
    install = directory/"opp"
    install.mkdir()
    run_command(["bash", str(packages[0]), "--install-path="+str(install)], project, directory/"install.log", timeout, env)
    vendors = [p for p in (install/"vendors").iterdir() if p.is_dir()]
    if len(vendors) != 1:
        raise TuningError("Expected one isolated OPP vendor directory")
    result = {"op_type": op, "params_sha256": parameter_hash, "generated_source_sha256": source_hash, "template_sha256": tree_hash(templates), "installed_sha256": tree_hash(vendors[0]),
              "opp_path": str(vendors[0]), "source_dir": str(project), "package": str(packages[0])}
    write_json(directory/"build_manifest.json", result)
    return result
