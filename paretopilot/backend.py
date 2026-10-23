import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .build import build_candidate, op_names
from .official import enabled as official_enabled, read_effective_tiling
from .profiling import read_profile
from .roofline import analyze
from .util import TuningError, digest, file_hash, read_json, run_command, write_json


def environment(opp_path=None):
    env = os.environ.copy()
    env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    if opp_path:
        env["ASCEND_CUSTOM_OPP_PATH"] = str(opp_path)
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


class AscendBackend:
    def __init__(self, config, root):
        self.c, self.root = config, Path(root)
        self.case = self.root/"case"
        self.case.mkdir()
        write_json(self.case/"export_request.json", {"config": config, "directory": str(self.case)})
        self.command([sys.executable, "-m", "paretopilot.worker", "export", str(self.case/"export_request.json")], self.case, "export", environment())

    def command(self, argv, directory, name, env):
        record = run_command(argv, directory, directory/f"{name}.log", self.c["measurement"]["timeout_s"], env)
        write_json(directory/f"{name}.command.json", record)

    def evaluate(self, params, directory, role):
        directory = Path(directory)
        env, build = environment(), None
        if role in ("candidate", "source_baseline"):
            build = build_candidate(self.c, params, directory, env)
            env = environment(build["opp_path"])
            self.candidate_graph(directory/"model.onnx")
        else:
            shutil.copyfile(self.case/"baseline.onnx", directory/"model.onnx")
        metadata = read_json(self.case/"case.json")
        shapes = ';'.join(t["name"]+':'+','.join(map(str, t["shape"])) for t in metadata["inputs"])
        atc = [self.c["ascend"].get("atc", "atc"), "--framework=5", "--model="+str(directory/"model.onnx"),
               "--output="+str(directory/"model"), "--soc_version="+self.c["ascend"]["soc_version"], "--input_shape="+shapes,
               "--input_format=ND", "--log=info"] + self.c["ascend"].get("atc_extra", [])
        self.command(atc, directory, "atc", env)
        om = directory/"model.om"
        om_hash = file_hash(om)
        if role in ("candidate", "source_baseline"):
            marker = "PARETOPILOT_PARAMS="+digest(params)
            if marker not in (directory/"atc.log").read_text(encoding="utf-8", errors="replace"):
                raise TuningError("ATC log does not show candidate Host tiling parameter signature")
            if official_enabled(self.c):
                build["effective_tiling"] = read_effective_tiling(directory/"atc.log", params)
            write_json(directory/"selection.json", {**build, "om_sha256": om_hash, "atc_log_sha256": file_hash(directory/"atc.log"), "marker": marker})
        write_json(directory/"runtime_request.json", {"config": self.c, "om": str(om), "case": str(self.case), "directory": str(directory)})
        self.command([sys.executable, "-m", "paretopilot.worker", "run", str(directory/"runtime_request.json")], directory, "runtime", env)
        runtime = read_json(directory/"runtime.json")
        if runtime["om_sha256"] != om_hash or file_hash(om) != om_hash or runtime["correctness"]["passed"] is not True:
            raise TuningError("OM changed or correctness failed")
        self.command([self.c["ascend"].get("msprof", "msprof"), "--export=on", "--output="+str(directory/"profiler")], directory, "profiler_export", env)
        paths = sorted((directory/"profiler").rglob("op_summary*.csv"))
        if not paths:
            raise TuningError("msprof produced no op_summary CSV; inspect profiler_export.log")
        m = self.c["measurement"]
        metrics = read_profile(paths, m["selectors"][role], m["repeats"], m.get("duration_column", "Task Duration(us)"), m.get("traffic_column"))
        return {"status": "ok", "mode": "ascend", **metrics, "correctness": runtime["correctness"], "om_sha256": om_hash,
                "om_path": str(om), "source_dir": build["source_dir"] if build else None, "selection": build}

    def candidate_graph(self, path):
        import onnx
        from onnx import helper, TensorProto
        case = read_json(self.case/"case.json")
        inputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT16, t["shape"]) for t in case["inputs"]]
        output = helper.make_tensor_value_info("c", TensorProto.FLOAT16, case["output_shape"])
        node = helper.make_node(op_names(self.c)[0], ["a", "b"], ["c"], name="target")
        graph = helper.make_graph([node], "candidate", inputs, [output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        # Custom op belongs to ATC's registered parser, not the standard ONNX schema.
        onnx.save(model, str(path))


class SimulationBackend:
    """Synthetic end-to-end controller exercise; never used as an Ascend fallback."""
    def __init__(self, config, root):
        self.c = config

    def evaluate(self, params, directory, role):
        m = self.c["measurement"]
        selector = m["selectors"][role]
        if role == "baseline":
            duration = 100.0
        else:
            tile = params.get("tile_m", params.get("tile_elems", 16)/16)
            duration = max(10, 110-min(60, tile/2)-params["block_dim"]/4-params["buffers"]*5)
        path = directory/"op_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, [*selector, "Task Duration(us)"])
            writer.writeheader()
            for i in range(m["repeats"]):
                writer.writerow({**selector, "Task Duration(us)": duration*(1+(i % 3-1)*0.001)})
        metrics = read_profile([path], selector, m["repeats"])
        return {"status": "ok", "mode": "simulation", **metrics, "correctness": {"passed": True, "simulated": True}}


def evaluate(backend, config, params, directory, role):
    directory.mkdir(parents=True)
    write_json(directory/"request.json", {"role": role, "params": params})
    try:
        trial = backend.evaluate(params, directory, role)
        trial["roofline"] = analyze(config, trial["duration_us"], trial["measured_bytes"])
    except (TuningError, OSError, subprocess.SubprocessError, KeyError, TypeError, ImportError) as exc:
        trial = {"status": "failed", "error": str(exc), "mode": config["backend"]}
    trial.update(role=role, params=params, directory=str(directory))
    write_json(directory/"trial.json", trial)
    return trial
