"""Measured V2 compiler-strategy experiment; not manual tile-size tuning."""
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics

from .agent import DummyAgent, validate_proposal
from .backend import AscendBackend, evaluate
from .doctor import inspect_environment
from .roofline import analyze
from .report import roofline_svg
from .tbe_probe_hook import CALLERS
from .util import TuningError, file_hash, read_json, write_json


def installed_source():
    opp = os.environ.get("ASCEND_OPP_PATH")
    if not opp:
        raise TuningError("ASCEND_OPP_PATH is missing; source the CANN environment")
    source = Path(opp)/"built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/batch_matmul_v2.py"
    if not source.is_file():
        raise TuningError(f"Audited installed V2 wrapper not found: {source}")
    return source.resolve()


def verify_build_records(path, source, expected_hash, enabled):
    if not path.is_file():
        raise TuningError("No V2 build evidence: ATC may not load sitecustomize/SourceFileLoader, or may reuse a binary. Probe result rejected.")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    calls = [r for r in records if r.get("event") == "build_call"]
    returned = [r for r in records if r.get("event") == "build_return"]
    if not calls or len(returned) != len(calls):
        raise TuningError("No complete V2 tbe.build calls; startup-hook evidence alone is insufficient")
    expected_calls = [{**r, "event": "build_return"} for r in calls]
    if sorted(map(json.dumps, returned)) != sorted(map(json.dumps, expected_calls)):
        raise TuningError("V2 build call/return evidence does not match")
    for record in calls:
        before, effective = record.get("before_build_args", {}), record.get("effective_build_args", {})
        if (record.get("source_path") != str(source) or record.get("source_sha256") != expected_hash
                or record.get("operator") != "BatchMatMulV2" or record.get("caller") not in CALLERS
                or not record.get("kernel_name") or before.get("enable_db_fold") is not True
                or effective.get("enable_db_fold") is not enabled
                or {k: v for k, v in before.items() if k != "enable_db_fold"}
                   != {k: v for k, v in effective.items() if k != "enable_db_fold"}):
            raise TuningError("V2 source identity/effective configuration mismatch; probe rejected")
    if file_hash(source) != expected_hash:
        raise TuningError("Installed V2 source changed during compilation")
    return {"source_path": str(source), "source_sha256": expected_hash,
            "strategy": {"enable_db_fold": enabled}, "build_calls": calls,
            "evidence_kind": "instrumented Python build arguments; not a hardware tiling trace"}


def bind_selected_kernels(evidence, directory):
    """Require a recorded kernel name in the OM; collect code bytes if available."""
    contents = (directory/"model.om").read_bytes()
    names = sorted({r["kernel_name"] for r in evidence["build_calls"] if r["kernel_name"].encode() in contents})
    if not names:
        raise TuningError("Cannot bind any instrumented build kernel name to the OM; probe rejected")
    artifacts = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*")
                 if p.is_file() and p.suffix in (".o", ".bin") and p.stem in names}
    return {**evidence, "om_kernel_names": names, "kernel_binary_sha256": artifacts}


class ProbeBackend(AscendBackend):
    def __init__(self, config, root):
        self.probe_params = None
        self.source = None
        super().__init__(config, root)
        self.hook = self.root/"python_hook"
        self.hook.mkdir()
        (self.hook/"sitecustomize.py").write_text("from paretopilot.tbe_probe_hook import install\ninstall()\n")

    def command(self, argv, directory, name, env):
        if name == "atc":
            env = env.copy()
            for key, folder in (("ASCEND_CACHE_PATH", "compiler_cache"), ("ASCEND_WORK_PATH", "compiler_work")):
                path = directory/folder
                path.mkdir()
                env[key] = str(path)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if self.probe_params is not None:
                env.update(PARETOPILOT_TBE_SOURCE=str(self.source),
                           PARETOPILOT_TBE_SOURCE_SHA256=self.source_hash,
                           PARETOPILOT_TBE_RECORD=str(directory/"tbe-build.jsonl"),
                           PARETOPILOT_TBE_POLICY=json.dumps({"enable_db_fold": bool(self.probe_params["enable_db_fold"])}))
                env["PYTHONPATH"] = str(self.hook)+os.pathsep+env.get("PYTHONPATH", "")
        return super().command(argv, directory, name, env)

    def evaluate(self, params, directory, role):
        self.probe_params = None if role == "baseline" else params
        if self.probe_params is not None and (set(params) != {"enable_db_fold"}
                or type(params["enable_db_fold"]) is not int or params["enable_db_fold"] not in (0, 1)):
            raise TuningError("Only enable_db_fold=0/1 is supported")
        # All roles retain the exact stock ONNX; no custom op or V3 substitution.
        result = super().evaluate({}, directory, "baseline")
        if result["operator_identity"].get("Op Type") != "BatchMatMulV2":
            raise TuningError("V2 strategy probe requires measured BatchMatMulV2")
        if self.probe_params is not None:
            evidence = verify_build_records(directory/"tbe-build.jsonl", self.source,
                self.source_hash, bool(params["enable_db_fold"]))
            result["selection"] = bind_selected_kernels(evidence, directory)
        return result


def comparison(config, trials):
    roles = ("baseline", "source_baseline", "candidate")
    if any(t.get("status") != "ok" for t in trials):
        return {"accepted": False, "reason": "incomplete or failed independent confirmation"}
    timings = {role: [t["duration_us"] for t in trials if t["role"] == role] for role in roles}
    rounds = config["measurement"]["confirmation_rounds"]
    if any(t.get("status") != "ok" for t in trials) or any(len(timings[r]) != rounds for r in roles):
        return {"accepted": False, "reason": "incomplete or failed independent confirmation"}
    medians = {r: statistics.median(v) for r, v in timings.items()}
    spread = max((max(v)-min(v))/statistics.median(v) for v in timings.values())
    gain = min(1-medians["candidate"]/medians[r] for r in ("baseline", "source_baseline"))
    accepted = gain > max(config["search"]["min_gain"], spread)
    return {"accepted": accepted, "reason": "compiler strategy beats both references beyond threshold/noise" if accepted else "compiler-strategy gain not established",
            "baseline_us": medians["baseline"], "source_baseline_us": medians["source_baseline"],
            "candidate_us": medians["candidate"], "speedup": medians["baseline"]/medians["candidate"],
            "source_speedup": medians["source_baseline"]/medians["candidate"], "relative_range": spread}


def run_probe(config, root):
    root = Path(root)
    write_json(root/"manifest.json", {"config": config, "experiment": "v2_compiler_strategy_probe",
               "manual_tiling_control": False, "controller_sha256": file_hash(Path(__file__))})
    environment = inspect_environment(config)
    write_json(root/"environment.json", environment)
    if not environment["ready"]:
        raise TuningError(f"Preflight failed; see {root/'environment.json'}")
    backend = ProbeBackend(config, root)
    def measure(role, params, directory):
        trial = evaluate(backend, config, params, directory, role)
        print(f"{directory.name}: {trial['status']}"+(f" {trial['duration_us']:.4f} us" if trial["status"] == "ok" else ": "+trial["error"]), flush=True)
        if trial["status"] != "ok":
            raise TuningError(f"V2 probe failed; see {directory/'trial.json'}")
        return trial
    baseline = measure("baseline", {}, root/"baseline")
    backend.source = installed_source()
    backend.source_hash = file_hash(backend.source)
    write_json(root/"installed-source.json", {"path": str(backend.source), "sha256": backend.source_hash})
    context = {"original_space": {"enable_db_fold": [1, 0]}, "baseline": baseline,
               "roofline": analyze(config, baseline["duration_us"]), "operator": config["operator"],
               "parameter_kind": "compiler_strategy", "manual_tiling_control": False}
    proposal = validate_proposal(DummyAgent().propose(deepcopy(context)), context["original_space"])
    write_json(root/"agent/input.json", context)
    write_json(root/"agent/output.json", proposal)
    source_baseline = measure("source_baseline", {"enable_db_fold": 1}, root/"source_baseline")
    candidate = measure("candidate", {"enable_db_fold": 0}, root/"candidate")
    trials = []
    for i in range(config["measurement"]["confirmation_rounds"]):
        roles = ("baseline", "source_baseline", "candidate")
        for role in (roles if i % 2 == 0 else tuple(reversed(roles))):
            params = {} if role == "baseline" else {"enable_db_fold": int(role == "source_baseline")}
            trials.append(measure(role, params, root/f"confirmation/{i:03d}-{role}"))
            write_json(root/"confirmation.json", trials)
    result = {"status": "passed", "experiment": "v2_compiler_strategy_probe", "manual_tiling_control": False,
              "agent": {"kind": "dummy"}, "baseline": baseline, "source_baseline": source_baseline,
              "candidate": candidate, "confirmation": comparison(config, trials), "trials": trials}
    code = {role: [set(t.get("selection", {}).get("kernel_binary_sha256", {}).values())
                   for t in [source_baseline, candidate, *trials] if t.get("role") == role]
            for role in ("source_baseline", "candidate")}
    verified = (all(code.values()) and all(values and all(values[0] == s for s in values) for values in code.values())
                and code["source_baseline"][0] != code["candidate"][0])
    result["code_generation_difference_verified"] = bool(verified)
    if not verified:
        result["confirmation"].update(accepted=False, reason="kernel code change not established; build-argument evidence alone is insufficient for accepting a gain")
    result["selected"] = "candidate" if result["confirmation"]["accepted"] else "baseline"
    write_json(root/"summary.json", result)
    text = "# BatchMatMulV2 compiler-strategy probe\n\nThis experiment changes only enable_db_fold, not tile sizes or synchronization settings.\n\n"
    text += json.dumps(result["confirmation"], indent=2)+"\n"
    (root/"report.md").write_text(text, encoding="utf-8")
    (root/"roofline.svg").write_text(roofline_svg(config, baseline, [source_baseline, candidate]), encoding="utf-8")
    return result
