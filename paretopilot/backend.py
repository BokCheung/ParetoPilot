"""Isolated external build/export/ATC/verify/profile adapter protocol."""
from pathlib import Path
import subprocess
from .artifacts import ConfigError, fingerprint, load_json, number, sha, write_json
from .profiling import read_profile
from .roofline import analyze


class Evaluator:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)

    def call(self, stage, directory, params, role):
        request = {"protocol_version": 1, "stage": stage, "role": role, "params": params,
                   "params_hash": fingerprint(params), "config": self.config, "workdir": str(directory.resolve())}
        req, res = directory / f"{stage}.request.json", directory / f"{stage}.response.json"
        write_json(req, request)
        with (directory / f"{stage}.log").open("w", encoding="utf-8") as log:
            subprocess.run(self.config["adapter"] + [str(req.resolve()), str(res.resolve())], cwd=directory,
                           shell=False, stdout=log, stderr=subprocess.STDOUT, timeout=self.config["timeout_s"], check=True)
        result = load_json(res)
        if not isinstance(result, dict) or result.get("status") != "ok" or result.get("mode") != self.config["mode"]:
            raise ConfigError(f"{stage}: adapter failure or mode mismatch")
        return result

    def evaluate(self, params, label, role="candidate"):
        directory = self.root / label
        directory.mkdir(parents=True, exist_ok=False)
        try:
            for stage in (("export", "atc") if role == "baseline" else ("build", "export", "atc")):
                selection = self.call(stage, directory, params, role)
            om = (directory / selection["om_path"]).resolve()
            om_hash = sha(om)
            if role == "candidate":
                if selection.get("selected_params_hash") != fingerprint(params) or selection.get("selected_source_sha256") != sha(self.config["kernel_source"]):
                    raise ConfigError("ATC selection does not match candidate source/parameters")
                if not selection.get("implementation_id") or not selection.get("selection_evidence"):
                    raise ConfigError("Missing ATC implementation selection evidence")
                selection["selection_evidence_sha256"] = sha(directory / selection["selection_evidence"])
            correctness = self.call("verify", directory, params, role)
            if correctness.get("passed") is not True or correctness.get("reference") != "pytorch" or correctness.get("om_sha256") != om_hash:
                raise ConfigError("Correctness must pass against PyTorch for this exact OM")
            profile = self.call("profile", directory, params, role)
            if profile.get("om_sha256") != om_hash or profile.get("warmup") != self.config["search"]["warmup"]:
                raise ConfigError("Profiling OM hash/warmup mismatch")
            if sha(om) != om_hash:
                raise ConfigError("OM changed during verification/profiling")
            metrics = read_profile(directory / profile["csv_path"], self.config)
            traffic = profile.get("measured_bytes")
            if traffic is not None:
                number(traffic, "measured_bytes", strict=True)
            trial = {"status": "ok", "role": role, "params": params, **metrics, "om_path": str(om), "om_sha256": om_hash,
                     "correctness": correctness, "selection": selection, "roofline": analyze(self.config, metrics["duration_us"], traffic)}
        except (ConfigError, OSError, subprocess.SubprocessError, KeyError, TypeError) as exc:
            trial = {"status": "failed", "role": role, "params": params, "error": str(exc)}
        write_json(directory / "trial.json", trial)
        return trial
