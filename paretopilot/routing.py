"""Route measured GE operators to existing source adapters, without substitution."""
from .official import enabled as official_enabled, COMMIT
from .workload import baseline_only
from .util import TuningError


def same_operator(config):
    return (config["backend"] == "ascend" and config["operator"]["kind"] == "matmul"
            and config.get("implementation", {}).get("require_same_operator", True))


def route(config, baseline):
    identity = baseline.get("operator_identity", {})
    observed = identity.get("Op Type")
    supported = (observed == "MatMulV3" and official_enabled(config)
                 and "batch" not in config["operator"] and not baseline_only(config))
    reason = ("Matched measured MatMulV3 to the pinned MatMulV3 source adapter" if supported else
              "No measured operator identity; refusing to select a source adapter" if not observed else
              f"No same-operator source adapter for measured {observed} and this shape; baseline retained")
    return {"policy": "same_operator", "operator_identity": identity,
            "input_shapes": baseline.get("input_shapes"), "output_shape": baseline.get("output_shape"),
            "tuning_supported": supported, "adapter": "cann_matmul_v3" if supported else None,
            "source_commit": COMMIT if supported else None, "reason": reason,
            "next_step": None if supported else "Use inspect_cann_operator.py on this baseline to establish the installed compile entry, dependencies and modifiable tiling inputs",
            "note": "An operator type match does not establish identical internal kernel or installed source revision; source and effective tiling evidence remain necessary."}


def check_identity(config, baseline, trial):
    if same_operator(config) and trial.get("status") == "ok":
        expected = baseline.get("operator_identity", {}).get("Op Type")
        observed = trial.get("operator_identity", {}).get("Op Type")
        if not expected or observed != expected:
            raise TuningError(f"Same-operator policy: expected {expected}, observed {observed}; rejecting this trial")
