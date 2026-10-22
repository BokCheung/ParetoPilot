"""Reviewable experiment decision and its limitations."""
import json


def write_report(root, result):
    text = ["# ParetoPilot Kernel Tuning", "", f"Mode: **{result['mode']}**", "",
            "Metric: baseline/candidate OM target-op Task Duration(us) median. PyTorch is correctness reference only.", "",
            f"Candidate evaluations: {result['evaluations']}; accepted: **{result['accepted']}**.", "",
            f"Baseline initial task time: {result['baseline']['duration_us']:.6f} us.",
            f"Confirmed relative gain: {result['confirmation_relative_gain']}; observed relative range: {result['observed_relative_range']}.", "",
            "Selected parameters:", "", "```json", json.dumps(result["selected_params"], indent=2), "```", "",
            "summary.json contains confirmation, Roofline hypotheses and adaptive domains. history.json and each trial directory preserve requests, responses, logs, OM and CSV.", "",
            "Shape-derived traffic is a lower bound. Local-memory estimates do not replace chip-specific L0/L1/UB checks. Confirmation uses a range heuristic, not a statistical confidence interval."]
    if result["mode"] == "simulation":
        text += ["", "**All timings, correctness results, OM and selection evidence are simulated. No hardware speedup has been measured.**"]
    (root / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
