"""Synthetic protocol fixtures only. Refuses real mode."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    q = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    c, stage, root = q["config"], q["stage"], Path(q["workdir"])
    if c["mode"] != "simulation":
        raise RuntimeError("Demo adapter refuses real mode")
    result = {"status": "ok", "mode": "simulation", "notice": "Synthetic data only"}
    om = root / "simulation.om"
    if stage == "atc":
        om.write_text(json.dumps({"params": q["params"], "role": q["role"]}), encoding="utf-8")
        (root / "selection.txt").write_text("SIMULATED ATC selection " + q["params_hash"], encoding="utf-8")
        result.update(om_path=str(om), selected_params_hash=q["params_hash"], selected_source_sha256=sha(c["kernel_source"]),
                      implementation_id="simulation-template", selection_evidence="selection.txt")
    elif stage == "verify":
        result.update(passed=True, reference="pytorch", om_sha256=sha(om), simulated=True)
    elif stage == "profile":
        p = q["params"]
        tile = p.get("tile_m", p.get("tile_elems", 16)/16)
        duration = 100 if q["role"] == "baseline" else 110-min(60, tile/2)-p["block_dim"]/4-p["buffers"]*5
        path = root / "op_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([c["profiling"]["name_column"], c["profiling"]["duration_column"]])
            writer.writerows([c["profiling"]["target_name"], duration*(1+(i % 3-1)*0.001)] for i in range(c["search"]["repeats"]))
        result.update(csv_path=str(path), warmup=c["search"]["warmup"], om_sha256=sha(om))
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
