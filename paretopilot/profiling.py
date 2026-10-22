"""Import exact-name target-op samples in microseconds."""
import csv
from pathlib import Path
import statistics
from .artifacts import ConfigError, number


def read_profile(path, config):
    p = config["profiling"]
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [r for r in rows if r[p["name_column"]] == p["target_name"]]
        if len(rows) < config["search"]["repeats"]:
            raise ConfigError("Insufficient exact-target profiling samples")
        samples = [number(float(r[p["duration_column"]]), "Task Duration(us)", strict=True) for r in rows]
    except (KeyError, ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid profiling CSV: {exc}") from exc
    return {"samples_us": samples, "duration_us": statistics.median(samples),
            "metric": "OM target-op Task Duration(us) median"}
