import csv
from pathlib import Path
import statistics
from .util import TuningError, number


def read_profile(paths, selector, repeats, duration_column="Task Duration(us)", traffic_column=None):
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or any(k not in reader.fieldnames for k in [*selector, duration_column]):
                raise TuningError(f"Profiling CSV missing required columns: {path}")
            for row in reader:
                if all(row[k] == v for k, v in selector.items()):
                    records.append(row)
    # Single-node fixed graph only. Reject duplicate exports, ambiguous ops and missing iterations.
    if len(records) != repeats:
        raise TuningError(f"Expected exactly {repeats} target task samples; got {len(records)}. Inspect selectors and exported CSVs.")
    for column in ("Op Name", "Device_id", "Device Id", "Stream Id", "Model Id"):
        values = {r[column] for r in records if column in r}
        if len(values) > 1:
            raise TuningError(f"Ambiguous profiling samples across {column}: {values}")
    try:
        values = [number(float(r[duration_column]), "task duration", 1e-15) for r in records]
        traffic = statistics.median(number(float(r[traffic_column]), "measured bytes", 1e-15) for r in records) if traffic_column else None
    except (TypeError, ValueError, KeyError) as exc:
        raise TuningError(f"Invalid profiling value: {exc}") from exc
    return {"duration_us": statistics.median(values), "samples_us": values, "measured_bytes": traffic,
            "metric": "OM target-op Task Duration(us) median", "selector": selector, "csv_files": list(map(str, paths))}
