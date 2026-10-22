import csv
from pathlib import Path
import statistics
from .util import TuningError, number


IDENTITY_COLUMNS = ("Op Name", "Op Type", "Device_id", "Stream Id", "Model Id")
COLUMN_ALIASES = {"OP Name": "Op Name", "OP Type": "Op Type", "Device Id": "Device_id",
                  "Device ID": "Device_id", "Stream ID": "Stream Id", "Model ID": "Model Id"}


def canonical_column(name):
    name = name.strip()
    return COLUMN_ALIASES.get(name, name)


def profile_reader(stream, path):
    reader = csv.DictReader(stream)
    original = reader.fieldnames or []
    normalized = [canonical_column(name) for name in original]
    if len(set(normalized)) != len(normalized):
        raise TuningError(f"Profiling CSV has duplicate or ambiguous column aliases: {path}; columns={original!r}")
    reader.fieldnames = normalized
    return reader, original


def inspect_profiles(paths):
    result = []
    for path in paths:
        counts = {}
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader, original = profile_reader(stream, path)
            columns = [k for k in IDENTITY_COLUMNS if k in reader.fieldnames]
            if not columns:
                raise TuningError(f"No task identifier columns in {path}; columns={original!r}")
            for row in reader:
                key = tuple(row[k] for k in columns)
                counts[key] = counts.get(key, 0)+1
        result.append({"file": str(path), "columns": original,
                       "tasks": [{**dict(zip(columns, k)), "samples": v} for k, v in counts.items()]})
    return result


def read_profile(paths, selector, repeats, duration_column="Task Duration(us)", traffic_column=None):
    normalized_selector = {canonical_column(k): v for k, v in selector.items()}
    if len(normalized_selector) != len(selector):
        raise TuningError(f"Profiling selector has duplicate column aliases: {list(selector)!r}")
    duration_column = canonical_column(duration_column)
    traffic_column = canonical_column(traffic_column) if traffic_column else None
    required = [*normalized_selector, duration_column] + ([traffic_column] if traffic_column else [])
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader, original = profile_reader(stream, path)
            missing = [k for k in required if k not in reader.fieldnames]
            if missing:
                raise TuningError(f"Profiling CSV missing required columns: {path}; missing={missing!r}; available={original!r}")
            for row in reader:
                if all(row[k] == v for k, v in normalized_selector.items()):
                    records.append(row)
    # Single-node fixed graph only. Reject duplicate exports, ambiguous ops and missing iterations.
    if len(records) != repeats:
        raise TuningError(f"Expected exactly {repeats} target task samples; got {len(records)}. Inspect selectors and exported CSVs.")
    for column in IDENTITY_COLUMNS:
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
