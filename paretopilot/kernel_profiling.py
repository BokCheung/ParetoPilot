# Strict per-trial direct-kernel profiler parsing.
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .profiling import IDENTITY_COLUMNS, profile_reader, read_profile
from .util import TuningError


CORE_TASKS = {"AI_CORE", "AI_CUBE", "AI_CUBE_CORE", "AI_VECTOR_CORE", "MIX_AIC"}


def check_block_dims(rows, expected):
    observed, checked, unavailable, zero_reported = [], 0, 0, 0
    for row in rows:
        raw = row.get("Block Dim")
        value = "" if raw is None else str(raw).strip()
        if value not in observed:
            observed.append(value)
        if value.upper() in {"", "N/A", "NA", "-", "--", "NONE", "NULL", "UNKNOWN"}:
            unavailable += 1
            continue
        try:
            number = Decimal(value)
            if not number.is_finite() or number < 0 or number != number.to_integral_value():
                raise ValueError()
        except (InvalidOperation, ValueError) as exc:
            raise TuningError(f"Invalid profiler Block Dim {raw!r}; expected integer {expected} or an unavailable marker") from exc
        if number == 0:
            # Preserve zero as unverified profiler metadata, never as a match
            # to the positive Host launch dimension or proof of a zero launch.
            zero_reported += 1
            unavailable += 1
            continue
        if number != expected:
            raise TuningError(f"Direct profiler Block Dim {raw!r} differs from Host tiling {expected}")
        checked += 1
    return {"status": "matched" if not unavailable else "partial" if checked else "unavailable",
            "expected": expected, "observed": observed, "checked_samples": checked,
            "unavailable_samples": unavailable, "zero_reported_samples": zero_reported}


def direct_profile(paths, repeats, block_dim):
    """The isolated capture contains exactly one launched kernel per iteration.

    Raw launches may have unknown Op Type/Name. Never select by fastest task,
    assume a V3 name, or silently accept extra core tasks/duplicate exports.
    """
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader, _ = profile_reader(stream, path)
            if "Task Type" not in reader.fieldnames:
                raise TuningError("Direct profiling needs Task Type to isolate core tasks; inspect profile_inventory.json")
            rows.extend(row for row in reader if row["Task Type"].strip().upper() in CORE_TASKS)
    if len(rows) != repeats:
        raise TuningError(f"Direct capture expected {repeats} core tasks, got {len(rows)}; inspect profile_inventory.json")
    selector = {k: rows[0][k] for k in (*IDENTITY_COLUMNS, "Task Type") if k in rows[0]}
    if any(any(row.get(k) != v for k, v in selector.items()) for row in rows):
        raise TuningError("Direct capture contains ambiguous core task identities; no target was guessed")
    block_check = check_block_dims(rows, block_dim)
    result = read_profile(paths, selector, repeats, "Task Duration(us)")
    result.update(metric="Direct target-kernel Task Duration(us) median",
                  identity_basis="isolated single-kernel capture; exact count, identity and available Block Dim checked",
                  task_rows=rows, block_dim_check=block_check)
    return result
