"""Strict JSON, content fingerprints and atomic experiment records."""
import hashlib
import json
import math
import os
from pathlib import Path


class ConfigError(ValueError):
    pass


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ConfigError(f"Cannot read JSON {path}: {exc}") from exc


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def number(value, name, minimum=0, integer=False, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{name} must be a finite number")
    if (value <= minimum if strict else value < minimum) or (integer and not isinstance(value, int)):
        raise ConfigError(f"Invalid {name}: {value}")
    return value
