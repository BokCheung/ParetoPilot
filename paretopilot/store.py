"""Versioned atomic experiment artifacts and manifest-scoped evaluation cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import EVALUATOR_VERSION, __version__
from .agent import AGENT_VERSION
from .config import ConfigError, fingerprint, load_json


class CacheError(ConfigError):
    """An invalid experiment artifact must abort rather than reject a candidate."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class RunStore:
    def __init__(self, directory, inputs, resume=False):
        self.root = Path(directory)
        manifest = {"format_version": 1, "package_version": __version__,
                    "evaluator_version": EVALUATOR_VERSION, "agent_version": AGENT_VERSION,
                    "inputs": inputs}
        manifest["run_id"] = fingerprint(manifest)
        manifest_path = self.root / "manifest.json"
        if self.root.exists() and not self.root.is_dir():
            raise ConfigError(f"Output is not a directory: {self.root}")
        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise ConfigError(f"Output directory is not empty: {self.root}; choose a new directory or use --resume")
            if not manifest_path.exists() or load_json(manifest_path) != manifest:
                raise ConfigError("Resume requires identical normalized inputs, search settings and evaluator versions")
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
        self.cache_hits = 0

    def evaluate(self, model, npu, workload, evaluator):
        key = fingerprint({"model": model, "npu": npu, "workload": workload, "evaluator_version": EVALUATOR_VERSION})
        path = self.root / "evaluations" / (key + ".json")
        if path.exists():
            try:
                cached = load_json(path)
            except ConfigError as exc:
                raise CacheError(f"Invalid cached evaluation: {path}: {exc}") from exc
            result = cached.get("result") if isinstance(cached, dict) else None
            if not isinstance(result, dict) or cached.get("checksum") != fingerprint(result) or result.get("evaluation_id") != key:
                raise CacheError(f"Invalid cached evaluation: {path}")
            self.cache_hits += 1
            return result
        result = evaluator(model, npu, workload)
        write_json(path, {"checksum": fingerprint(result), "result": result})
        return result

    def checkpoint(self, trials):
        write_json(self.root / "progress.json", {"completed_trials": len(trials), "trials": trials})
