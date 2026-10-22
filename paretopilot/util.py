import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


class TuningError(ValueError):
    pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise TuningError(f"Cannot read {path}: {exc}") from exc


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and not any(x in {".git", "__pycache__", "build"} for x in p.relative_to(root).parts))
    return digest({p.relative_to(root).as_posix() if root.is_dir() else p.name: file_hash(p) for p in files})


def number(value, label, minimum=0, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or (integer and not isinstance(value, int)):
        raise TuningError(f"Invalid {label}: {value}")
    return value


def run_command(argv, cwd, log, timeout, env=None):
    """No shell interpolation; terminate the entire build tree on Linux timeout."""
    started = time.monotonic()
    record = {"argv": list(map(str, argv)), "cwd": str(cwd), "timeout_s": timeout, "log": str(log)}
    write_json(Path(log).with_suffix(".command.json"), record)
    with Path(log).open("w", encoding="utf-8") as out:
        process = subprocess.Popen([str(x) for x in argv], cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                # Windows venv python.exe may be a launcher with a child interpreter.
                # Killing only the launcher leaves workers and inherited log handles alive.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if process.poll() is None:
                    process.kill()
            process.wait()
            raise
    if code:
        raise TuningError(f"Command exited {code}; see {log}")
    return {**record, "seconds": time.monotonic()-started}
