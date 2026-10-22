"""Portable source archive; excludes runs, build products and machine state."""
from pathlib import Path
import zipfile
from .util import TuningError, file_hash


def create_package(output):
    root = Path(__file__).resolve().parent.parent
    if not (root/"examples").is_dir() or not (root/"README.md").is_file():
        raise TuningError("Run package from the source checkout, not an installed wheel")
    files = [root/"pyproject.toml", root/"README.md", root/".gitignore"]
    for name in ("paretopilot", "examples", "docs", "tests"):
        files += [p for p in (root/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    output = Path(output).resolve()
    if output.exists():
        raise TuningError("Archive already exists; choose another output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            entry = zipfile.ZipInfo("ParetoPilot/"+path.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    sha = file_hash(output)
    output.with_suffix(output.suffix+".sha256").write_text(sha+"  "+output.name+"\n", encoding="utf-8")
    return {"archive": str(output), "sha256": sha, "files": len(files)}
