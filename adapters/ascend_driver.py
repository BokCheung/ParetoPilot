"""Real-engineering integration point. Never claims a synthetic run is real.

Implement the five stages in docs/ADAPTER.md against your matched CANN sources.
Each stage receives a request path and must write a response JSON path.
"""
import json
from pathlib import Path
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    response = {"status": "not_configured", "mode": "real", "stage": request["stage"],
                "error": "Connect version-matched CANN build/export/ATC/verify/profile commands; see docs/ADAPTER.md"}
    Path(sys.argv[2]).write_text(json.dumps(response), encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
