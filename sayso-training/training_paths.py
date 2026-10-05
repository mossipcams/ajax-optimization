"""Shared Python search path for the imported tools and their SaySo checkout."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAYSO = ROOT / "vendor" / "SaySo"
PATHS = (ROOT, ROOT / "training", SAYSO, SAYSO / "satellite")


def pythonpath() -> str:
    return os.pathsep.join(dict.fromkeys([
        *(str(path) for path in PATHS),
        *(path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path),
    ]))


def configure() -> None:
    if not (SAYSO / "sayso_contract.py").is_file():
        raise RuntimeError(
            "Initialize SaySo from the repository root: "
            "git submodule update --init sayso-training/vendor/SaySo"
        )
    paths = pythonpath().split(os.pathsep)
    sys.path[:] = paths + [path for path in sys.path if path not in paths]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


if __name__ == "__main__":
    configure()
    print(pythonpath())
