
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from generators.audit import audit_corpus
from generators.config import GeneratorConfig
from generators.pipeline import run_build, run_generation, write_jsonl
from generators.manifest import write_manifest


def _audit(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m generators.cli audit",
        description="Run the hard label gates over a corpus .jsonl and print a JSON summary",
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    args = parser.parse_args(argv)
    print(json.dumps(audit_corpus(args.corpus, limit=args.limit), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "audit":
        return _audit(argv[1:])
    parser = argparse.ArgumentParser(description="Generate SaySo synthetic training dataset")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML recipe under training/configs/generation/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate in memory only; do not write output files",
    )
    parser.add_argument("--output", type=Path, help="Override recipe output path")
    parser.add_argument("--manifest", type=Path, help="Override recipe manifest path")
    args = parser.parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        candidate = TRAINING_ROOT / config_path
        if candidate.is_file():
            config_path = candidate
        elif (REPO_ROOT / config_path).is_file():
            config_path = REPO_ROOT / config_path

    config = GeneratorConfig.from_yaml(config_path, repo_root=REPO_ROOT)
    if args.output:
        config.output_path = args.output.resolve()
    if args.manifest:
        config.manifest_path = args.manifest.resolve()

    if config.real_home_path and not config.synthetic_only:
        from generators.real_home import TESTABLE_EXPOSURE_SOURCES, require_exposure_source

        require_exposure_source(config.real_home_path, allowed=TESTABLE_EXPOSURE_SOURCES)

    if args.dry_run:
        result = run_generation(config)
        print(json.dumps({"accepted": len(result["rows"]), "manifest": result["manifest"]}, indent=2))
        return 0

    result = run_build(config)
    print(
        json.dumps(
            {
                "accepted": len(result["rows"]),
                "output": result.get("output_path"),
                "manifest": result.get("manifest_path"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
