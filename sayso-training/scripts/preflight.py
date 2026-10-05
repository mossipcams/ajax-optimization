"""Run upstream preflight against this training tree and its configuration."""

import importlib.util

from training_paths import ROOT, SAYSO, configure

configure()
# Bind the local packages before upstream preflight prepends its training path.
import adapters
import generators

spec = importlib.util.spec_from_file_location(
    "_sayso_preflight", SAYSO / "scripts" / "preflight.py"
)
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
upstream.ROOT = ROOT
configure()
inspect_dataset = upstream.inspect_dataset

if __name__ == "__main__":
    raise SystemExit(upstream.main())
