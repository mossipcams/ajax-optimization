#!/usr/bin/env python3
"""Quantize an F16 GGUF with the llama-cpp-python build that serves it, and print its sha256.

The HF -> F16 step needs torch, so ./sayso runs llama.cpp's convert_hf_to_gguf.py in the
trainer container first; this script runs in the host venv next to the release eval.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize(source: Path, dest: Path, quant: str) -> None:
    if quant.upper() == "F16":
        shutil.copyfile(source, dest)
        return
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{quant.upper()}")
    if llama_cpp.llama_model_quantize(str(source).encode(), str(dest).encode(), ctypes.byref(params)) != 0:
        raise SystemExit(f"llama_model_quantize failed: {source} -> {dest} ({quant})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("f16", type=Path, help="F16 GGUF from convert_hf_to_gguf.py")
    parser.add_argument("out", type=Path)
    parser.add_argument("--quant", default="Q4_K_M", help="llama.cpp ftype suffix, e.g. Q4_K_M, Q8_0, F16")
    args = parser.parse_args()
    with args.f16.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"not a GGUF file: {args.f16}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantize(args.f16, args.out, args.quant)
    print(sha256(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
