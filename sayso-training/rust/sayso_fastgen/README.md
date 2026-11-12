# sayso_fastgen

Optional Rust (PyO3) acceleration for the SFT corpus generator. Spike for evaluating a Rust port.

Currently provides one function, `dump_entities`, a byte-identical replacement for the PyYAML
emission in `generators/context.py::_dump` (the static-context block in every system prompt).
It returns `None` for anything it does not cover (tabs, line breaks, non-BMP or non-printable
characters, empty lists), and `_dump` then falls back to PyYAML. Without the wheel installed, the
generator runs unchanged.

```bash
pip install maturin
cd sayso-training/rust/sayso_fastgen
maturin build --release
pip install target/wheels/sayso_fastgen-*.whl
```

Parity is enforced by `training/tests/test_fastgen_yaml_parity.py` (fuzzed against PyYAML) and was
checked on a full `full_sft_v13` 40k corpus (identical sha256 with and without the extension).
