#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import evals.runner as runner
from evals.adapters import InMemoryAdapter
from evals.cases import select_cases
from evals.outcomes import PRODUCTION_MAX_OUTPUT_TOKENS, PRODUCTION_TEMPERATURE
from evals.scorer import summarize
from eval_baseline import baseline_failures

SUITES = ("smoke", "promotion", "context", "dev", "regression")
GATED_SUITES = {"smoke", "promotion", "context"}


def embedded_template_transform():
    # ponytail: lifted from the integration's source, not copied, so the eval can't drift from it;
    # inference.py imports Home Assistant, so a plain import isn't possible here.
    path = ROOT / "custom_components/sayso/inference.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_messages_for_embedded_template")
    namespace: dict[str, Any] = {"json": json, "Any": Any}
    exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def gguf_predictor(path: str, threads: int):
    """Mirror the integration's EmbeddedEngine: llama-cpp-python in-process, same template transform."""
    import llama_cpp

    llm = llama_cpp.Llama(model_path=path, n_ctx=8192, n_threads=threads, verbose=False)
    transform = embedded_template_transform()

    def predict(messages, tools):
        kwargs = {"messages": transform(messages), "temperature": PRODUCTION_TEMPERATURE,
                  "max_tokens": PRODUCTION_MAX_OUTPUT_TOKENS}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        return llm.create_chat_completion(**kwargs)

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return predict, {"runtime": f"llama-cpp-python {llama_cpp.__version__}", "artifact_sha256": digest.hexdigest()}


def checkpoint_predictor(path: str, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    end = "<|im_end|>"

    def predict(messages, tools):
        encoded = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=PRODUCTION_MAX_OUTPUT_TOKENS, do_sample=False)
        text = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False).split(end)[0]
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return predict, {"runtime": f"transformers fp32 {torch.__version__}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--profile", choices=("checkpoint", "canary"), default="checkpoint")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "training/configs/training_baseline.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    build = gguf_predictor if args.checkpoint.endswith(".gguf") else checkpoint_predictor
    predict, runtime = build(args.checkpoint, args.threads)
    runner.RESULTS_DIR = args.out
    adapter = InMemoryAdapter(predict)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["eval"]
    failures: list[str] = []
    for label, cases in [(args.suite, select_cases(suite=args.suite)), *[(tag, select_cases(tag=tag)) for tag in args.tag]]:
        results = runner.evaluate(cases, adapter)
        summary = summarize(results)
        dest = runner.write_run(
            cases=cases,
            results=results,
            adapter=adapter,
            suite=label if label in GATED_SUITES else None,
            extra_metadata={"checkpoint": args.checkpoint, "gate_label": label, **runtime},
        )
        print(f"{label}: {summary['passed']}/{summary['total']} passed -> {dest}")
        from evals.cases import load_gates
        from evals.scorer import check_gates

        gate_suite = args.suite if label == args.suite and args.suite in GATED_SUITES else None
        if gate_suite:
            gates = baseline["canary"]["gates"] if args.profile == "canary" else load_gates(gate_suite)
            failures.extend(
                f"{label} {failure}"
                for failure in check_gates(summary, gates, expected_count=len(cases))
            )
        if label != "smoke":
            continue
        failures.extend(baseline_failures(summary, baseline, args.profile))

    for failure in failures:
        print(f"EVAL FAIL: {failure}", file=sys.stderr)
    print(f"EVAL: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
