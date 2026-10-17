import importlib.machinery
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_sayso():
    loader = importlib.machinery.SourceFileLoader("sayso_cli", str(ROOT / "sayso"))
    spec = importlib.util.spec_from_loader("sayso_cli", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summaries(*counts):
    return {f"gate-{index}": {"summary": {"passed": count, "total": count + 2}, "promotion": None}
            for index, count in enumerate(counts)}


def test_passed_count_sums_gate_passed():
    module = load_sayso()
    assert module.passed_count(summaries(3, 4)) == 7
    assert module.passed_count(None) == 0
    assert module.passed_count({}) == 0


def test_passed_count_ignores_missing_keys():
    module = load_sayso()
    assert module.passed_count({"gate": {"summary": {"total": 5}}}) == 0
    assert module.passed_count({"gate": {"total": 5}}) == 0


def test_rank_candidates_context_then_smoke_then_earlier_step():
    module = load_sayso()
    rows = [
        {"candidate": "final", "step": float("inf"),
         "context": module.passed_count(summaries(13)), "smoke": module.passed_count(summaries(24))},
        {"candidate": "checkpoint-600", "step": 600,
         "context": module.passed_count(summaries(13)), "smoke": module.passed_count(summaries(24))},
        {"candidate": "checkpoint-800", "step": 800,
         "context": module.passed_count(summaries(9)), "smoke": None},
        {"candidate": "checkpoint-400", "step": 400,
         "context": module.passed_count(summaries(13)), "smoke": None},
    ]
    ranked = module.rank_candidates(rows)
    assert [row["candidate"] for row in ranked] == [
        "checkpoint-600", "final", "checkpoint-400", "checkpoint-800"
    ]


def test_rank_candidates_final_tie_earlier_step_wins():
    module = load_sayso()
    rows = [
        {"candidate": "final", "step": float("inf"),
         "context": module.passed_count(summaries(13)), "smoke": module.passed_count(summaries(24))},
        {"candidate": "checkpoint-600", "step": 600,
         "context": module.passed_count(summaries(13)), "smoke": module.passed_count(summaries(24))},
    ]
    ranked = module.rank_candidates(rows)
    assert [row["candidate"] for row in ranked] == ["checkpoint-600", "final"]


def test_rank_candidates_smoke_breaks_context_tie_only_among_scored():
    module = load_sayso()
    rows = [
        {"candidate": "checkpoint-200", "step": 200,
         "context": module.passed_count(summaries(10)), "smoke": module.passed_count(summaries(5))},
        {"candidate": "checkpoint-400", "step": 400,
         "context": module.passed_count(summaries(11)), "smoke": None},
    ]
    ranked = module.rank_candidates(rows)
    assert [row["candidate"] for row in ranked] == ["checkpoint-400", "checkpoint-200"]


def test_trainer_args_save_total_limit_env(monkeypatch):
    monkeypatch.delenv("SAYSO_TRAINING_HOST_ROOT", raising=False)
    monkeypatch.delenv("TRAIN_MODEL", raising=False)
    monkeypatch.delenv("TRAIN_LR", raising=False)
    monkeypatch.delenv("TRAIN_WARMUP_RATIO", raising=False)
    module = load_sayso()
    assert module.trainer_args() == ["--lr", "2e-5"]
    monkeypatch.setenv("TRAIN_SAVE_TOTAL_LIMIT", "0")
    assert module.trainer_args() == ["--lr", "2e-5", "--save-total-limit", "0"]


def test_find_candidates_reads_host_dir_and_returns_container_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("SAYSO_TRAINING_HOST_ROOT", raising=False)
    module = load_sayso()
    out_dir = tmp_path / "host" / "runs" / "r1-full"
    for name in ("checkpoint-200", "checkpoint-100"):
        (out_dir / name).mkdir(parents=True)
    (out_dir / "final").mkdir()
    monkeypatch.setattr(module, "HOST_LFM", str(tmp_path / "host"))
    steps = module.find_candidates(out_dir)
    assert [step for step, _ in steps] == [100, 200, float("inf")]
    assert [path for _, path in steps] == [
        "/workspace/host/lfm/runs/r1-full/checkpoint-100",
        "/workspace/host/lfm/runs/r1-full/checkpoint-200",
        "/workspace/host/lfm/runs/r1-full/final",
    ]


def test_find_candidates_no_final_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("SAYSO_TRAINING_HOST_ROOT", raising=False)
    module = load_sayso()
    out_dir = tmp_path / "host" / "runs" / "r1-full"
    (out_dir / "checkpoint-50").mkdir(parents=True)
    monkeypatch.setattr(module, "HOST_LFM", str(tmp_path / "host"))
    assert module.find_candidates(out_dir) == [
        (50, "/workspace/host/lfm/runs/r1-full/checkpoint-50")
    ]
