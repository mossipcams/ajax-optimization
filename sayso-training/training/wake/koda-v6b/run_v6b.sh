#!/usr/bin/env bash
# v6b: v6 synthetic features (cached) + real data re-staged from judged labels (spool_labels_v2).
set -euo pipefail
cd /srv/llm/data/wake/runs/koda-phonics-v6-20261018
PY=/srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/srv/llm/data/wake/hf-cache
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
CFG=koda-phonics-v6b.yaml OUT=output/koda-phonics-v6b SRC=output/koda-phonics-v6
stage() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1"; }
cpu() { env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 19 taskset -c 0-3 "$@"; }
stage config
sed -e 's/^model_name: koda-phonics-v6$/model_name: koda-phonics-v6b/' -e 's/^  real_positive: 40$/  real_positive: 20/' -e 's/^  hard_real_words: 30$/  hard_real_words: 50/' koda-phonics-v6.yaml > "$CFG"
grep -q '^model_name: koda-phonics-v6b$' "$CFG"; grep -q '^  real_positive: 20$' "$CFG"; grep -q '^  hard_real_words: 50$' "$CFG"
stage link-cached-features
mkdir -p "$OUT"
for f in positive_features_train positive_features_test negative_features_train negative_features_test background_noise_features_train background_noise_features_test \
         far_field_replay_features_train far_field_replay_features_val far_field_replay_features_test real_speech_media_features_train; do
  ln -f "$SRC/$f.npy" "$OUT/$f.npy"
done
stage real-data
cpu "$PY" -B staging/stage_real_v2.py koda-phonics-v6b
cpu "$PY" -B - "$OUT" <<'PY'
import sys, numpy as np
from pathlib import Path
from livekit.wakeword.data import features as F
from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path
out = Path(sys.argv[1])
x = F.extract_features_from_directory(clip_dir=out / "real_positive_train", mel_frontend=MelSpectrogramFrontend(onnx_path=get_mel_model_path()),
    speech_embedding=SpeechEmbedding(onnx_path=get_embedding_model_path()), clean=True, expected_samples=32000)
np.save(out / "real_positive_features_train.npy", x); print("real_positive", x.shape)
PY
stage mine
rm -f "$OUT"/hard_real_words_features_train.* "$OUT"/hard_real_words_mined_seen.json
cpu "$PY" -m livekit.wakeword mine "$CFG" --hits staging/hard_real_koda-phonics-v6b/hits.jsonl --audio-root staging/hard_real_koda-phonics-v6b --out-summary staging/mine_summary_v6b.json
cat staging/mine_summary_v6b.json
stage train
/srv/llm/bin/gpu train wake --serve-after env PYTHONPATH="$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 "$PY" -m livekit.wakeword train "$CFG"
stage export
cpu "$PY" -m livekit.wakeword export "$CFG"
stage eval
cpu "$PY" -m livekit.wakeword eval "$CFG"
stage DONE
