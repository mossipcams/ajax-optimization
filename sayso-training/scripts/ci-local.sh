#!/usr/bin/env bash
set -euo pipefail

MODE=local
REMOTE=
RUN_DIR=
HEAD_SHORT=
MAIN_SHORT=

while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE=${2:?usage: ci-local.sh --remote USER@HOST}; shift 2 ;;
    --post) MODE=post; shift ;;
    --run) MODE=run; RUN_DIR=${2:?}; HEAD_SHORT=${3:?}; MAIN_SHORT=${4:?}; shift 4 ;;
    *) echo "usage: ci-local.sh [--remote USER@HOST | --post]" >&2; exit 2 ;;
  esac
done

run_phase() {
  local dir=$1
  local venv=${SAYSO_CI_VENV:-$HOME/.cache/sayso-ci/venv}
  local want summary counts status=0
  command -v python3.12 >/dev/null || { echo "local-ci: python3.12 not on PATH" >&2; exit 2; }
  cd "$dir"
  test -f vendor/SaySo/sayso_contract.py || { echo "local-ci: vendor/SaySo missing from export" >&2; exit 2; }
  grep -E '^(hassil|jsonschema|PyYAML|hypothesis|pytest|transformers|tokenizers|voluptuous|Jinja2)==' training/requirements.txt > ci-requirements.txt
  want=$(python3 -c 'import hashlib; print(hashlib.sha256(open("ci-requirements.txt", "rb").read()).hexdigest())')
  if [ ! -x "$venv/bin/python" ] || [ "$(cat "$venv/ci-pins.sha256" 2>/dev/null || true)" != "$want" ]; then
    python3.12 -m venv --clear "$venv"
    "$venv/bin/python" -m pip install --quiet -r ci-requirements.txt
    printf '%s\n' "$want" > "$venv/ci-pins.sha256"
  fi
  export PYTHONPATH=$("$venv/bin/python" training_paths.py)
  nice -n 19 "$venv/bin/python" -m pytest training/tests -q 2>&1 | tee pytest.log || status=$?
  summary=$(grep -E ' in [0-9.]+s' pytest.log | tail -n 1 || true)
  count() { printf '%s\n' "$summary" | tr -d ',' | awk -v w="$1" '{for (i = 2; i <= NF; i++) if ($i == w || $i == w "s") c = $(i - 1)} END {print c + 0}'; }
  counts="$(count passed) passed, $(count skipped) skipped, $(count failed) failed, $(count error) errors"
  if [ -z "$summary" ]; then
    status=1
    counts="no pytest summary"
  fi
  if [ "$status" -eq 0 ]; then
    echo "local-ci: PASS $counts head=$HEAD_SHORT main=$MAIN_SHORT"
  else
    echo "local-ci: FAIL $counts head=$HEAD_SHORT main=$MAIN_SHORT"
  fi
  exit "$status"
}

local_flow() {
  local root git_dir head_sha main_sha tree pin stage rec_dir log rdir="" status line
  root=$(git rev-parse --show-toplevel)
  git_dir=$(git rev-parse --absolute-git-dir)
  git -C "$root" fetch -q origin main
  head_sha=$(git -C "$root" rev-parse HEAD)
  main_sha=$(git -C "$root" rev-parse origin/main)
  HEAD_SHORT=$(git -C "$root" rev-parse --short HEAD)
  MAIN_SHORT=$(git -C "$root" rev-parse --short origin/main)
  tree=$(git -C "$root" merge-tree --write-tree HEAD origin/main) || { echo "local-ci: HEAD conflicts with origin/main; resolve the merge first" >&2; exit 1; }
  stage=$(mktemp -d)
  trap 'rm -rf "$stage"' EXIT
  git -C "$root" archive "$tree" sayso-training | tar -C "$stage" -xf -
  pin=$(git -C "$root" ls-tree "$tree" sayso-training/vendor/SaySo | awk '{print $3}')
  mkdir -p "$stage/sayso-training/vendor/SaySo"
  git -C "$root/sayso-training/vendor/SaySo" archive "$pin" | tar -C "$stage/sayso-training/vendor/SaySo" -xf -
  cp "$0" "$stage/sayso-training/scripts/ci-local.sh"
  rec_dir="$git_dir/sayso-ci-local"
  mkdir -p "$rec_dir"
  log="$rec_dir/$head_sha.log"
  if [ -n "$REMOTE" ]; then
    rdir=$(COPYFILE_DISABLE=1 tar -C "$stage/sayso-training" -czf - . | ssh -o BatchMode=yes "$REMOTE" 'd=$(mktemp -d) && tar -C "$d" -xzf - && echo "$d"')
  fi
  set +e
  if [ -n "$rdir" ]; then
    ssh -o BatchMode=yes "$REMOTE" "bash '$rdir/scripts/ci-local.sh' --run '$rdir' '$HEAD_SHORT' '$MAIN_SHORT'; s=\$?; rm -rf '$rdir'; exit \$s" 2>&1 | tee "$log"
  else
    bash "$stage/sayso-training/scripts/ci-local.sh" --run "$stage/sayso-training" "$HEAD_SHORT" "$MAIN_SHORT" 2>&1 | tee "$log"
  fi
  status=${PIPESTATUS[0]}
  set -e
  line=$(grep '^local-ci: ' "$log" | tail -n 1 || true)
  if [ -z "$line" ]; then
    line="local-ci: FAIL no result line head=$HEAD_SHORT main=$MAIN_SHORT"
    status=1
    echo "$line"
  fi
  printf '%s\n' "$line" > "$rec_dir/$head_sha"
  exit "$status"
}

post() {
  local head_sha rec line state description
  head_sha=$(git rev-parse HEAD)
  rec="$(git rev-parse --absolute-git-dir)/sayso-ci-local/$head_sha"
  if [ ! -f "$rec" ]; then
    echo "local-ci: no recorded result for HEAD ${head_sha:0:7}; run ci-local.sh first" >&2
    exit 1
  fi
  if [ -z "$(git branch -r --contains "$head_sha")" ]; then
    echo "local-ci: HEAD ${head_sha:0:7} is not pushed; push first" >&2
    exit 1
  fi
  line=$(cat "$rec")
  case "$line" in
    "local-ci: PASS "*) state=success ;;
    *) state=failure ;;
  esac
  description=${line#local-ci: }
  description=${description:0:140}
  gh api "repos/{owner}/{repo}/statuses/$head_sha" -f state="$state" -f context=local-ci -f description="$description" --jq '"posted local-ci \(.state) on \(.sha)"'
}

case "$MODE" in
  run) run_phase "$RUN_DIR" ;;
  post) post ;;
  *) local_flow ;;
esac
