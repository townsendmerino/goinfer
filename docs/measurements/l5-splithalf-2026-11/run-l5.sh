#!/usr/bin/env bash
# L5's registered A/B on nobara (README.md beside this script): split-half ON against OFF, BenchmarkDecode, ten ABBA pairs.
# Usage: bash docs/measurements/l5-splithalf-2026-11/run-l5.sh [path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf]
# Writes raw.jsonl and grade.txt under ~/goinfer-bench/l5/<UTC time>-<commit>/ and copies both beside this script.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
model=${1:-$( (ls ~/models/qwen15/*oder*1.5b*nstruct*q4_k_m*.gguf ~/models/qwen15/*.gguf 2>/dev/null || true) | head -1)}
case "$model" in
  "") echo "VOID: no model; pass the Qwen2.5-Coder-1.5B-Instruct Q4_K_M .gguf under ~/models" >&2; exit 1 ;;
  /srv/models/*|/Volumes/*) echo "VOID: $model is on the archive; timed runs read ~/models (docs/benchmarks.md, Model storage)" >&2; exit 1 ;;
esac
model=$(readlink -f "$model")
case "$model" in "$HOME"/models/*) ;; *) echo "VOID: $model is not under ~/models" >&2; exit 1 ;; esac

sha=$(git -C "$repo" rev-parse --short=9 HEAD)
if [ -n "$(git -C "$repo" status --porcelain -- decoder go.mod go.sum)" ]; then
  echo "VOID: decoder/ or go.mod has uncommitted changes; run from a clean tree at a commit" >&2; exit 1
fi
out=~/goinfer-bench/l5/$(date -u +%Y%m%dT%H%M%SZ)-$sha
mkdir -p "$out"
echo "L5 A/B at $sha, model $model, record $out; start $(date '+%H:%M %Z'), expected finish about 15 min later"
(cd "$repo" && go test -c -o "$out/decoder.test" ./decoder)

export GOINFER_PREQUANT_GGUF=$model GOINFER_BENCH_QUANT=int4
python3 "$repo/scripts/timing_lock.py" run --label "l5-splithalf-ab" -- \
  python3 "$here/l5_ab.py" --bin "$out/decoder.test" --out "$out" --model "$model" --commit "$sha"
cp "$out/raw.jsonl" "$out/grade.txt" "$here/"
cat "$out/grade.txt"
