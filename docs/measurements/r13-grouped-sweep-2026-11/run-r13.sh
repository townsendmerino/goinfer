#!/usr/bin/env bash
# P41's registered sweep on the Mac (README.md beside this script): R13's grouped decode attention on against off.
# Usage: bash docs/measurements/r13-grouped-sweep-2026-11/run-r13.sh <qwen2.5-1.5b-instruct q4_k_m .gguf> <qwen2.5-coder-0.5b-instruct q4_k_m .gguf>
# Both models from ~/models. Writes raw.jsonl and grade.txt under ~/goinfer-bench/r13/<UTC time>-<commit>/ and copies both
# beside this script.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
[ "$(uname -m)" = arm64 ] || { echo "VOID: this sweep is arm64 only (the grouped kernels are NEON; amd64 runs per-head in both arms)" >&2; exit 1; }
[ $# -eq 2 ] || { echo "usage: $0 <1.5B q4_k_m .gguf> <0.5B q4_k_m .gguf>" >&2; exit 1; }
for m in "$1" "$2"; do
  r=$(readlink -f "$m" 2>/dev/null || true)
  case "$r" in "$HOME"/models/*) ;; *) echo "VOID: $m is not under ~/models (docs/benchmarks.md, Model storage)" >&2; exit 1 ;; esac
  [ -f "$r" ] || { echo "VOID: no file at $m" >&2; exit 1; }
done
model=$(readlink -f "$1"); control=$(readlink -f "$2")

sha=$(git -C "$repo" rev-parse --short=9 HEAD)
if [ -n "$(git -C "$repo" status --porcelain -- decoder cmd/hostmem go.mod go.sum)" ]; then
  echo "VOID: decoder/ or go.mod has uncommitted changes; run from a clean tree at a commit" >&2; exit 1
fi
out=~/goinfer-bench/r13/$(date -u +%Y%m%dT%H%M%SZ)-$sha
mkdir -p "$out"
echo "P41 sweep at $sha, model $model, control $control, record $out; start $(date '+%H:%M %Z'), expected finish about 30 min later"
(cd "$repo" && go test -c -o "$out/decoder.test" ./decoder && go build -o "$out/hostmem" ./cmd/hostmem)

python3 "$repo/scripts/timing_lock.py" run --label "p41-r13-sweep" -- \
  python3 "$here/r13_ab.py" --bin "$out/decoder.test" --hostmem "$out/hostmem" --out "$out" \
    --model "$model" --control "$control" --commit "$sha" ${R13_PRIOR:+--control-only --prior "$R13_PRIOR"}
cp "$out/raw.jsonl" "$out/grade.txt" "$here/"
cat "$out/grade.txt"
