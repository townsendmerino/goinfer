#!/usr/bin/env bash
# build-embed.sh — build the single-file "entire LLM in one binary" demo.
#
# Compresses a GGUF model with zstd, //go:embed-s it, and cross-compiles a
# static, no-cgo binary per target. The model is NOT committed (gitignored);
# pass its path. Outputs land in demo/chat/dist/.
#
# Usage:
#   ./build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
#   ./build-embed.sh <model.gguf> darwin/arm64 linux/amd64 windows/amd64
#
# With no targets it builds for the host. zstd level is tunable via ZSTD_LEVEL
# (default 19; the model dominates the binary, so max compression is worth it).
set -euo pipefail

MODEL="${1:?usage: build-embed.sh <model.gguf> [os/arch ...]}"
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
ASSET="$DIR/model.gguf.zst"
LEVEL="${ZSTD_LEVEL:-19}"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi
command -v zstd >/dev/null || { echo "zstd CLI required (brew install zstd)" >&2; exit 1; }

echo "compressing $(basename "$MODEL") -> model.gguf.zst (zstd -$LEVEL)"
zstd "-$LEVEL" -T0 -f -o "$ASSET" "$MODEL"
printf "  %s -> %s\n" \
  "$(du -h "$MODEL"  | cut -f1)" \
  "$(du -h "$ASSET" | cut -f1)"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=("$(go env GOOS)/$(go env GOARCH)"); fi

mkdir -p "$DIR/dist"
for t in "${TARGETS[@]}"; do
  os="${t%/*}"; arch="${t#*/}"
  out="$DIR/dist/goinfer-chat-$os-$arch"
  [ "$os" = "windows" ] && out="$out.exe"
  echo "building $os/$arch -> dist/$(basename "$out")"
  CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" \
    go build -tags embed -ldflags="-s -w" -trimpath -o "$out" "$DIR"
done
echo "done:"
ls -lah "$DIR/dist"
