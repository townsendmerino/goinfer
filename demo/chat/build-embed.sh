#!/usr/bin/env bash
# build-embed.sh — build the single-file "entire LLM in one binary" demo.
#
# Two embed modes (the model is NOT committed — gitignored; pass its path):
#
#   PREQUANT (default): bake a prequant bundle (.giw) — the int8 resident weights
#     pre-serialized + a metadata-only GGUF for the tokenizer. The binary maps the
#     weights straight from its image (no dequant/requant, no heap copy): ~5×
#     faster cold start and ~10× less heap RAM. Bigger asset (int8 ≈ +30% vs q4).
#
#   --gguf: bake the raw GGUF and quantize at launch. Smaller asset; slower start
#     and a full-size weight heap. Use if asset size matters more than RAM/speed.
#
# Cross-compiles a static, no-cgo binary per target into demo/chat/dist/.
#
# Usage:
#   ./build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
#   ./build-embed.sh <model.gguf> darwin/arm64 darwin/amd64 linux/amd64 linux/arm64 windows/amd64
#   ./build-embed.sh --gguf <model.gguf> [os/arch ...]
#
# With no targets it builds for the host.
set -euo pipefail

MODE=prequant
if [ "${1:-}" = "--gguf" ]; then MODE=gguf; shift; fi
MODEL="${1:?usage: build-embed.sh [--gguf] <model.gguf> [os/arch ...]}"
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi

# //go:embed needs the asset inside the package dir and does not follow symlinks.
if [ "$MODE" = prequant ]; then
  echo "building prequant bundle -> model.giw"
  ( cd "$ROOT" && go run ./cmd/prequant -o "$DIR/model.giw" "$MODEL" )
  TAGS=prequant
else
  echo "staging $(basename "$MODEL") -> model.gguf ($(du -h "$MODEL" | cut -f1))"
  cp "$MODEL" "$DIR/model.gguf"
  TAGS=embed
fi

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=("$(go env GOOS)/$(go env GOARCH)"); fi

mkdir -p "$DIR/dist"
for t in "${TARGETS[@]}"; do
  os="${t%/*}"; arch="${t#*/}"
  out="$DIR/dist/goinfer-chat-$os-$arch"
  [ "$os" = "windows" ] && out="$out.exe"
  echo "building $os/$arch ($TAGS) -> dist/$(basename "$out")"
  CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" \
    go build -tags "$TAGS" -ldflags="-s -w" -trimpath -o "$out" "$DIR"
done
echo "done:"
ls -lah "$DIR/dist"
