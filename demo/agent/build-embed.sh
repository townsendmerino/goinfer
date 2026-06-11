#!/usr/bin/env bash
# build-embed.sh — build the agent demos with the model baked into the binary.
#
# The model is NOT committed (gitignored); pass the path to a GGUF — the
# demo is tuned for Qwen2.5-Coder-0.5B-Instruct q4_k_m, same as demo/chat.
# The GGUF is staged once into internal/embedmodel/ and shared by both
# commands. (Scaffold note: gguf-embed only for now; port chat's prequant
# mode here once the demo settles — see demo/chat/build-embed.sh.)
#
# The companion ken binary is built separately, in the ken repo:
#   ken build-index /tmp/go-stdlib-curated -o demos/go-stdlib/index.bin \
#       --mode=hybrid --chunker=regex --model ~/.ken/model
#   CGO_ENABLED=0 go build -tags=kendemo -o ken-demo-go-stdlib ./demos/go-stdlib
# (full steps incl. corpus assembly: ken/demos/README.md)
#
# Usage:
#   ./build-embed.sh <model.gguf> [os/arch ...]
# With no targets it builds for the host. Builds both stdlib-agent (CLI)
# and agent-web (browser UI) into dist/.
set -euo pipefail

MODEL="${1:?usage: build-embed.sh <model.gguf> [os/arch ...]}"
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi

# //go:embed needs the asset inside the package dir and does not follow symlinks.
echo "staging $(basename "$MODEL") -> internal/embedmodel/model.gguf ($(du -h "$MODEL" | cut -f1))"
cp "$MODEL" "$DIR/internal/embedmodel/model.gguf"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=("$(go env GOOS)/$(go env GOARCH)"); fi

mkdir -p "$DIR/dist"
for t in "${TARGETS[@]}"; do
  os="${t%/*}"; arch="${t#*/}"
  for cmd in stdlib-agent agent-web; do
    out="$DIR/dist/$cmd-$os-$arch"
    [ "$os" = windows ] && out="$out.exe"
    echo "building $out"
    ( cd "$DIR" && CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" go build -tags=embed -o "$out" "./cmd/$cmd" )
  done
done
echo "done — binaries in $DIR/dist/"
