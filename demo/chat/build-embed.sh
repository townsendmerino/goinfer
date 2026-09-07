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
# Output binaries are <name>-<os>-<arch>[.exe] (--name sets <name>, default
# goinfer-chat) — so two model tiers can build side by side without clobbering.
#
# Usage:
#   ./build-embed.sh ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
#   ./build-embed.sh --name goinfer-chat-1.5b --tier 1.5b <model.gguf> darwin/arm64 linux/amd64 ...
#   ./build-embed.sh --gguf <model.gguf> [os/arch ...]
#
# With no targets it builds for the host.
#
# R6 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): `--version` on a released embed-tier
# binary had no way to say what quant it actually shipped at, and --help's shared --quant text
# ("Default int4") does not apply to a PREQUANT build at all — it bakes at a fixed quant chosen
# HERE, at build time, never reading the --quant flag (internal/chatapp/prequant.go's
# loadEmbedded ignores opts.Quant entirely). QUANT below is that single source of truth: it is
# both what gets baked in and what --version reports, so the two cannot drift apart the way the
# flag's stated default and the binary's actual behavior had. GOINFER_RELEASE_TAG/GOINFER_TIER
# are read from the environment (set by release-assets.yml) rather than added as more flags,
# since a local/manual run has no tag to inject and should fall back to the toolchain's own VCS
# stamp exactly as an ordinary `go build` would.
set -euo pipefail

MODE=prequant
NAME=goinfer-chat
TIER="${GOINFER_TIER:-}"
QUANT=int8int8   # cmd/prequant's own default; kept explicit here rather than implicit so a
                 # change to one is a change to both, since it is passed to the actual bake below.
while true; do
  case "${1:-}" in
    --gguf) MODE=gguf; shift ;;
    --name) NAME="${2:?--name needs a value}"; shift 2 ;;
    --tier) TIER="${2:?--tier needs a value}"; shift 2 ;;
    --quant) QUANT="${2:?--quant needs a value}"; shift 2 ;;
    *) break ;;
  esac
done
MODEL="${1:?usage: build-embed.sh [--gguf] [--name <basename>] <model.gguf> [os/arch ...]}"
shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
# The REPL (and its //go:embed directives) live in internal/chatapp since audit M-19;
# demo/chat is now a thin CPU-only main shim. Stage the embed asset next to the directive
# there, but keep the built binary + dist under demo/chat.
PKGDIR="$ROOT/internal/chatapp"

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi

# //go:embed needs the asset inside the package dir and does not follow symlinks.
if [ "$MODE" = prequant ]; then
  echo "building prequant bundle (quant=$QUANT) -> internal/chatapp/model.giw"
  ( cd "$ROOT" && go run ./cmd/prequant -quant "$QUANT" -o "$PKGDIR/model.giw" "$MODEL" )
  TAGS=prequant
  # --gguf mode bakes the raw GGUF and quantizes at LAUNCH per --quant (default int4, and that
  # flag has its ordinary effect there) — only the prequant path fixes a quant at build time, so
  # only it injects embeddedQuant. `-X` values cannot contain spaces (go build's own ldflags
  # parser word-splits them), which is the other reason this stays a single token.
  EMBED_QUANT_LDFLAG="-X github.com/townsendmerino/goinfer/internal/chatapp.embeddedQuant=$QUANT"
else
  echo "staging $(basename "$MODEL") -> internal/chatapp/model.gguf ($(du -h "$MODEL" | cut -f1))"
  cp "$MODEL" "$PKGDIR/model.gguf"
  TAGS=embed
  EMBED_QUANT_LDFLAG=""
fi

LDFLAGS="-s -w $EMBED_QUANT_LDFLAG"
[ -n "$TIER" ] && LDFLAGS="$LDFLAGS -X github.com/townsendmerino/goinfer/internal/chatapp.embeddedTier=$TIER"
[ -n "${GOINFER_RELEASE_TAG:-}" ] && LDFLAGS="$LDFLAGS -X github.com/townsendmerino/goinfer/internal/chatapp.injectedVersion=$GOINFER_RELEASE_TAG"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=("$(go env GOOS)/$(go env GOARCH)"); fi

mkdir -p "$DIR/dist"
for t in "${TARGETS[@]}"; do
  os="${t%/*}"; arch="${t#*/}"
  out="$DIR/dist/$NAME-$os-$arch"
  [ "$os" = "windows" ] && out="$out.exe"
  echo "building $os/$arch ($TAGS) -> dist/$(basename "$out")"
  CGO_ENABLED=0 GOOS="$os" GOARCH="$arch" \
    go build -tags "$TAGS" -ldflags="$LDFLAGS" -trimpath -o "$out" "$DIR"
done
echo "done:"
ls -lah "$DIR/dist"
