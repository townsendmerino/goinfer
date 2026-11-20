#!/usr/bin/env bash
# build_webgpu_serve.sh build <tag> <out>  — build goinfer-serve with the WebGPU backend for this host (cgo, native, no cross-compile).
# build_webgpu_serve.sh smoke <bin> <want> — start the built binary and read back its backend and adapter.
#
# build: ./backend/webgpu/cmd/serve against THIS checkout, through a go.work outside it (so no tracked file changes and the VCS
# stamp reads unmodified), with the tag injected as serve's --version. wgpu-native links statically from the binding's libs-*
# modules, so nothing ships beside the binary; it links the C library dynamically and loads the Vulkan or DX12 driver at run time.
#
# smoke <want>: `adapter` requires `check --hardware` to report an opened WebGPU adapter (a linux runner with Mesa installed);
# `any` accepts a "no usable adapter" line (a windows runner may expose none). Either way `--version` must list webgpu among the
# compiled-in backends. A CI runner's adapter is a software one: this proves the binary starts and opens a device, not that it
# picks a hardware adapter.
set -euo pipefail

mode="${1:-}"
case "$mode" in
  build)
    tag="$2" out="$3"
    root="$PWD"
    command -v cygpath >/dev/null 2>&1 && root="$(cygpath -m "$PWD")" # Go on windows reads D:/a/..., not /d/a/...
    case "$out" in /*|[A-Za-z]:*) ;; *) out="$root/$out" ;; esac
    gowork="$root/../webgpu-assets.go.work"
    rm -f "$gowork" "$gowork.sum"
    GOWORK="$gowork" go work init . ./backend/webgpu
    mkdir -p "$(dirname "$out")"
    echo "building $out from backend/webgpu/cmd/serve (tags=gpu, CGO_ENABLED=1, tag $tag)"
    ( cd backend/webgpu && CGO_ENABLED=1 GOWORK="$gowork" go build -tags gpu -trimpath \
        -ldflags="-s -w -X github.com/townsendmerino/goinfer/internal/app/serveapp.injectedVersion=$tag" \
        -o "$out" ./cmd/serve )
    ls -l "$out"
    ;;
  smoke)
    bin="$2" want="$3"
    ver="$("$bin" --version)"
    echo "$ver"
    echo "$ver" | grep -qE '^backends:.*\bwebgpu\b' || { echo "FAIL: $bin --version does not list webgpu among its backends"; exit 1; }
    hw="$("$bin" check --hardware --no-selftest)"
    echo "$hw"
    line="$(echo "$hw" | grep -E '^  webgpu: (adapter|no usable adapter):' | head -1 || true)"
    [ -n "$line" ] || { echo "FAIL: check --hardware printed no webgpu adapter line"; exit 1; }
    if [ "$want" = adapter ] && ! echo "$line" | grep -q '^  webgpu: adapter:'; then
      echo "FAIL: no WebGPU adapter opened: $line"; exit 1
    fi
    echo "smoke ok: $line"
    ;;
  *)
    echo "usage: $0 build <tag> <out> | smoke <bin> adapter|any" >&2; exit 2 ;;
esac
