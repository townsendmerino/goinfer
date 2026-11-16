#!/bin/bash
# A serve binary that loads a checkpoint's Metal .giw bundle when its .gguf is absent from ~/models (the local bench set keeps the bundles; serve
# opens the .gguf first to validate its sidecar and exits when it is missing). Reads only ~/models; copies nothing; the bundle loaded is the one
# the sidecar would have loaded. Set SERVE_WRAP_REAL to the real serve binary; every argument passes through, and `-model name=X.gguf` becomes
# `-model name=X.int4.metal.v17.giw` when X.gguf is absent and the bundle is present. exec, so the harness signals the real process.
real="${SERVE_WRAP_REAL:?SERVE_WRAP_REAL names the serve binary}"
args=()
for a in "$@"; do
  case "$a" in
    *=*.gguf)
      p="${a#*=}"; giw="${p%.gguf}.int4.metal.v17.giw"
      if [ ! -e "$p" ] && [ -e "$giw" ]; then a="${a%%=*}=$giw"; fi ;;
  esac
  args+=("$a")
done
exec "$real" "${args[@]}"
