#!/usr/bin/env bash
# OP2b step 2, tier 1 input (docs/tasks/task-forward-architecture-2026-10.md §4.7):
# nobara builds NVIDIA-Nemotron-Nano-9B-v2's int4 .giw from the registered Q8_0 GGUF
# (GOINFER_NEMOTRON_GGUF), and the Mac copies only the .giw into its ~/models.
#
# Run from the Mac:  bash docs/measurements/op2b-step2-2026-10/make-nemotron-giw.sh build
#                    bash docs/measurements/op2b-step2-2026-10/make-nemotron-giw.sh pull
# `build` exits at once if nobara's timing lock is held (the check and the work are one ssh).
# It holds the lock itself for the conversion, so no timed run overlaps it. `pull` checks
# the lock in the same ssh as each read, and that the Mac has room for the bundle plus 2 GB.
set -euo pipefail
GGUF_NAME=nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
GIW_NAME=nemotron-nano-9b-v2-int4-metal.giw
REMOTE_DIR=models/nemotron           # under nobara's ~
LOCAL_DIR="$HOME/models/nemotron"

case "${1:-}" in
build)
  ssh -o ConnectTimeout=8 nobara GGUF_NAME="$GGUF_NAME" GIW_NAME="$GIW_NAME" REMOTE_DIR="$REMOTE_DIR" bash -s <<'REMOTE'
set -euo pipefail
cd ~/mycode/goinfer
python3 scripts/timing_lock.py status || { echo "nobara is measuring; not touching it"; exit 3; }
mkdir -p ~/"$REMOTE_DIR"
src=~/"$REMOTE_DIR/$GGUF_NAME"
if [ ! -s "$src" ]; then
  # The archive is storage: copy it to ~/models, never read it in place.
  [ -s "/srv/models/nemotron/$GGUF_NAME" ] || { echo "no $GGUF_NAME in ~/models or /srv/models/nemotron"; exit 4; }
  rsync -a --partial "/srv/models/nemotron/$GGUF_NAME" ~/"$REMOTE_DIR/"
fi
# prequant from origin/main in a detached worktree, so this box's checkout is not moved.
git fetch -q origin
wt=$(mktemp -d /tmp/prequant-wt.XXXXXX)
git worktree add -q --detach "$wt" origin/main
rev=$(git -C "$wt" rev-parse --short HEAD)
(cd "$wt" && GOWORK=off go build -o "$wt/prequant" ./cmd/prequant)
out=~/"$REMOTE_DIR/$GIW_NAME"
start=$(date +%s)
python3 scripts/timing_lock.py run --label nemotron-int4-giw -- \
  "$wt/prequant" -target metal -quant int4 -o "$out.partial" "$src"
mv "$out.partial" "$out"
echo "built $out at origin/main $rev in $(( $(date +%s) - start )) s"
ls -l "$out"; (cd "$(dirname "$out")" && sha256sum "$GIW_NAME" | tee "$GIW_NAME.sha256")
git worktree remove --force "$wt"
REMOTE
  ;;
pull)
  need=$(ssh -o ConnectTimeout=8 nobara "cd ~/mycode/goinfer && python3 scripts/timing_lock.py status >/dev/null && stat -c %s ~/$REMOTE_DIR/$GIW_NAME") \
    || { echo "nobara busy, or the .giw is not built"; exit 3; }
  free=$(( $(df -k "$HOME" | awk 'NR==2{print $4}') * 1024 ))
  [ "$free" -gt $(( need + (2 << 30) )) ] || { echo "Mac has $((free >> 20)) MiB free; the .giw needs $((need >> 20)) MiB + 2 GiB"; exit 5; }
  mkdir -p "$LOCAL_DIR"
  # One ssh per file: the lock check and the read are the same command.
  for f in "$GIW_NAME.sha256" "$GIW_NAME"; do
    ssh -o ConnectTimeout=8 nobara "cd ~/mycode/goinfer && python3 scripts/timing_lock.py status >/dev/null && cat ~/$REMOTE_DIR/$f" \
      > "$LOCAL_DIR/$f.partial" || { echo "nobara started measuring, or the read failed; not reading from it"; exit 3; }
  done
  mv "$LOCAL_DIR/$GIW_NAME.sha256.partial" "$LOCAL_DIR/$GIW_NAME.sha256"
  mv "$LOCAL_DIR/$GIW_NAME.partial" "$LOCAL_DIR/$GIW_NAME"
  (cd "$LOCAL_DIR" && shasum -a 256 -c "$GIW_NAME.sha256")
  ;;
*) echo "usage: $0 build|pull"; exit 2 ;;
esac
