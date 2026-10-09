#!/usr/bin/env bash
# Mellum2.1 Gate 1, part 1 (docs/tasks/task-mellum21-2026-10.md): pull JetBrains/Mellum2.1-12B-A2.5B-Thinking at the pinned revision to ~/models, verify every shard against Hugging Face's own sha256, and pin
# the two parity goldens (forward and window) from the real 2.1 weights with transformers (bf16, CPU), the way scripts/pin_mellum2.py pinned 2.0's. Writes the pair under $GOLD (durable, outside the repo)
# for run-mellum21-gates.sh. Nothing here is a verdict; the verdicts are part 2's.
#   1. pull      snapshot_download at REV (24.3 GB over 5 shards; the .png/.svg are skipped)             -> ~/models/mellum2.1-thinking
#   2. verify    each shard's sha256 against the tree API's lfs oid; the five small files against the gate-0 copies (tokenizer.json etc. must be byte-identical to 2.0's)
#   3. pin       MELLUM_PIN_* env -> scripts/pin_mellum2.py: chat golden with enable_thinking=False (answer token) and the 1441-token window golden; the record carries revision, shard hashes, versions
# Estimate ~50 min (pull 5-25 min by network; sha256 of 24 GB twice ~5 min; HF bf16 load ~3 min; the 1,441-token forward 10-25 min); queue at 90:
#   python3 scripts/night.py add mellum21-pull-pin --est 90 --by "nobara session, Mellum2.1 Gate 1" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-pull-pin.sh
# MELLUM_DRY=1 checks preconditions and prints the plan. Everything lands under ~/models (local NVMe), never the archive.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REPO=JetBrains/Mellum2.1-12B-A2.5B-Thinking
REV=92ddae9fc7665e9f801d141d2e5a6b2caf2460c4
DEST=$HOME/models/mellum2.1-thinking
GOLD=${GOLD:-$HOME/goinfer-logs/mellum21-gold}
OUT=${1:-$HOME/goinfer-logs/mellum21-pull-pin-$(date +%F)}
GATE0=$HOME/goinfer-bench/mellum21/gate0
PY=$HOME/g4venv/bin/python
fatal() { echo "FATAL: $*" >&2; exit 2; }
[ -x "$PY" ] || fatal "$PY is missing"
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json; do [ -e "$GATE0/$f" ] || fatal "$GATE0/$f is missing (the gate-0 copies)"; done
case "$DEST$GOLD$OUT" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 60 ] || fatal "only ${free_gb} GB free under ~/models, need 60 (24 GB weights, headroom)"
"$PY" -I -c "import huggingface_hub, transformers, torch; print('hub', huggingface_hub.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)" || fatal "python deps"
mkdir -p "$OUT" "$GOLD"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "repo: $REPO @ $REV"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = pull to $DEST, verify, pin goldens to $GOLD"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; $(du -sm "$DEST" 2>/dev/null | cut -f1) MB in $DEST; last: $(tail -n1 "$OUT/$name.log" 2>/dev/null | cut -c1-100)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-10s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }

pull() { "$PY" -I - "$REPO" "$REV" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
p = snapshot_download(repo_id=repo, revision=rev, local_dir=dest, ignore_patterns=["*.png", "*.svg"])
print("snapshot at", p)
PY
}
verify() { "$PY" -I - "$REPO" "$REV" "$DEST" "$GATE0" <<'PY'
import hashlib, json, os, sys, urllib.request
repo, rev, dest, gate0 = sys.argv[1:5]
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1", timeout=60))
bad = 0
for e in tree:
    if e["type"] != "file" or "lfs" not in e or e["path"].endswith((".png", ".svg")): continue  # the images are not pulled
    p = os.path.join(dest, e["path"])
    if not os.path.exists(p): print("MISSING", e["path"]); bad += 1; continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    ok = h.hexdigest() == e["lfs"]["oid"] and os.path.getsize(p) == e["size"]
    print(("OK      " if ok else "BAD     ") + e["path"], os.path.getsize(p), h.hexdigest()[:16]); bad += (not ok)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "config.json"):
    same = open(os.path.join(dest, f), "rb").read() == open(os.path.join(gate0, f), "rb").read()
    print(("SAME    " if same else "DIFFERS ") + f + " against the gate-0 copy"); bad += (not same)
print("VERIFY", "FAILED" if bad else "PASSED"); sys.exit(1 if bad else 0)
PY
}
pin() { env MELLUM_PIN_PATH="$DEST" MELLUM_PIN_PREFIX=mellum21 MELLUM_PIN_TESTDATA="$GOLD" MELLUM_PIN_MODEL_ID="$REPO" MELLUM_PIN_REVISION="$REV" MELLUM_PIN_NOTHINK=1 "$PY" -I "$SRC/scripts/pin_mellum2.py"; }

step pull pull || fatal "the pull failed: $(tail -3 "$OUT/pull.log")"
step verify verify || { echo "!! verification failed: the pin step is not run" | tee -a "$SUM"; tail -8 "$OUT/verify.log" | tee -a "$SUM"; exit 1; }
tail -3 "$OUT/verify.log" | tee -a "$SUM"
step pin pin || { echo "!! the pin failed" | tee -a "$SUM"; tail -5 "$OUT/pin.log" | tee -a "$SUM"; exit 1; }
for f in mellum21_forward_golden.json mellum21_window_golden.json; do [ -s "$GOLD/$f" ] || { echo "!! $GOLD/$f is missing or empty" | tee -a "$SUM"; exit 1; }; done
"$PY" -I - "$GOLD" <<'PY' | tee -a "$SUM"
import json, sys
for k in ("forward", "window"):
    g = json.load(open(f"{sys.argv[1]}/mellum21_{k}_golden.json"))
    print(k, "golden:", len(g["ids"]), "ids; argmax", g["argmax"], repr(g["argmax_token"]), "; provenance revision", g["provenance"]["revision"][:12], "transformers", g["provenance"]["transformers"], "torch", g["provenance"]["torch"])
PY
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0
