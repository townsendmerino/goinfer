#!/bin/bash
# G-I8-6 on the Mac (docs/tasks/task-integration-harness-2026-10.md, "I8", registered 2026-10-13): is the request the
# 7B refused on the night of 2026-10-12 answered once Metal prefills in blocks and admission prices a block? Serve
# Qwen2.5-7B under --backend metal with -ctx 24576, send one request of Claude Code's first-request size (the client
# sizes it at 20,199 tokens or more; it was 20,954 last night), read swap before the load and after the answer.
# Two arms, the same -ctx: the build with I8, and main before it as the do-nothing arm (expected to refuse with 413).
# Runs with nobody watching: both binaries and the client are pinned copies in $B; night.py holds the timing lock.
#
#   B=~/goinfer-bench/i8-2026-10-13 bash $B/run-g-i8-night.sh
set -u
B=${B:-$HOME/goinfer-bench/i8-2026-10-13}
OUT=${OUT:-$B/out}
NEW=$B/bin/serve-metal-i8
OLD=$B/bin/serve-metal-main
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
mkdir -p "$OUT" || exit 2
for f in "$NEW" "$OLD" "$B/m2_served_check.py" "$M7"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new: $(cat "$NEW.rev" 2>/dev/null)"; echo "old: $(cat "$OLD.rev" 2>/dev/null)"
  echo "load: $(uptime)"; echo "swap: $(sysctl -n vm.swapusage)"; vm_stat | sed -n 1,8p
} > "$OUT/provenance.txt"
for arm in new old; do
  bin=$NEW; [ "$arm" = old ] && bin=$OLD
  echo "== arm $arm: $(date '+%T')"
  python3 -u "$B/m2_served_check.py" "$bin" "$M7" "$OUT/served-$arm" --tokens 20199 --extra-arg=-ctx --extra-arg=24576 2>&1 | tee "$OUT/served-$arm.log"
done
# The verdict reads the two logs: the new arm answered with swap growth of 64 MB or under; the old arm is reported.
python3 - "$OUT" <<'PY' | tee "$OUT/g-i8-6.verdict.txt"
import re, sys
out = sys.argv[1]
def read(arm):
    t = open(f"{out}/served-{arm}.log").read()
    ans = re.search(r"answered in ([\d.]+) s: prompt_tokens (\d+)", t)
    ref = re.search(r"REFUSED \((\d+)\)", t)
    g = re.search(r"growth ([+-]?\d+) MB", t)
    ctx = re.search(r"context: (\d+) tokens", t)
    return ans, ref, (int(g.group(1)) if g else None), (int(ctx.group(1)) if ctx else None), "did not come up" in t
na, nr, ng, nctx, ndown = read("new")
oa, orf, og, octx, odown = read("old")
def say(a, r, g, ctx, down):
    if down: return "serve did not come up"
    s = f"context {ctx}; "
    s += f"answered {a.group(2)} tokens in {a.group(1)} s" if a else (f"refused ({r.group(1)})" if r else "no answer line")
    return s + f"; swap growth {g:+d} MB" if g is not None else s
print("new arm:", say(na, nr, ng, nctx, ndown))
print("old arm:", say(oa, orf, og, octx, odown))
if ndown or nctx != 24576:
    print("G-I8-6: VOID (the new arm did not load at -ctx 24576)")
    sys.exit(1)
ok = bool(na) and ng is not None and ng <= 64
print("G-I8-6:", "PASS" if ok else "FAIL", "(answered, swap growth 64 MB or under)" if ok else "")
if ok and oa:
    print("NOTE: the old arm answered too, so tonight's memory, not the change, may be what admitted the request")
sys.exit(0 if ok else 1)
PY
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
