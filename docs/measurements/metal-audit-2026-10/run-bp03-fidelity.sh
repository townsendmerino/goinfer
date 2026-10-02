#!/usr/bin/env bash
# Night job: B-P03's fidelity gate (docs/tasks/task-metal-audit-2026-10.md, "B-P03: fidelity gate at the 1024 floor"),
# pre-registered there on 2026-10-02 before any graded run, under the owner's 2026-09-25 amended bar for
# reordering-only decode-attention kernels (docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md).
#   P1: TestR17KernelAccuracy, arms bp03 (exact vs production's block kernel), set B, 10 prompts, depths 1024/1280/1535,
#       the 1.5B and the 7B from their .int4.metal.giw sidecars.
#   P2: TestR17_decodeFidelityGate, candidate attention_fa (production's block kernel), K = 1024, floor 1024, set B,
#       10 prompts x 64 teacher-forced positions: the 1.5B (graded) with its exact-null control, and the 7B against
#       its D7-K1024 references (reported).
# Deterministic, so one run each. Binary built at REV with -tags goinfer_testhooks:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -tags goinfer_testhooks \
#     -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-bp03 --est 30 --by "Claude, Metal audit B-P03 fidelity" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-bp03-fidelity.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/bp03/; the verdict lines in results.txt.
set -uo pipefail
REV=2c5cfe8f
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
LOG=$HOME/goinfer-logs/metal-audit-2026-10/bp03
G15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.giw
G7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for f in "$G15" "$G7" "$HOME/goinfer-logs/prefill-ref-b/S-K1024-p9.bin" "$HOME/goinfer-logs/prefill-ref-b/D7-K1024-p9.bin"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
} | tee "$LOG/provenance.txt"

run() { # <log name> <test regexp> ENV=VALUE...
  local name=$1 re=$2
  shift 2
  echo "== $name — $(date '+%T')" | tee -a "$LOG/provenance.txt"
  env GOINFER_PREFILL_GATE_PROMPTS=b "$@" "$BIN" -test.v -test.count=1 -test.timeout 60m -test.run "$re" > "$LOG/$name.log" 2>&1
  echo "$name rc=$?" | tee -a "$LOG/provenance.txt"
}

P1="GOINFER_METAL_R17=1 GOINFER_METAL_R17_ACC_ARMS=bp03 GOINFER_METAL_R17_PROMPTS=10 GOINFER_METAL_R17_DEPTHS=1024,1280,1535"
run p1-1.5b '^TestR17KernelAccuracy$' $P1 GOINFER_METAL_R17_MODEL="$G15"
run p1-7b '^TestR17KernelAccuracy$' $P1 GOINFER_METAL_R17_MODEL="$G7"
P2="GOINFER_HEAVY_TESTS=1 GOINFER_METAL_GATE_K=1024 GOINFER_METAL_GATE_FLOOR=1024"
run p2-1.5b '^TestR17_decodeFidelityGate$' $P2 GOINFER_METAL_R17_CAND=attention_fa
run p2-1.5b-null '^TestR17_decodeFidelityGate$' $P2 GOINFER_METAL_R17_CAND=exact-null
run p2-7b '^TestR17_decodeFidelityGate$' $P2 GOINFER_METAL_R17_CAND=attention_fa GOINFER_METAL_GATE_MODEL="$G7" GOINFER_METAL_GATE_CELL=D7

{
  for f in p1-1.5b p1-7b; do echo "-- $f"; grep -E '^=== R17 kernel accuracy|^  (exact \(shipped attention\)|production \(r\.pAttnFA\)) +[0-9]' "$LOG/$f.log" | head -3; done
  for f in p2-1.5b p2-1.5b-null p2-7b; do echo "-- $f"; grep -E 'SUSPECTED|verdict|KL ratio|critA|critB|ceiling' "$LOG/$f.log" | tail -8; done
} | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
