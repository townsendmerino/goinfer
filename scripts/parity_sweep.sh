#!/usr/bin/env bash
#
# parity_sweep.sh — the release-tag ritual: one full, asset-gated parity sweep on
# the EXACT commit you intend to tag, so every family/quant/tokenizer gate is
# green on the release SHA (not just per-commit along the way).
#
# Run it on the box that has the full asset set (the model zoo + the goldens +,
# for qwen3_5_moe, the 35B GGUF and torch). A required gate that SKIPS (asset
# missing) or FAILS is a RELEASE BLOCKER — the script exits non-zero and prints
# which family is not green.
#
# The skip-if-absent gates look for checkpoints at their default paths
# (testdata/<...>, ~/models/<...>) or via env vars; put the assets in place /
# export the vars before running. Known env vars (see the test files):
#   GINFER_PREQUANT_GGUF      — TestDecodeParity / serialize round-trip / quant gates
#   GOINFER_QWEN35_GGUF       — qwen3_5_moe real-checkpoint gates (-tags realckpt)
#   GOINFER_QWEN35_GOLDEN     — its Gate-2 golden manifest dir
#   GOINFER_SERVE_MODEL[2]    — cmd/serve soak/chaos (optional; not a family gate)
#
# Usage:  scripts/parity_sweep.sh            # full sweep, fail on any non-green gate
#         REALCKPT=0 scripts/parity_sweep.sh # skip the 35B realckpt gates

set -uo pipefail
cd "$(dirname "$0")/.."

SHA="$(git rev-parse --short HEAD)"
TAGREF="$(git describe --tags --always 2>/dev/null || echo "$SHA")"
TIMEOUT="${TIMEOUT:-120m}"
REALCKPT="${REALCKPT:-1}"
LOG="$(mktemp -t parity_sweep.XXXXXX)"

echo "== parity sweep on ${SHA} (${TAGREF}) =="
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: working tree is dirty — tag on a clean checkout, not this."
fi

# Required gates: "family-label|TestName". Each MUST report PASS. One or two
# canonical gates per family + the quant-format and tokenizer parity gates.
GATES=(
  "gemma3            |TestGGUF_gemma3_parity"
  "gemma3-forward    |TestForward_logitParity"
  "gemma3-sliding    |TestForward_slidingWindowParity"
  "gemma4-E2B/E4B    |TestGemma4_logitParity"
  "gemma4-12B        |TestGemma4_12B_logitParity"
  "qwen2             |TestQwen2_forwardParity"
  "qwen2-gguf        |TestGGUF_qwen2_parity"
  "qwen3             |TestQwen3_forwardParity"
  "qwen3-gguf        |TestGGUF_qwen3_parity"
  "qwen2moe          |TestQwen2Moe_forwardParity"
  "llama             |TestLlama_forwardParity"
  "llama3.2          |TestLlama32_forwardParity"
  "mistral           |TestMistral_forwardParity"
  "mixtral           |TestMixtral_forwardParity"
  "mellum2           |TestMellum2_logitParity"
  "mellum2-window    |TestMellum2_windowParity"
  "gpt2              |TestGPT2_forwardParity"
  "qwen3_5_moe-tiny  |TestQwen35_forwardParity"
  "deltanet          |TestGatedDeltaNet_parity"
  "deepseek-tiny     |TestDeepseek_textParity"
  "kimi-tiny         |TestKimi_textParity"
  "phi3-tiny         |TestPhi3_textParity"
  "llama4-tiny       |TestLlama4_textParity"
  "gguf-Q8_0         |TestGGUF_Q8_0_parity"
  "gguf-Q4_0         |TestGGUF_Q4_0_parity"
  "gguf-Q4_K_M       |TestGGUF_Q4_K_M_parity"
  "gguf-Q4_K_S       |TestGGUF_Q4_K_S_parity"
  "gguf-Q5_K_M       |TestGGUF_Q5_K_M_parity"
  "gguf-Q6_K         |TestGGUF_Q6_K_parity"
  "gguf-Q3_K_M       |TestGGUF_Q3_K_M_parity"
  "gguf-Q2_K         |TestGGUF_Q2_K_parity"
  "gptq              |TestGPTQ_parity"
  "awq               |TestAWQ_parity"
  "w4a8-int4         |TestW4A8DecodeParity"
  "tok-gemma         |TestEncodeDecode_goldenParity"
  "tok-qwen3         |TestByteLevel_qwen3GoldenParity"
  "tok-llama3        |TestByteLevel_llama3GoldenParity"
  "tok-mellum2       |TestByteLevel_mellum2GoldenParity"
  "tok-tinyllama     |TestLoadGGUF_tinyllamaParity"
)
# Real-checkpoint gates (build-tagged realckpt). Heavy (large downloads / RAM); each
# SKIPs when its asset is absent (see the test files for the default paths / env vars).
REALCKPT_GATES=(
  "qwen3.6-gguf-gate |TestQwen35GGUF_gate"
  "qwen3.6-weightdiff|TestQwen35GGUF_weightDiff"
  "qwen3.6-gate2     |TestQwen35Real_gate2FullModel"
  "deepseek-v2lite   |TestDeepseekV2LiteReal_gate"
  "deepseek-moonlight|TestDeepseekMoonlightReal_gate"
  "deepseek-gguf     |TestDeepseekGGUFReal_gate"
  "phi3-mini-oracle  |TestPhi3MiniReal_gate"
  "phi3-gguf         |TestPhi3GGUFReal_gate"
  "llama4-scout-gguf |TestLlama4Real_gate"
)

echo "-- running ./decoder/ ./tokenizer/ (verbose) ..."
go test -v -timeout "$TIMEOUT" ./decoder/ ./tokenizer/ >>"$LOG" 2>&1
if [ "$REALCKPT" = "1" ]; then
  echo "-- running -tags realckpt real-model gates ..."
  go test -tags realckpt -v -timeout "$TIMEOUT" ./decoder/ -run 'Qwen35|Real_gate' >>"$LOG" 2>&1
fi

# classify(name) → PASS | FAIL | SKIP | MISSING (the test never ran)
classify() {
  local line
  line="$(grep -E "^--- (PASS|FAIL|SKIP): $1 \(" "$LOG" | tail -1)"
  case "$line" in
    *"--- PASS"*) echo PASS ;;
    *"--- FAIL"*) echo FAIL ;;
    *"--- SKIP"*) echo SKIP ;;
    *)            echo MISSING ;;
  esac
}

blockers=0
printf '\n%-20s %-34s %s\n' "FAMILY" "GATE" "RESULT"
printf '%s\n' "------------------------------------------------------------------------"
check_all() {
  for g in "$@"; do
    local fam="${g%%|*}"; local name="${g##*|}"
    fam="$(echo "$fam" | xargs)" # trim
    local r; r="$(classify "$name")"
    local mark
    case "$r" in
      PASS)    mark="✅ pass" ;;
      FAIL)    mark="❌ FAIL (blocker)"; blockers=$((blockers+1)) ;;
      SKIP)    mark="⚠️  SKIP — asset missing (blocker)"; blockers=$((blockers+1)) ;;
      MISSING) mark="⛔ DID NOT RUN (blocker)"; blockers=$((blockers+1)) ;;
    esac
    printf '%-20s %-34s %s\n' "$fam" "$name" "$mark"
  done
}
check_all "${GATES[@]}"
[ "$REALCKPT" = "1" ] && check_all "${REALCKPT_GATES[@]}"

# Safety net: any OTHER parity/gate-shaped test that skipped (a family we forgot).
echo
echo "-- catch-all: other *Parity / *gate tests that SKIPPED (review) --"
grep -E '^--- SKIP: Test' "$LOG" \
  | grep -iE 'parity|gate|golden' \
  | grep -vE "$(printf '%s|' "${GATES[@]##*|}" "${REALCKPT_GATES[@]##*|}" | sed 's/|$//')" \
  | sed -E 's/\(.*//; s/^--- SKIP: /   skipped: /' | sort -u || true

echo
echo "== ${SHA}: $([ "$blockers" -eq 0 ] && echo 'ALL REQUIRED GATES GREEN ✅' || echo "${blockers} BLOCKER(S) ❌") =="
echo "full log: $LOG"
exit "$([ "$blockers" -eq 0 ] && echo 0 || echo 1)"
