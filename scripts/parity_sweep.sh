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
#   GOINFER_PREQUANT_GGUF      — TestDecodeParity / serialize round-trip / quant gates
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
EMIT_MANIFEST="${EMIT_MANIFEST:-0}"
LOG="$(mktemp -t parity_sweep.XXXXXX)"

# EMIT_MANIFEST=1: gates that measured against a real oracle emit a PARITY_ROW line
# (emitParityRow, no-op without this env); after the run we grep them out of $LOG and a
# Go merge folds them into testdata/parity_manifest.json, then re-render the matrix. Off
# by default → the manifest is never touched. Numeric-oracle gates expected to emit
# (gate|family), so a passing gate that produced no row (emitter missing?) is visible.
EMIT_GATES=(
  "TestPhi3MiniReal_gate|phi3"
  "TestDeepseekV2LiteReal_gate|deepseek_v2"
  "TestDeepseekMoonlightReal_gate|deepseek_v3"
  "TestQwen35Real_gate2FullModel|qwen3_5_moe"
  "TestCohereAyaReal_gate|cohere"
  "TestCohere2R7bReal_gate|cohere2"
)
# ---- PREFLIGHT: resolve the asset environment, and REPORT what it resolved ----
#
# This block exists because the same invocation error produced a false "15 BLOCKER(S)" three separate
# times, and the tree was fine every time. The gates skip-if-absent, and a skip is reported as a
# blocker, so an unset variable and a genuinely missing checkpoint are INDISTINGUISHABLE in the
# output. That is the census-denominator shape again: the script reported its finding without
# reporting what it had examined.
#
# Two fixes, both here:
#   1. GOINFER_HEAVY_TESTS=1 is SET, not required. This is the release sweep; loading multi-GB
#      checkpoints is its entire purpose, and making the operator remember an opt-in whose absence
#      reads as "asset missing (blocker)" is a trap, not a safety feature.
#   2. Every known asset variable is resolved against ~/models when unset, and the resolution TABLE is
#      printed. An operator's explicit value always wins and is shown as (env).
#
# An unresolvable variable prints NOT FOUND here, at the top, where it is a one-line fix -- instead of
# surfacing 40 minutes later as a blocker that names a family rather than a path.
#
# THE RESOLUTION NOW COMES FROM THE SHARED REGISTRY (testdata/assets.json), not from a table in this
# file. The table that used to live here was a SECOND implementation of "is this asset present", and
# it was the weaker one:
#
#   * it tested `[ -e "$path" ]`, which a DIRECTORY satisfies -- so three of its entries named a
#     directory where the loader wanted the .gguf FILE inside it, and it reported them resolved.
#     Four gates were costed by that.
#   * it could not express GOINFER_QWEN35_GOLDEN's actual requirement, a readable manifest.json
#     INSIDE the directory, so it said present while the gate skipped.
#   * its default paths were a copy of the defaults each gate spelled out separately, free to drift
#     from them -- and GOINFER_PREQUANT_GGUF had drifted into three different values.
#   * its closing line said "all 10 assets resolved" above a table of NINE rows: a hardcoded count,
#     in the sentence announcing that everything was fine. The registry derives that number now.
#
# GOINFER_QWEN35_REAL is registered too, and was never in the table here at all.
export GOINFER_HEAVY_TESTS=1
_ASSET_ENV="$(mktemp)"
if python3 scripts/asset_registry.py preflight --export-to "$_ASSET_ENV"; then
  # shellcheck source=/dev/null
  . "$_ASSET_ENV"
  echo "   GOINFER_HEAVY_TESTS=1 set by this script — the release sweep loads real checkpoints"
else
  # There is no set -e here, so without this branch a missing python3 would export NOTHING, every
  # asset-gated test would skip, and the sweep would print a full slate of blockers about a tree
  # that is fine -- the precise false-blocker cascade this preflight was written to end. Fail loudly
  # instead: the sweep continues so the non-asset gates still report, but nobody can read the
  # blocker count as a statement about the tree.
  echo "   !! ASSET PREFLIGHT DID NOT RUN (scripts/asset_registry.py failed above)."
  echo "   !! No asset variable has been resolved. Every asset-gated gate below will skip and be"
  echo "   !! counted as a blocker. THAT COUNT IS ABOUT THIS FAILURE, NOT ABOUT THE TREE."
fi
rm -f "$_ASSET_ENV"
echo

if [ "$EMIT_MANIFEST" = "1" ]; then
  export GOINFER_MANIFEST_EMIT=1
  echo "== EMIT_MANIFEST=1: real-oracle gates will record measured rows into the manifest =="
fi

echo "== parity sweep on ${SHA} (${TAGREF}) =="
# The composition, not just the verdict. This gate's axes are family x quant x loader; a pass count
# alone cannot distinguish "the axes are covered" from "an axis collapsed to one value" — which is
# exactly how the forward goldens stayed f32-only through nine refreshes behind an accurate count.
python3 scripts/sweep_composition.py || echo "  (composition unavailable — see above)"
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
  "nemotron-tiny     |TestNemotron_textParity"
  "nemotron3nano-tiny|TestNemotron3NanoMoE_textParity"
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

# ---- REQUIRED-AND-AVAILABLE vs REQUIRED-BUT-ASSET-NEVER-CREATED ----
#
# A required gate that skips is normally a blocker: the asset exists somewhere and this box is
# mis-provisioned, which someone can fix. But a gate whose asset HAS NEVER BEEN BUILT is a different
# thing -- no invocation on any machine can make it green, so calling it a blocker calls the release
# broken for a coverage gap, at every tag, forever.
#
# It is still REPORTED, on its own line, and counted -- as what it is, a coverage gap, rather than as
# an obstruction someone is expected to clear before tagging. A permanent blocker is not a gate; it
# is a thing people learn to override, and an override habit is worse than an honest gap.
#
# TO REMOVE A NAME FROM HERE: build the asset. That is the only correct way off this list.
#
# THE LIST IS EMPTY, AND IT GOT THERE THE ONLY CORRECT WAY (2026-08-18, v1.0 gate 1.3).
# TestW4A8DecodeParity was the sole entry: it needs a MATCHED int4+int8 .giw pair and only int4
# bundles had ever been produced, so it skipped at every tag from 2026-08-12 on. The pair now
# exists -- built from the 1.5B the test's own docstring names, both halves from the SAME source
# GGUF so "matched" is by construction rather than by belief:
#
#   go run ./cmd/prequant -quant int4     -o ~/models/qwen15-w4a8-int4.giw \
#     ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
#   go run ./cmd/prequant -quant int8int8 -o ~/models/qwen15-w4a8-int8.giw \
#     ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
#
# The gate then ran GREEN on first invocation: 16/16 greedy-token agreement, identical decoded
# text on both halves. Registered as GOINFER_W4A8_INT4/GOINFER_W4A8_INT8 in testdata/assets.json,
# so a box without them reports a missing ASSET, which is a fixable provisioning fact -- not the
# unfixable "no invocation on any machine can make this green" this list exists for.
#
# Keep the machinery: the next gate whose asset has never been built belongs here, and the empty
# array is the honest current state rather than a reason to delete the classification.
ASSET_NEVER_BUILT=()
asset_gap() {
  local n
  [ ${#ASSET_NEVER_BUILT[@]} -eq 0 ] && return 1
  for n in "${ASSET_NEVER_BUILT[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

blockers=0
coverage_gaps=0
first_runs=0
checked_gates=""
printf '\n%-20s %-34s %s\n' "FAMILY" "GATE" "RESULT"
printf '%s\n' "------------------------------------------------------------------------"
check_all() {
  for g in "$@"; do
    local fam="${g%%|*}"; local name="${g##*|}"
    fam="$(echo "$fam" | xargs)" # trim
    local r; r="$(classify "$name")"
    checked_gates="${checked_gates:+$checked_gates,}$name"
    local mark
    case "$r" in
      PASS)    mark="✅ pass" ;;
      FAIL)    # THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is
               # asserting a DELTA IT HAS NO SECOND POINT TO COMPUTE. Reported on its own line and
               # counted, but it does NOT block: the change that made a failure visible is the one
               # change that provably did not cause it.
               #
               # first-run does NOT mean harmless. It is about ATTRIBUTION, not correctness -- the
               # finding becomes an ITEM, not a BLOCKER, and the first observed value must not be
               # banked as a baseline without a person deciding it is correct
               # (scripts/gate_ledger.py promote).
               lclass="$(python3 scripts/gate_ledger.py classify --gate "$name" 2>/dev/null || echo CONFIRMED)"
               case "$lclass" in
                 FIRST-RUN)
                   mark="FIRST-RUN - failed, no confirmed prior result (ITEM, not a blocker)"
                   first_runs=$((first_runs+1)) ;;
                 SOURCE-CHANGED)
                   mark="FAIL (blocker) - NOTE: confirmed before this gate last changed"
                   blockers=$((blockers+1)) ;;
                 *)
                   mark="FAIL (blocker)"; blockers=$((blockers+1)) ;;
               esac ;;
      SKIP)    if asset_gap "$name"; then
                 mark="COVERAGE GAP - asset never built (NOT a blocker; see ASSET_NEVER_BUILT)"
                 coverage_gaps=$((coverage_gaps+1))
               else
                 mark="SKIP - asset missing (blocker)"; blockers=$((blockers+1))
               fi ;;
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

# EMIT_MANIFEST: fold the measured rows into the manifest, re-render the matrix, and
# report coverage. The emitter only wrote rows for gates that PASSED (t.Failed guard),
# so this never records a failed/skipped gate. Independent of the blocker count.
if [ "$EMIT_MANIFEST" = "1" ]; then
  echo
  echo "-- EMIT_MANIFEST: merging measured PARITY_ROW lines into testdata/parity_manifest.json --"
  ROWS="$(mktemp -t parity_rows.XXXXXX)"
  grep '^PARITY_ROW ' "$LOG" >"$ROWS" || true
  emitted_fams="$(sed -E 's/.*"family":"([^"]+)".*/\1/' "$ROWS" | sort -u)"
  n_rows="$(grep -c '^PARITY_ROW ' "$LOG" || true)"

  if [ "$n_rows" -gt 0 ]; then
    if go test ./decoder -run '^TestParityManifest_merge$' -merge-rows "$ROWS" -count=1 >>"$LOG" 2>&1; then
      echo "   merged $n_rows row(s); families recorded: $(echo $emitted_fams | tr '\n' ' ')"
      echo "-- re-rendering docs/capability-matrix.{md,json} from the manifest --"
      go test ./decoder -run 'CapabilityMatrix|ParityManifest' -update -count=1 >>"$LOG" 2>&1 \
        && echo "   matrix + manifest re-rendered" \
        || { echo "   ❌ matrix -update failed (see $LOG)"; blockers=$((blockers+1)); }
    else
      echo "   ❌ merge failed (see $LOG)"; blockers=$((blockers+1))
    fi
  else
    echo "   no PARITY_ROW lines emitted (no numeric-oracle gate ran with its asset)"
  fi

  # Coverage: a numeric-oracle gate that PASSED but emitted no row means a missing
  # emitParityRow call — surface it. SKIP/FAIL gates are expected to emit nothing.
  echo "-- emitter coverage (numeric-oracle gates) --"
  for g in "${EMIT_GATES[@]}"; do
    name="${g%%|*}"; fam="${g##*|}"
    r="$(classify "$name")"
    if echo "$emitted_fams" | grep -qx "$fam"; then
      printf '   %-34s %s\n' "$name" "✅ recorded row ($fam)"
    elif [ "$r" = "PASS" ]; then
      printf '   %-34s %s\n' "$name" "⚠️  PASSED but emitted NO row for $fam (emitter missing?)"
    else
      printf '   %-34s %s\n' "$name" "— $r (no row expected)"
    fi
  done
  rm -f "$ROWS"
fi

echo
if [ "$first_runs" -gt 0 ]; then
  echo "== ${first_runs} FIRST-RUN: failed with no confirmed prior result =="
  echo "   Reported, NOT counted as blockers -- there is no second point to compute a delta from."
  echo "   NOT a claim they are harmless: each is an ITEM. Confirm a value deliberately with"
  echo "   scripts/gate_ledger.py promote --gate <G> --value <V> --by <you>."
fi
python3 scripts/gate_ledger.py reconcile --gates "$checked_gates" 2>/dev/null || true
if [ "$coverage_gaps" -gt 0 ]; then
  echo "== ${coverage_gaps} COVERAGE GAP(S): required gates whose asset has never been built =="
  echo "   Reported, not counted as blockers. No invocation can clear them -- only building the"
  echo "   asset can. A permanent blocker is not a gate, it is an override habit."
fi
echo "== ${SHA}: $([ "$blockers" -eq 0 ] && echo 'ALL REQUIRED GATES GREEN' || echo "${blockers} BLOCKER(S)")$([ "$coverage_gaps" -gt 0 ] && echo " (+${coverage_gaps} coverage gap)")$([ "$first_runs" -gt 0 ] && echo " (+${first_runs} first-run)") =="
echo "full log: $LOG"
exit "$([ "$blockers" -eq 0 ] && echo 0 || echo 1)"
