#!/usr/bin/env bash
# heavy_gate.sh — run the pure-Go HEAVY tier: the real-checkpoint tests that `go test ./...` never
# executes, so no CI job runs them and the only thing standing between "green suite" and a real-model
# regression is someone remembering to run them by hand. This makes that one command with an honest
# verdict.
#
# The tier is gated in TWO layers, both handled here:
#   1. GOINFER_HEAVY_TESTS=1 (requireHeavyModel) — ~108 files. Compile always, but t.Skip unless the
#      env is set. Visible in `go test ./...` output as SKIP.
#   2. //go:build realckpt — 25 files (deepseek, nemotron, llama4, qwen35, gemma4-26b, granite,
#      cohere-aya, gptoss, phi3, mistral, gemma3, qwen2/3, …). These do NOT COMPILE without the tag,
#      so plain `go test` (and CI) never even builds them — the strongest "run by nothing". This gate
#      passes `-tags realckpt` so they compile, then GOINFER_HEAVY_TESTS=1 so the env-gated ones run.
#
# It is the CPU/pure-Go sibling of gpu_gate.sh (which covers the backend-specific GPU checks and
# is left untouched here — a separate pending edit ports its INCONCLUSIVE handling). Run this on
# a box that has the real checkpoints (the RTX box's ~/models zoo); a laptop with a partial zoo
# will run the tests it has models for and REPORT the rest as skipped, never as passed.
#
# HONESTY RULES (from gpu_gate.sh, learned the hard way):
#   - A skip is NOT a pass. With GOINFER_HEAVY_TESTS=1 set, a heavy test STILL skips if its
#     specific checkpoint is absent — so green here means "ran and passed", and every skip is
#     listed with its reason so a missing model can't masquerade as coverage.
#   - Zero tests run is a FAIL, not a pass. A compile break or a -run that matches nothing exits
#     0 and prints "ok"; this refuses to call that green.
#   - Sequential (-p 1). Heavy tests each load a multi-GB model; parallel packages contend for RAM
#     and fail as spurious numerics or an OOM kill rather than a clean result.
#
# Usage:
#   scripts/heavy_gate.sh                       # full pure-Go heavy tier
#   GOINFER_HEAVY_RUN='TestGemma4' scripts/heavy_gate.sh   # smoke a subset (a -run filter)
# Env:
#   GOINFER_GATE_MODELS   dir holding the real checkpoints (default: $HOME/models)
#   GOINFER_HEAVY_RUN     optional `go test -run` regex to narrow the run (default: all)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
DIRTY=""; [ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" +dirty"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODELS="${GOINFER_GATE_MODELS:-$HOME/models}"
RUN="${GOINFER_HEAVY_RUN:-.}"
if [ -n "${GOINFER_HEAVY_PKGS:-}" ]; then read -ra PKGS <<<"$GOINFER_HEAVY_PKGS"; else PKGS=(./decoder/ ./internal/serveapp/); fi

PASSED=0 SKIPPED=0 FAILED=0 RAN=0
SKIPS=()

printf '\033[1m== heavy_gate provenance ==\033[0m\n'
echo "  repo        $COMMIT$DIRTY"
echo "  date (UTC)  $DATE"
echo "  host        $(uname -s) $(uname -m)"
echo "  models      $MODELS"
echo "  run filter  $RUN"
echo "  packages    ${PKGS[*]}"
[ -n "$DIRTY" ] && echo "  NOTE: working tree DIRTY — this verdict does not describe a committed state."
if [ ! -d "$MODELS" ]; then
  echo "  models dir missing — nothing to run against; refusing to report a verdict."
  exit 2
fi

export GOINFER_HEAVY_TESTS=1
export GOINFER_MODELS_DIR="$MODELS"

for pkg in "${PKGS[@]}"; do
  printf '\n\033[1m== %s ==\033[0m\n' "$pkg"
  # -v is required: without it `go test` prints no --- SKIP lines, so a skip would be invisible.
  # tee to a per-package log (not command substitution) so the raw output — including a panic or a
  # crash that never prints a "--- FAIL:" line — survives for inspection; PIPESTATUS keeps go's rc.
  pkglog="/tmp/heavy_gate_$(printf '%s' "$pkg" | tr '/.' '__').log"
  go test "$pkg" -tags 'goinfer_testhooks realckpt' -run "$RUN" -v -count=1 -p 1 -timeout 40m >"$pkglog" 2>&1
  rc=${PIPESTATUS[0]}
  out="$(cat "$pkglog")"
  # top-level results only (subtest lines are indented; anchor at column 0)
  p=$(printf '%s\n' "$out" | grep -cE '^--- PASS: ' || true)
  s=$(printf '%s\n' "$out" | grep -cE '^--- SKIP: ' || true)
  f=$(printf '%s\n' "$out" | grep -cE '^--- FAIL: ' || true)
  ran=$(printf '%s\n' "$out" | grep -cE '^=== RUN ' || true)
  PASSED=$((PASSED + p)); SKIPPED=$((SKIPPED + s)); FAILED=$((FAILED + f)); RAN=$((RAN + ran))
  printf '  \033[32mPASS %d\033[0m  \033[33mSKIP %d\033[0m  \033[31mFAIL %d\033[0m  (RUN lines: %d, go rc: %d)\n' "$p" "$s" "$f" "$ran" "$rc"
  # record skip reasons so a missing checkpoint is visible, never silent coverage
  while IFS= read -r line; do [ -n "$line" ] && SKIPS+=("$line"); done < <(printf '%s\n' "$out" | grep -E '^ *--- SKIP: ' | sed 's/^ *--- SKIP: /    /')
  if [ "$f" -gt 0 ]; then
    printf '%s\n' "$out" | grep -E '^--- FAIL: |^\s+.*\.go:[0-9]+:' | head -30 | sed 's/^/    /'
  fi
  # rc-aware honesty: a non-zero exit is a FAILURE even when no "--- FAIL:" line was counted — a panic
  # in a goroutine, a fatal error, a timeout, or a 0-match crashes/aborts the binary without the
  # per-test FAIL line. Counting only --- FAIL lines would report GREEN on a crashed package (the exact
  # "a green that means nothing" this gate exists to prevent). Count it, and surface the cause.
  if [ "$rc" -ne 0 ] && [ "$f" -eq 0 ]; then
    FAILED=$((FAILED + 1))
    echo "    go test exited $rc with 0 --- FAIL lines — a HIDDEN failure (panic / fatal / timeout / 0-match). cause:"
    grep -nE 'panic:|fatal error|SIGSEGV|SIGABRT|test timed out|^FAIL\b|goroutine [0-9]+ \[running\]|no matches for pattern|build failed|\.go:[0-9]+: ' "$pkglog" | head -20 | sed 's/^/      /'
  fi
  echo "    raw log: $pkglog"
done

printf '\n\033[1m== verdict ==\033[0m\n'
echo "  PASSED $PASSED   SKIPPED $SKIPPED   FAILED $FAILED"
if [ "${#SKIPS[@]}" -gt 0 ]; then
  echo "  skipped (missing checkpoint or unmet gate — NOT coverage):"
  printf '%s\n' "${SKIPS[@]}"
fi
if [ "$FAILED" -gt 0 ]; then echo -e "  \033[31mRED — $FAILED failed\033[0m"; exit 1; fi
if [ "$PASSED" -eq 0 ]; then echo -e "  \033[31mRED — nothing actually ran (all skipped / no match); a green with 0 tests is not green\033[0m"; exit 1; fi
echo -e "  \033[32mGREEN — $PASSED heavy-tier tests ran and passed ($SKIPPED skipped for missing models)\033[0m"
