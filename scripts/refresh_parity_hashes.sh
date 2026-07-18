#!/usr/bin/env bash
#
# refresh_parity_hashes.sh — the sound, goldens-gated deps_hash refresh.
#
# Mechanizes the ONE legitimate use of the "a deps_hash refresh is not a re-validation"
# exception (docs/parity-coverage-policy.md): when a change to a `core` forward file is
# provably NON-numeric (a guarded diagnostic seam, a comment, a rename), the staleness gate
# (TestParityManifest_fresh) false-positives on the whole-file content hash and re-stales
# every family that `uses: core`. The correct fix is to refresh deps_hash while preserving
# validated_at — but that is ALSO the shape of the forbidden gate-silencing move, so it must
# never be done on faith.
#
# This script makes the exception AUDITABLE and ABUSE-RESISTANT:
#   1. It runs the forward GOLDENS — the independent numeric ground truth. If any that RAN
#      failed, the change is numeric, not a refresh: it REFUSES (exit 1).
#   2. If ZERO goldens ran (all skipped for want of fixtures), "green" is vacuous: it REFUSES.
#      Regenerate the tiny checkpoints (scripts/pin_*.py) so the proof is real.
#   3. Only then does it refresh deps_hash (deps_hash ONLY — validated_at/metrics/dates
#      untouched) and print a PROOF block to paste into the commit.
#
# It does NOT touch the hash mechanism — the whole-file hash stays conservative on purpose
# (it never misses a numeric change). It only removes the manual toil, gated on real proof.
#
# IMPORTANT — coverage is the developer's call. The goldens that RUN prove only the core
# paths they exercise. The script prints which families ran vs skipped; if your core change
# touches a path only a skipped family's golden would exercise (e.g. an MLA/SSM-only helper),
# regenerate that family's fixture and re-run before trusting the refresh.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# The forward-numeric goldens: they compare a forward pass to a committed golden, so a
# numeric change to the forward breaks them. (Deliberately NOT the spec-decode / session /
# KV / vision parity tests — those gate features, not the forward numerics.)
GOLDEN_RE='(_forwardParity|_logitParity|_textParity)$'

echo "==> Running forward goldens (the numeric proof a core refresh is non-numeric)…"
set +e
out=$(go test ./decoder/ -run "$GOLDEN_RE" -v 2>&1)
set -e

pass=$(printf '%s\n' "$out" | grep -c '^--- PASS:' || true)
fail=$(printf '%s\n' "$out" | grep -c '^--- FAIL:' || true)
skip=$(printf '%s\n' "$out" | grep -c '^--- SKIP:' || true)

echo "    forward goldens: ${pass} passed, ${fail} failed, ${skip} skipped"

if [ "$fail" -gt 0 ]; then
	echo
	echo "REFUSING to refresh: ${fail} forward golden(s) FAILED — the core change is NUMERIC."
	echo "This is not a hash refresh: re-run the T3 parity gate and bump validated_at, or fix the regression."
	printf '%s\n' "$out" | grep '^--- FAIL:' | sed 's/^/    /'
	exit 1
fi

if [ "$pass" -eq 0 ]; then
	echo
	echo "REFUSING to refresh: no forward golden RAN (all ${skip} skipped) — 'goldens green' would be vacuous."
	echo "Regenerate the tiny checkpoints so the numeric proof is real, e.g.:"
	echo "    ls scripts/pin_*.py   # then run the ones for the families you changed"
	exit 1
fi

echo
echo "    ran green:"
printf '%s\n' "$out" | grep '^--- PASS:' | sed 's/^--- PASS: /      /;s/ (.*)//'
if [ "$skip" -gt 0 ]; then
	echo "    skipped (no local fixture — NOT covered by this proof; regenerate if your change touches them):"
	printf '%s\n' "$out" | grep '^--- SKIP:' | sed 's/^--- SKIP: /      /;s/ (.*)//'
fi

echo
echo "==> Refreshing deps_hash (deps_hash ONLY; validated_at / metrics / dates preserved)…"
go test ./decoder/ -run TestParityManifest -update >/dev/null

# Confirm the gate is now green and confirm we changed ONLY deps_hash lines.
go test ./decoder/ -run TestParityManifest_fresh >/dev/null
nonhash=$(git diff -- testdata/parity_manifest.json | grep -E '^[+-]' | grep -vE 'deps_hash|^(\+\+\+|---)' || true)
if [ -n "$nonhash" ]; then
	echo
	echo "ABORT: the update changed more than deps_hash — refusing to leave a mixed edit:"
	printf '%s\n' "$nonhash" | sed 's/^/    /'
	echo "Revert (git checkout testdata/parity_manifest.json) and investigate."
	exit 1
fi
changed=$(git diff --numstat -- testdata/parity_manifest.json | awk '{print $1}')
echo "    staleness gate green; ${changed:-0} deps_hash line(s) refreshed, nothing else."

head=$(git rev-parse --short HEAD)
echo
echo "==> PROOF (paste into the commit body — makes the exception auditable):"
echo "    non-numeric core refresh; validated_at preserved."
echo "    forward goldens green at ${head}: ${pass} passed / ${skip} skipped / 0 failed."
