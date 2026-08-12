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

# PRE-FLIGHT: the manifest must be CLEAN before the refresh runs.
#
# This exists because of a real mistake, not a hypothetical one. An aikit bump edited
# aikit_version by hand and then ran this script. The abort below fired correctly — the diff was
# no longer deps_hash-only — but it fired on a file that already held an edit worth keeping, so
# "revert and investigate" was advice the operator did not take, and the mixed edit got committed
# anyway. The guard detected the problem and still did not prevent it.
#
# Requiring a clean start fixes both halves. It catches that case BEFORE the 4-minute goldens run
# rather than after, and it makes the abort's automatic revert safe: with nothing uncommitted at
# entry, `git checkout` can only discard what this script itself wrote.
if ! git diff --quiet -- testdata/parity_manifest.json || ! git diff --cached --quiet -- testdata/parity_manifest.json; then
	echo "ABORT: testdata/parity_manifest.json already has uncommitted changes."
	git diff HEAD -- testdata/parity_manifest.json | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | sed 's/^/    /'
	echo
	echo "Commit or revert them first, THEN re-run. A refresh started from a dirty manifest cannot"
	echo "tell its own output from yours, and this script's proof block would claim the goldens"
	echo "vouched for both."
	exit 1
fi

# The forward-numeric goldens: they compare a forward pass to a committed golden, so a
# numeric change to the forward breaks them. (Deliberately NOT the spec-decode / session /
# KV / vision parity tests — those gate features, not the forward numerics.)
# ^TestGGUF_.*_parity$ is included deliberately, and it cost nothing to add. The cross-gate check
# (scripts/sweep_composition.py) showed parity_sweep.sh covering the GGUF quant formats while these
# goldens did not — and parity_sweep.sh is RELEASE-ONLY, run by hand on the box, so between releases
# a frozen-core edit got no GGUF-quant proof at all. The gates already existed and already passed;
# they were simply outside the selector. 11 gates, 26.8 s measured. Same shape as the int8int8
# goldens that skipped for want of an env var: unplumbed coverage, not missing coverage, and far
# cheaper than authoring the goldens that would have duplicated them.
GOLDEN_RE='(_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$'

# GOINFER_HEAVY_TESTS is set by default here, and that is a coverage decision rather than a
# convenience. Without it the three int8int8 goldens (gemma4, gemma4-12B, mellum2) all skip on
# "heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in", which meant EVERY golden that ran
# was f32 — the refresh proved f32 numerics and silently nothing else, on a runtime whose documented
# default quantization is int4. Two of those three pass here in ~70 s, so the quantized coverage was
# UNPLUMBED rather than missing. Set GOINFER_REFRESH_HEAVY=0 to opt back out on a box without the
# checkpoints; the counts printed below say what actually ran either way.
HEAVY="${GOINFER_REFRESH_HEAVY:-1}"
echo "==> Running forward goldens (the numeric proof a core refresh is non-numeric)…"
if [ "$HEAVY" = "1" ]; then
	echo "    heavy checkpoints ENABLED — the int8 goldens participate (GOINFER_REFRESH_HEAVY=0 to skip)"
fi
set +e
if [ "$HEAVY" = "1" ]; then
	out=$(GOINFER_HEAVY_TESTS=1 go test ./decoder/ -run "$GOLDEN_RE" -v -timeout 60m 2>&1)
else
	out=$(go test ./decoder/ -run "$GOLDEN_RE" -v 2>&1)
fi
set -e

pass=$(printf '%s\n' "$out" | grep -c '^--- PASS:' || true)
fail=$(printf '%s\n' "$out" | grep -c '^--- FAIL:' || true)
skip=$(printf '%s\n' "$out" | grep -c '^--- SKIP:' || true)

echo "    forward goldens: ${pass} passed, ${fail} failed, ${skip} skipped"
# The QUANTIZATION breakdown, not just the count. A run of 19 green f32 goldens and a run of 21 that
# includes two int8 ones are different proofs, and "19 passed" cannot tell them apart — which is how
# the f32-only hole stayed invisible. Q1 in docs/QUEUE.md.
nonf32=$(printf '%s\n' "$out" | grep '^--- PASS:' | grep -cE 'TestGemma4_logitParity|TestGemma4_12B_logitParity|TestMellum2_logitParity|TestGptOssReal_logitParity|TestInt4_forwardParity|^--- PASS: TestGGUF_' || true)
echo "    of those, ${nonf32} drive a QUANTIZED path (int4/int8/int8int8); the rest are f32."
# WHERE this ran is part of what the count means. Fixture checkpoints are gitignored, so a fresh
# `git worktree` has none of them: the same commit that proves 33 goldens in the main checkout proves
# 7 in a worktree, and the difference is the machine, not the change. The count alone cannot say
# which — so say it.
if [ "$skip" -gt "$pass" ]; then
	echo
	echo "    MORE SKIPPED (${skip}) THAN PASSED (${pass}). Almost always a checkout without the"
	echo "    gitignored fixture checkpoints — a fresh git worktree has none of them. This refresh"
	echo "    is REAL but much weaker than the same commit would prove where the fixtures live."
	echo "    Re-run it on a fixture-bearing checkout before treating it as the proof of record."
fi
if [ "$nonf32" -eq 0 ]; then
	echo "    NOTE: this refresh proves f32 numerics ONLY — no quantized golden ran. See Q1."
fi

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
	echo "The file has been REVERTED automatically — nothing is left on disk to commit by accident."
	git checkout -- testdata/parity_manifest.json
	echo
	echo "If the extra change is an intentional aikit_version bump, commit THAT alone first (it"
	echo "re-stales the matrix), then re-run this script so it sees only deps_hash. That is the"
	echo "shape of 278637f followed by 9624dd9."
	exit 1
fi
changed=$(git diff --numstat -- testdata/parity_manifest.json | awk '{print $1}')
echo "    staleness gate green; ${changed:-0} deps_hash line(s) refreshed, nothing else."

head=$(git rev-parse --short HEAD)
# The execution arch is load-bearing: an expression-rewrite to a float path can pass the
# argmax+cosine goldens on one arch and breach on the other (arm64 fuses FMA; amd64 baseline does
# not — see parity-coverage-policy.md "arch-scoped"). Stamping it here is the difference between a
# later reader being able to READ which arch ran the f32 goldens vs having to infer it (or re-run).
arch="$(go env GOARCH 2>/dev/null || uname -m)"
echo
echo "==> PROOF (paste into the commit body — makes the exception auditable):"
echo "    non-numeric core refresh; validated_at preserved."
echo "    forward goldens green at ${head} on arch=${arch}: ${pass} passed / ${skip} skipped / 0 failed."
echo
echo "    Deps-Hash-Refresh: ${head} goldens=${pass} arch=${arch}"
echo
echo "    (Trailer renamed from Parity-Deps-Refresh, which had been used for dependency bumps too."
echo "     It is now documentation only — the recurrence counter reads the manifest diff, not this.)"

# Recurrence counter — MEASURED FROM THE MANIFEST, not from a commit trailer.
#
# It used to grep '^Parity-Deps-Refresh:'. That trailer is overloaded (it has been used for
# dependency bumps too) and it post-dates some refreshes, so it is wrong in BOTH directions —
# measured 2026-08-05: 35 commits carried the trailer, only 24 of those actually changed a
# deps_hash, and ~10 refreshes changed one without carrying it. A counter that can't be trusted
# either way trains people to dismiss the warning, which is worse than not counting.
#
# The precise definition of "a script-driven non-numeric refresh" is written into the manifest
# itself: deps_hash lines changed, validated_at lines NOT. A real re-validation moves both. Read
# that directly — self-correcting over history, and immune to how anyone words a commit.
count_refreshes() {
	local n=0 d
	for h in $(git log --format='%H' -- testdata/parity_manifest.json 2>/dev/null); do
		d=$(git show --format= --unified=0 "$h" -- testdata/parity_manifest.json 2>/dev/null)
		printf '%s\n' "$d" | grep -q '^[+-].*deps_hash' || continue
		printf '%s\n' "$d" | grep -q '^[+-].*validated_at' && continue
		n=$((n + 1))
	done
	printf '%s' "$n"
}
prior=$(count_refreshes)
echo
echo "==> ${prior} prior non-numeric deps_hash refreshes (deps_hash moved, validated_at preserved)."
echo "    Counted from the manifest history, not from commit trailers — see the note above."
echo
echo "    This is EXPECTED, not a defect to fix by building option C. The classification in"
echo "    docs/task-parity-staleness-diagnostic-seams.md found most refreshes are ordinary"
echo "    non-numeric edits to shared core files, NOT the inert diagnostic seams option C targets —"
echo "    and the recorded verdict was: DON'T build option C (it addresses a small minority while"
echo "    adding an invisible channel for a genuine numeric change to slip through), keep paying"
echo "    this small tax, and make the counter honest. The counter is now honest."
echo
echo "    Re-open option C only if the CLASSIFICATION shifts — i.e. refreshes start being dominated"
echo "    by inert diagnostic seams. The raw count alone is not that signal and never was."
