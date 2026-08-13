#!/usr/bin/env bash
# gpu_gate.sh — the pre-tag GPU correctness gate. Run it on EACH GPU box, paste the
# verdict, then tag.
#
# WHY THIS SCRIPT IS THE GATE. CI never runs the GPU backends: ci.yml only *builds* and
# *vets* under -tags cuda, because the runners have no GPU. Metal is darwin-only and CUDA
# is Linux+NVIDIA, so no single machine — and no CI job — can cover both. Every GPU
# correctness claim this project makes therefore rests on a human running tests by hand on
# two boxes. This makes that reproducible: one command, one verdict, provenance attached.
#
# It is deliberately NOT "run everything". A gate that takes two hours gets skipped, and a
# skipped gate is worse than no gate because it still implies assurance. This runs the
# checks that map to bugs we actually shipped.
#
# HONESTY RULES (learned the hard way):
#   - A skip is NOT a pass. Go prints "ok" for a package whose tests all skipped, so the
#     script counts real runs and reports SKIPPED separately. Green here must mean tested.
#   - Stray GPU processes invalidate results. A clean-tree control once "confirmed" a
#     pre-existing failure while 3.4 GB of leaked serve processes held the card — both
#     sides equally poisoned. This checks first and refuses to pretend.
#   - The CUDA suite must run SEQUENTIALLY (-p 1). Its tests each build a context and
#     several load real models; parallel packages contend for VRAM and the failures come
#     back as bogus numerics ("cosine 0.000000") rather than "you are out of memory".
#
# Usage:
#   scripts/gpu_gate.sh                # auto-detect the backend for this box
#   GOINFER_GATE_BACKEND=cuda  scripts/gpu_gate.sh
#   GOINFER_GATE_BACKEND=metal scripts/gpu_gate.sh
#
# Env:
#   GOINFER_GATE_BACKEND   cuda | metal (default: auto — nvidia-smi ⇒ cuda, darwin ⇒ metal)
#   GOINFER_GATE_MODELS    dir holding the real checkpoints (default: $HOME/models)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
DIRTY=""; [ -n "$(git status --porcelain 2>/dev/null)" ] && DIRTY=" +dirty"
DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODELS="${GOINFER_GATE_MODELS:-$HOME/models}"

BACKEND="${GOINFER_GATE_BACKEND:-}"
if [ -z "$BACKEND" ]; then
	if command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda
	elif [ "$(uname -s)" = "Darwin" ]; then BACKEND=metal
	else BACKEND=none; fi
fi

FAILED=0
SKIPPED=0
PASSED=0
RAN=0
NOTES=()

# GROUP ACCOUNTING (audit G-01). The tally used to be computed purely from what emitted, so a check
# that died mid-block simply vanished and the gate still reported PASS — it had tested nothing and
# said so in no way that a reader could notice. Counting what emitted can never detect what did not.
# So the expected groups are DECLARED up front and reconciled at the end: a group that emits no
# verdict, or an unexpected group id, is itself a FAIL.
case "$BACKEND" in
cuda)  EXPECT=(cleangpu seam suite parity heavy graphsforced cgofree ptx repo) ;;
metal) EXPECT=(cleangpu seam suite cgofree lifecycle prefill repo) ;;
*)     EXPECT=(cleangpu seam suite repo) ;;
esac
declare -A EMITTED=()
CURGROUP=""
grp() { CURGROUP="$1"; }                                  # set by each hdr, before its checks run
mark() { [ -n "$CURGROUP" ] && EMITTED[$CURGROUP]=1; return 0; }

hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASSED=$((PASSED + 1)); mark; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=$((FAILED + 1)); mark; }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIPPED=$((SKIPPED + 1)); NOTES+=("SKIPPED: $1"); mark; }

# nomatch: a `-run` pattern that matches NOTHING is a FAIL, not a zero-test pass (ported from
# aikit's gate). `go test -run NoSuchTest` exits 0 and prints "ok", so renaming a test away silently
# deletes a check while the gate stays green — the same shape as a skip counted as a pass. Counts
# "=== RUN" lines, so it needs -v on the invocation it judges.
# vram_note: on a cosine of EXACTLY zero, print the card's free VRAM and point at A12.
#
# A cosine of 0.000000 is not a parity result — an all-zero buffer is what a failed allocation leaves
# behind, and this script's own header records the history: an OOM wore a parity bug's clothes long
# enough that two people concluded "the tests just interfere" and moved on.
#
# WORDING IS DELIBERATE. It states the READING and points at the entry; it does NOT name a mechanism.
# "Suspect retention" was the obvious phrasing and is now DISPROVEN — A12 measured Close() returning
# all 4892 MiB synchronously in 123 ms, with a 0 MiB asynchronous tail. Naming a mechanism a gate
# cannot see is how the last three explanations became someone else's wasted afternoon.
vram_note() {
	printf '%s' "$1" | grep -q 'cosine 0\.000000' || return 0
	local free
	free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"
	echo "      NOTE: a cosine of EXACTLY 0.000000 is an all-zero buffer, which is what a failed"
	echo "            allocation leaves behind — not necessarily a numerics defect."
	echo "            free VRAM on the card right now: ${free}"
	echo "            (measured AFTER the run, so it is a bound, not the value at failure)"
	echo "            See docs/QUEUE.md A12. Mechanism NOT established: parallelism (-p 1, one"
	echo "            package, no t.Parallel) and async teardown (Close is synchronous) are both"
	echo "            REFUTED, and there is no leak. Do not assume; measure."
}

nomatch() { # $1 = go test output, $2 = pattern (for the message); returns 0 if nothing ran
	[ "$(printf '%s' "$1" | grep -cE '^=== RUN' || true)" -eq 0 ]
}

hdr "provenance"
echo "  repo        $COMMIT$DIRTY"
echo "  date (UTC)  $DATE"
echo "  host        $(uname -s) $(uname -m)"
echo "  backend     $BACKEND"
echo "  models      $MODELS"
case "$BACKEND" in
cuda)
	if command -v nvidia-smi >/dev/null 2>&1; then
		echo "  gpu         $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
	fi
	;;
metal) echo "  gpu         $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')" ;;
esac
[ -n "$DIRTY" ] && NOTES+=("WORKING TREE DIRTY — this verdict does not describe a committed state.")

# ---- 0. the card must be quiet, or every memory-sensitive result below is noise ----
grp cleangpu; hdr "0. clean GPU"
if [ "$BACKEND" = cuda ] && command -v nvidia-smi >/dev/null 2>&1; then
	PROCS="$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
	USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d ' MiB')"
	if [ -n "$PROCS" ]; then
		printf '  processes holding the GPU:\n%s\n' "$PROCS" | sed 's/^/    /'
		# Only OUR leftovers are a problem to call out; a display server legitimately holds some.
		if echo "$PROCS" | grep -qiE "serve|goinfer|gi_serve"; then
			fail "stray goinfer/serve processes hold the GPU — kill them and re-run; leaked processes have
        silently poisoned a control run before (both sides equally, which made a real bug look
        pre-existing). Try: pkill -f '[s]erve'"
		else
			pass "no stray goinfer processes (${USED:-?} MiB in use by others)"
		fi
	else
		pass "no compute processes on the GPU (${USED:-?} MiB baseline)"
	fi
else
	skip "clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)"
fi

# ---- 1. seam: no GPU needed, and it is the class that cost five weeks ----
grp seam; hdr "1. seam (runs anywhere — no GPU, no model download)"
if out="$(go test ./decoder/ -run 'TestSeam_' -count=1 -v 2>&1)"; then
	if nomatch "$out" ; then
		fail "seam gate: -run 'TestSeam_' matched NO tests — the pattern or the test names moved, and a
        zero-test run exits 0. This gate reported PASS for a check it never executed."
	else
	RAN=$((RAN + 1))
	pass "serve↔decoder↔backend seam: residency is actually reached, backend names validate"
	fi
else
	fail "seam gate — GPU serve may be silently CPU-only (see 7557723 / 727f198)"
	echo "$out" | tail -8 | sed 's/^/      /'
fi

# ---- 2/3/4. per-backend suites ----
case "$BACKEND" in
cuda)
	# The header used to read "CUDA kernels + parity" while running NEITHER the resident parity
	# gates NOR anything that asserts a forward. Every resident parity gate is behind
	# `goinfer_testhooks` (backend_wired, the gemma4/GLM/MoE resident parities, sliding-window),
	# so for the whole of v0.10.x/v0.11.0 this block ran 53 kernel-level tests and the release
	# record said "full cuda suite". Combined with parity_manifest.json's shared_sets covering
	# decoder/*.go ONLY — no cuda/ file appears in it, so deps_hash cannot go stale on
	# resident.go — a change to CUDA forward numerics had no enforced signal anywhere in the
	# gate. TWO groups now, because they answer different questions and one is not evidence for
	# the other (audit G-01: the artifact must not be adjacent to what it is read as).
	grp suite; hdr "2a. CUDA kernel-level suite (no testhooks: kernels, admission, lint)"
	# -v is REQUIRED, not cosmetic: without it `go test` prints no "--- SKIP" lines at all, so the
	# census below silently counts zero and prints nothing — a check that reports nothing while
	# looking healthy, which is the very defect this block exists to close. Caught by writing it
	# without -v first and getting an empty census on a suite known to skip six.
	if out="$(CGO_ENABLED=0 go test -tags cuda -p 1 ./cuda/ -count=1 -short -v 2>&1)"; then
		RAN=$((RAN + 1))
		pass "cuda kernel-level suite"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "cuda kernel-level suite"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi
	# Census the skips INSIDE the passing suite. "ok" hides them, and a skip is not a pass.
	SK="$(printf '%s' "$out" | grep -cE '^--- SKIP' || true)"
	if [ "${SK:-0}" -eq 0 ]; then
		# A zero here is far more likely to mean "the census broke" than "nothing skipped".
		echo "      skip census: 0 — verify -v is still on this invocation before believing it"
	fi
	if [ "${SK:-0}" -gt 0 ]; then
		echo "      skipped within it: $SK (all GOINFER_HEAVY_TESTS=1 — 4 bandwidth benchmarks,"
		echo "      TestRealWeightGemvParity (real q4_K_M weights), TestResidentSpecServe (loads a 1.5B model))"
		printf '%s' "$out" | grep -E '^--- SKIP' | sed 's/--- SKIP: /        · /;s/ (.*//'
	fi

	grp parity; hdr "2b. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)"
	if out="$(CGO_ENABLED=0 go test -tags 'cuda goinfer_testhooks' -p 1 ./cuda/ -count=1 2>&1)"; then
		RAN=$((RAN + 1))
		pass "resident parity gates (gemma4 dense/two-geom/MoE+router, GLM partial-rotary, mixtral MoE, sliding-window, rope-partial)"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "resident parity gates — a CUDA forward moved. This is the group 2a cannot see."
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
		vram_note "$out"
	fi

	# ---- heavy tier: the 119-test real-model group NOTHING has ever run ----
	grp heavy; hdr "2c. heavy tier (GOINFER_HEAVY_TESTS=1 — real models; ~28 min)"
	# This tier existed and was never executed by anything: no script set the variable, so the tests
	# behind it were written, committed, and skipped forever. Declared here so it cannot quietly stop
	# running again, and TIMED into the verdict line so its cost is visible up front rather than
	# discovered by someone waiting 28 minutes for a gate they thought took one.
	if [ -n "${GOINFER_GATE_SKIP_HEAVY:-}" ]; then
		skip "heavy tier (GOINFER_GATE_SKIP_HEAVY set) — the real-model gates did NOT run: 26B expert
        streaming, real-weight GEMV parity, resident spec-serve, the bandwidth benchmarks"
	else
		HEAVY_T0=$(date +%s)
		# Persist the FULL output. The first run of this group failed and printed twelve lines of
		# PASSING log text, because the failure filter was copied from groups 2a/2b which do not use
		# -v — under -v every t.Logf carries a "file.go:NNN:" prefix, so the filter matched everything
		# and `head` truncated the actual "--- FAIL" away. A 28-minute group that loses its own
		# evidence has to be re-run from scratch to learn what broke, which is exactly the cost this
		# gate exists to avoid. Keep the log; point at it on failure.
		HEAVY_LOG="${TMPDIR:-/tmp}/gpu_gate_heavy_$$.log"
		# STREAM progress, do not buffer 28 minutes of silence. The first version captured straight
		# into `out="$(...)"`, so nothing printed until the group finished — and a running heavy tier
		# and a hung one produced byte-identical output (none). Liveness had to be checked with
		# nvidia-smi, from outside the gate, which is the silence-reads-as-health shape this script
		# exists to prevent, one level up in the tooling.
		#
		# So: write the full log, stream one line per completed test, and take go test's status from
		# PIPESTATUS[0] — NOT the pipeline's, which is `sed`'s and always zero. That last point is
		# load-bearing: piping without it makes this group structurally incapable of reporting red.
		# Mutation-checked both directions.
		# ---- THE PARTITION, DERIVED FROM A MARKER (A13) ----
		# The draining tests take the device to refusal, which is A13's only reproducible poisoning
		# stimulus. They therefore run in their OWN process, after the main tier, so an exhausted
		# device cannot reach anything else.
		#
		# The group is DERIVED, not listed. `drainsDevice(t, why)` in cuda/drain_marker_test.go is
		# the marker; this awk walks the test files, tracks the enclosing `func TestX`, and emits X
		# for each one that calls it. A hand-kept -run list would be a constant restating a property
		# — the same drift shape the census denominators just made visible.
		DRAIN_TESTS="$(awk '
			/^func Test[A-Za-z0-9_]*\(/ { name=$2; sub(/\(.*/,"",name); next }
			/drainsDevice\(t,/ { if (name != "") { print name; name="" } }
		' "$ROOT"/cuda/*_test.go | sort -u)"
		DRAIN_N="$(printf '%s\n' "$DRAIN_TESTS" | grep -c . || true)"
		if [ "${DRAIN_N:-0}" -eq 0 ]; then
			fail "heavy tier partition: the marker derivation found ZERO draining tests"
			NOTES+=("drainsDevice() marker matched nothing — the derivation is broken, not the tree clean")
		fi
		DRAIN_RE="^($(printf '%s\n' "$DRAIN_TESTS" | paste -sd'|' -))$"
		TOTAL_N="$(CGO_ENABLED=0 go test -tags 'cuda goinfer_testhooks' ./cuda/ -list '.*' 2>/dev/null \
			| grep -cE '^Test' || true)"
		echo "      partition (derived from drainsDevice() in cuda/drain_marker_test.go):"
		echo "        package total     ${TOTAL_N:-?} test(s)   [go test -list '.*']"
		echo "        drain group        ${DRAIN_N} test(s)   $(printf '%s' "$DRAIN_TESTS" | tr '\n' ' ')"
		echo "        main group         $(( ${TOTAL_N:-0} - DRAIN_N )) test(s)   [complement, by construction]"

		echo "      streaming (one line per test; full log at $HEAVY_LOG):"
		GOINFER_HEAVY_TESTS=1 CGO_ENABLED=0 go test -tags 'cuda goinfer_testhooks' -p 1 ./cuda/ \
			-count=1 -timeout 60m -v 2>&1 | tee "$HEAVY_LOG" \
			| grep --line-buffered -E '^--- (PASS|FAIL|SKIP)' \
			| sed -u 's/^--- /        · /'
		HEAVY_RC=${PIPESTATUS[0]}

		# ---- INVOCATION 2: the drain group, its own process, after everything else ----
		# GOINFER_DRAIN_GROUP is what un-skips the marker. -run restricts it to the derived set so
		# the rest of the package is not paid for twice.
		DRAIN_LOG="${TMPDIR:-/tmp}/gpu_gate_drain_$$.log"
		echo "      drain group, separate process (full log at $DRAIN_LOG):"
		GOINFER_DRAIN_GROUP=1 GOINFER_HEAVY_TESTS=1 CGO_ENABLED=0 \
			go test -tags 'cuda goinfer_testhooks' -p 1 ./cuda/ \
			-count=1 -timeout 20m -run "$DRAIN_RE" -v 2>&1 | tee "$DRAIN_LOG" \
			| grep --line-buffered -E '^--- (PASS|FAIL|SKIP)' \
			| sed -u 's/^--- /        · /'
		DRAIN_RC=${PIPESTATUS[0]}
		dout="$(cat "$DRAIN_LOG")"

		# ---- RECONCILIATION: a partition that silently drops a test is the failure mode ----
		# Counted from the marker's own tokens, in both directions:
		#   main tier   -> every marked test must have SKIPPED, and no more than the derived count
		#   drain tier  -> every marked test must have RUN
		# A derivation miss shows up as a main-tier DRAIN-GROUP-SKIP with no matching drain-tier run,
		# and fails here rather than being quietly absent from both halves.
		MAIN_SKIPPED="$(grep -c 'DRAIN-GROUP-SKIP' "$HEAVY_LOG" 2>/dev/null || true)"
		DRAIN_RAN="$(grep -c 'DRAIN-GROUP-RUN' "$DRAIN_LOG" 2>/dev/null || true)"
		echo "      reconciliation: marked=${DRAIN_N}  skipped-in-main=${MAIN_SKIPPED:-0}  ran-in-drain=${DRAIN_RAN:-0}"
		if [ "${MAIN_SKIPPED:-0}" -ne "$DRAIN_N" ] || [ "${DRAIN_RAN:-0}" -ne "$DRAIN_N" ]; then
			fail "heavy tier partition does not reconcile — a test is in neither half or in both"
			NOTES+=("partition mismatch: ${DRAIN_N} marked, ${MAIN_SKIPPED:-0} skipped in main, ${DRAIN_RAN:-0} ran in drain")
		fi
		if [ "$DRAIN_RC" -eq 0 ]; then
			RAN=$((RAN + 1)); pass "drain group (${DRAIN_N} test(s), separate process)"
		else
			fail "drain group (${DRAIN_N} test(s), separate process)"
			echo "$dout" | grep -E "^(---|    ---) FAIL|^panic:" | head -8 | sed 's/^/      /'
			echo "      full output: $DRAIN_LOG"
		fi
		out="$(cat "$HEAVY_LOG")"
		if [ "$HEAVY_RC" -eq 0 ]; then
			HEAVY_SECS=$(( $(date +%s) - HEAVY_T0 ))
			RAN=$((RAN + 1))
			pass "heavy tier (real models) — ${HEAVY_SECS}s"
			echo "$out" | grep -E "^ok" | sed 's/^/      /'
		else
			HEAVY_SECS=$(( $(date +%s) - HEAVY_T0 ))
			fail "heavy tier (real models) — ${HEAVY_SECS}s"
			# -v-safe: name the failing TESTS first, then their assertion lines. Never a bare
			# "file.go:N:" match, which under -v is every log line in the run.
			echo "$out" | grep -E "^(---|    ---) FAIL|^panic:" | head -12 | sed 's/^/      /'
			echo "      full output: $HEAVY_LOG"
		fi
		# Census its skips BY NAME with their reason. A tier whose value is "it runs the real models"
		# is worth nothing if the real-model tests inside it skipped.
		HSK="$(printf '%s' "$out" | grep -cE '^--- SKIP' || true)"
		echo "      ran $(printf '%s' "$out" | grep -cE '^=== RUN' || true) tests, skipped ${HSK:-0}"
		if [ "${HSK:-0}" -gt 0 ]; then
			printf '%s' "$out" | grep -E '^--- SKIP' | sed 's/--- SKIP: /        · /;s/ (.*//'
			NOTES+=("heavy tier skipped ${HSK} test(s) — see the 2c census for which")
		fi
	fi

	# ---- graphs, FORCED: the code path is otherwise never exercised on this box ----
	grp graphsforced; hdr "2d. CUDA graphs bit-exactness, FORCED (GOINFER_CUDA_GRAPHS_UNSAFE=1)"
	# SEPARATE and LABELLED, deliberately. admitGraphs declines under DEFAULT compute mode without
	# MPS, which is correct production behaviour and must stay that way — so on this box the graph
	# capture/replay path is never exercised at all. Forcing it here tests the CODE without changing
	# the admission POLICY. Keeping it out of 2b matters: a forced result must never be read as
	# evidence that graphs are admitted in production, which is the adjacency the group split exists
	# to prevent (audit G-01).
	if out="$(GOINFER_CUDA_GRAPHS_UNSAFE=1 CGO_ENABLED=0 go test -tags 'cuda goinfer_testhooks' -p 1 ./cuda/ -count=1 -run 'TestGemma4Graphs_' -v 2>&1)"; then
		if nomatch "$out"; then
			fail "graphs (forced): -run 'TestGemma4Graphs_' matched NO tests — zero-test runs exit 0"
		else
			RAN=$((RAN + 1))
			GSK="$(printf '%s' "$out" | grep -cE '^--- SKIP' || true)"
			pass "graphs replay == live launches, FORCED capture ($(printf '%s' "$out" | grep -cE '^=== RUN' || true) tests, ${GSK:-0} skipped)"
			[ "${GSK:-0}" -gt 0 ] && printf '%s' "$out" | grep -E '^--- SKIP' | sed 's/--- SKIP: /        · /;s/ (.*//'
		fi
	else
		fail "graphs bit-exactness FAILED under forced capture — replay diverges from live launches"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi
	NOTES+=("graphs are FORCED in 2d (GOINFER_CUDA_GRAPHS_UNSAFE): this proves the capture/replay CODE, not that graphs are admitted in production — admitGraphs still declines here (DEFAULT compute mode, no MPS).")

	grp cgofree; hdr "3. cgo-free (the whole premise — verify, never assume)"
	# Build the CUDA SUBMODULE entrypoint. The root ./cmd/serve has been a DELIBERATE compile
	# error under -tags cuda since v0.10.0 (cmd/serve/backendtag_guard_cuda.go: the root command
	# builds no backend, and failing loudly beats silently producing a CPU-only binary named as
	# though it had CUDA). This check pointed at the root command for that entire period, so it
	# could not pass — see audit G-01.
	if (cd cuda && CGO_ENABLED=0 go build -tags cuda -o /tmp/gpu_gate_serve ./cmd/serve) 2>/dev/null; then
		RAN=$((RAN + 1))
		LINKED="$(ldd /tmp/gpu_gate_serve 2>/dev/null | grep -iE "libcuda|libnvrtc|libcudart" || true)"
		if [ -n "$LINKED" ]; then
			fail "binary links CUDA libraries — the cgo-free claim is false:"
			echo "$LINKED" | sed 's/^/      /'
		else
			pass "serve builds CGO_ENABLED=0 and links no CUDA toolkit (driver is dlopen'd at runtime)"
		fi
		rm -f /tmp/gpu_gate_serve
	else
		fail "cuda/cmd/serve does not build under -tags cuda (CGO_ENABLED=0)"
	fi

	grp ptx; hdr "4. PTX reproduces from source, each at the NVRTC it records"
	# INTEGRITY: this block must ALWAYS reach pass/fail/skip. An earlier revision of it died on a
	# bash error midway and the gate still reported PASS overall — a check that can neither pass
	# nor fail is the same defect as one that can only fail (audit G-01). PTX4_DONE is asserted at
	# the end and checked after; if the block ever exits early, that becomes a FAIL, not silence.
	PTX4_DONE=0
	# Every .ptx states the toolchain that produced it in its own header:
	#   // Cuda compilation tools, release 12.6, V12.6.85
	# That is the artifact's provenance, and it is what we rebuild against — NOT whatever NVRTC
	# this box happens to default to. The tree legitimately carries a MIX (kernels added after a
	# toolchain bump were built at the newer one, and the audited ones are deliberately pinned),
	# so a single-toolchain rebuild reports a false FAIL on every file from the other era. That is
	# what made this check unpassable for the whole of v0.10.x/v0.11.0 — see audit G-01.
	#
	# Nothing is exempted by name. A file is only skipped when the NVRTC version IT RECORDS is not
	# installed here, and the skip names the version so it is actionable.
	#
	# Provide extra toolchains via GOINFER_NVRTC_DIRS (colon-separated dirs, each containing
	# lib/libnvrtc.so.12), e.g. a pinned pip venv:
	#   python3 -m venv ~/nvrtc-12.6.85 && ~/nvrtc-12.6.85/bin/pip install \
	#       nvidia-cuda-nvrtc-cu12==12.6.85 nvidia-cuda-runtime-cu12==12.6.77
	if [ -x cuda/build_ptx.sh ]; then
		PROBE="$(mktemp -d)"
		# Build version -> NVRTC dir by PROBING each candidate: compile a trivial kernel and read
		# the version out of the PTX it emits. Exact, and it exercises the same path the real
		# build uses (filename/soname heuristics only give major.minor, and the patch matters).
		declare -A NVRTC_FOR=()
		# GOINFER_NVRTC_DIRS is an OVERRIDE, not an addition: set it and ONLY those toolchains are
		# used. That makes the "toolchain absent" path reachable on a box that happens to have it,
		# which is how the counted-skip behaviour below is tested rather than assumed.
		if [ -n "${GOINFER_NVRTC_DIRS:-}" ]; then
			CANDS="$GOINFER_NVRTC_DIRS"
		else
			CANDS=""
			for g in "$HOME"/nvrtc-*/lib/python*/site-packages/nvidia "$HOME"/.venv*/lib/python*/site-packages/nvidia; do
				[ -d "$g/cuda_nvrtc/lib" ] && CANDS="$CANDS:$g"
			done
		fi
		printf 'extern "C" __global__ void p(float* o){ o[0]=1.f; }\n' > "$PROBE/p.cu"
		IFS=':' read -ra CAND_ARR <<< "$CANDS"
		for c in "${CAND_ARR[@]}"; do
			[ -z "$c" ] && continue
			L="$c/cuda_nvrtc/lib"; I="$c/cuda_runtime/include"
			[ -d "$L" ] || { L="$c/lib"; I="$c/include"; }
			[ -f "$L/libnvrtc.so.12" ] || continue
			if (cd cuda && LD_LIBRARY_PATH="$L" NVRTC_SO="$L/libnvrtc.so.12" \
				python3 nvrtc_compile.py "$PROBE/p.cu" "$PROBE/p.ptx" "${ARCH:-compute_75}" "$I") >/dev/null 2>&1; then
				V="$(sed -n 's|.*Cuda compilation tools, release [0-9.]*, V\([0-9.]*\).*|\1|p' "$PROBE/p.ptx" | head -1)"
				[ -n "$V" ] && [ -z "${NVRTC_FOR[$V]:-}" ] && NVRTC_FOR[$V]="$L|$I"
			fi
		done
		TCS="${!NVRTC_FOR[*]}"   # NOT ${!NVRTC_FOR[*]:-none}: that is indirect expansion and aborts the block
		echo "  toolchains available: ${TCS:-none}"

		BEFORE="$(mktemp -d)"; cp cuda/testdata/*.ptx "$BEFORE"/ 2>/dev/null
		DIFF=0; OKN=0; UNAVAIL=""; TOTAL=0; NUNAVAIL=0
		for f in cuda/testdata/*.ptx; do
			b="$(basename "$f" .ptx)"
			[ -f "cuda/$b.cu" ] || continue   # no source ⇒ not ours to reproduce
			TOTAL=$((TOTAL + 1))
			WANT="$(sed -n 's|.*Cuda compilation tools, release [0-9.]*, V\([0-9.]*\).*|\1|p' "$f" | head -1)"
			if [ -z "$WANT" ]; then
				UNAVAIL="$UNAVAIL $b(no recorded version)"; NUNAVAIL=$((NUNAVAIL + 1)); continue
			fi
			ENT="${NVRTC_FOR[$WANT]:-}"
			if [ -z "$ENT" ]; then
				UNAVAIL="$UNAVAIL $b(needs V$WANT)"; NUNAVAIL=$((NUNAVAIL + 1)); continue
			fi
			L="${ENT%%|*}"; I="${ENT##*|}"
			if (cd cuda && NVRTC_LIB="$L" CUDA_INC="$I" ./build_ptx.sh "$b") >/dev/null 2>&1; then
				if cmp -s "$f" "$BEFORE/$b.ptx"; then OKN=$((OKN + 1)); else
					DIFF=$((DIFF + 1)); echo "      DIFFERS: $b.ptx (rebuilt at its recorded V$WANT)"
				fi
			else
				UNAVAIL="$UNAVAIL $b(build failed at V$WANT)"; NUNAVAIL=$((NUNAVAIL + 1))
			fi
		done
		cp "$BEFORE"/*.ptx cuda/testdata/ 2>/dev/null   # restore; this check must not mutate the tree
		rm -rf "$BEFORE" "$PROBE"

		if [ "$DIFF" -eq 0 ] && [ "$OKN" -gt 0 ]; then
			RAN=$((RAN + 1))
			pass "$OKN/$TOTAL PTX regenerate byte-identically at their recorded NVRTC"
		elif [ "$DIFF" -gt 0 ]; then
			RAN=$((RAN + 1))
			fail "$DIFF PTX differ from their committed form — the shipped kernels do not match their .cu"
		else
			skip "PTX reproducibility (no usable NVRTC for any recorded version)"
		fi
		# A partial verification must never read as a full one: name the count AND the files, and
		# make it a counted SKIP so it appears in the verdict's skipped tally and the notes block.
		# "21/21 verified" and "11/21 verified" must be visibly different outcomes.
		[ -n "$UNAVAIL" ] && skip "$NUNAVAIL/$TOTAL PTX NOT verified on this box (toolchain absent):$UNAVAIL"
	else
		skip "PTX reproducibility (cuda/build_ptx.sh missing)"
	fi
	PTX4_DONE=1
	;;

metal)
	grp suite; hdr "2. Metal suite"
	if out="$(go test -p 1 ./metal/ -count=1 -short 2>&1)"; then
		RAN=$((RAN + 1))
		pass "full metal suite"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "metal suite"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi

	grp cgofree; hdr "3. cgo-free"
	if CGO_ENABLED=0 go build -o /tmp/gpu_gate_serve ./cmd/serve 2>/dev/null; then
		RAN=$((RAN + 1))
		pass "serve builds CGO_ENABLED=0 (Metal is dlopen'd via purego-objc)"
		rm -f /tmp/gpu_gate_serve
	else
		fail "serve does not build CGO_ENABLED=0"
	fi

	grp lifecycle; hdr "4. lifecycle"
	# Was a SKIP pointing at this work; the gate landed, so it is now a real check. Metal HAD the
	# same hole CUDA did — Close() froze a channel and freed nothing, leaking ~267 MB per
	# Load+Close on a 0.5B (aacec89). These run WITHOUT -short (the suite above uses it) because
	# they load real models, and they cover BOTH conditions: the sequential sawtooth, and a
	# second model alive — the case that made CUDA's first fix look correct when it was not.
	if [ -f "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" ]; then
		if out="$(go test -count=1 -run 'TestMetal_CloseFreesMemory|TestMetal_CloseWithSecondModelAlive' ./metal/ 2>&1)"; then
			RAN=$((RAN + 1))
			pass "Close() frees — sawtooth not staircase, and frees with a second model resident"
			echo "$out" | grep -E "trajectory|A\+B alive" | sed 's/^/      /'
		else
			fail "Close() leaks memory"
			echo "$out" | grep -E "^--- FAIL|LEAK|did NOT free|USE-AFTER-FREE|\.go:[0-9]+:" | head -8 | sed 's/^/      /'
		fi
	else
		skip "Close() lifecycle gate needs ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
	fi

	grp prefill; hdr "4b. prefill (f16-MMA TTFT — a shipped path, and it shipped NaN)"
	# The newest bug that maps to this doctrine. PrefillLast — the f16 simdgroup_matrix TTFT
	# path — emitted NaN logits at EVERY prompt length (incl. the minimal single-tile M=8) after
	# the LM head was pinned to int8: prefill still ran the int8 head weights through the int4
	# gemm_w4f16, misreading them as packed nibbles (19ef47d). It hit the DENSE control, a
	# model Metal ships. Nothing exercised it against a real checkpoint until a hand-run, so it
	# was invisible on push. These run WITHOUT -short (they load a real model) and cover both the
	# value (last-token logits match the sequential decode path) and the failure mode (finite
	# logits across single-tile / multi-tile / padded lengths). Reuses the 0.5B the lifecycle
	# gate already needs, via GOINFER_METAL_MODEL, so the gate's asset footprint stays one file.
	if [ -f "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" ]; then
		if out="$(GOINFER_METAL_MODEL="$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" \
			go test -count=1 -run 'TestPrefillParity|TestPrefillNoNaN' ./metal/ -v 2>&1)"; then
			RAN=$((RAN + 1))
			pass "prefill matches sequential decode and emits finite logits (no NaN)"
			echo "$out" | grep -E "argmax matches|faster TTFT" | sed 's/^/      /'
		else
			fail "prefill parity/NaN gate — the f16-MMA TTFT path is wrong on a shipped model"
			echo "$out" | grep -E "^--- FAIL|parity FAIL|contain NaN|\.go:[0-9]+:" | head -8 | sed 's/^/      /'
		fi
	else
		skip "prefill gate needs ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
	fi
	;;

*)
	grp suite; hdr "2-4. backend suites"
	skip "no GPU backend detected on this host — only the seam gate ran"
	;;
esac

# ---- 5. repo hygiene: run what CI runs, DERIVED rather than duplicated (B0) ----
#
# This block used to run `gofmt -l .` and `go vet ./decoder/ ./cmd/...` — a hand-written list that
# was a strict SUBSET of CI's: no staticcheck at all, vet without the goinfer_testhooks tag and over
# narrower packages, no build. So CI went red on `staticcheck -tags cuda` and stayed red for three
# commits, and running this gate — the thing you run INSTEAD of remembering — would not have caught
# it either.
#
# Adding staticcheck to the list would fix the instance and leave the class open: the next check CI
# gains reopens the gap. So the list is now DERIVED from .github/workflows/ci.yml by
# scripts/ci_checks.py, and a check CI adds appears here with no edit to this file.
grp repo; hdr "5. repo hygiene (derived from .github/workflows/ci.yml)"
# The queue's citations, commit AND path:line. A state document is cited without being re-derived, so
# a wrong reference in it propagates with more confidence than the same error in conversation —
# 9e5f8fa was cited several times, from the file, without anyone opening it, and cuda/resident.go:244
# kept an audit critical listed as open for weeks after it was fixed.
if python3 scripts/queue_citation_lint.py >/tmp/sha_lint.out 2>&1; then
	pass "$(tail -1 /tmp/sha_lint.out)"
else
	fail "docs/QUEUE.md SHA citations"
	head -6 /tmp/sha_lint.out | sed 's/^/      /'
fi
CI_ROWS="$(python3 scripts/ci_checks.py 2>/tmp/ci_checks.err)"
CI_RC=$?
if [ "$CI_RC" -ne 0 ] || [ -z "$CI_ROWS" ]; then
	# A derivation that fails must FAIL, not silently degrade to the old hand-written list. That
	# would be the very substitution this block exists to prevent, and it would look like a pass.
	fail "cannot derive CI's check set: $(head -2 /tmp/ci_checks.err 2>/dev/null | tr '\n' ' ')"
else
	# This host runs the linux jobs; the *-darwin ones are a COUNTED SKIP naming why, never dropped.
	# A check that is skipped and a check that passed must not look the same (B0a).
	case "$(uname -s)" in
	Darwin) MINE='-darwin$' ; OTHER="linux" ;;
	*)      MINE='^(root|gpu|cuda)$' ; OTHER="darwin" ;;
	esac
	ci_ok=0; ci_bad=0; ci_skipped=0
	while IFS=$'\t' read -r job name kind env cmd; do
		[ -z "$job" ] && continue
		if ! printf '%s' "$job" | grep -Eq "$MINE"; then
			ci_skipped=$((ci_skipped + 1)); continue
		fi
		if [ "${kind%%:*}" = "runner" ]; then
			skip "CI[$job] $name — ${kind#runner:}"
			continue
		fi
		# The ENVIRONMENT is part of the check. CI's root job has no go.work, so the
		# module-boundary guard sees the root module graph in isolation; a developer box with a
		# committed go.work unions every submodule and the guard reports a false red. Derived from
		# whether the job sets up a workspace, not hardcoded here.
		# "-" means no override. An EMPTY column would be collapsed by read (tab is IFS
		# whitespace), shifting every field after it.
		[ "$env" = "-" ] && env=""
		if printf '%b' "$cmd" | env ${env:+"$env"} bash >/tmp/ci_step.out 2>&1; then
			ci_ok=$((ci_ok + 1))
		else
			ci_bad=$((ci_bad + 1))
			fail "CI[$job] $name"
			head -5 /tmp/ci_step.out | sed 's/^/      /'
		fi
	done <<< "$CI_ROWS"
	if [ "$ci_skipped" -gt 0 ]; then
		skip "$ci_skipped ${OTHER}-only CI hygiene step(s) — wrong platform for this host"
	fi
	if [ "$ci_bad" -eq 0 ]; then
		pass "$ci_ok CI hygiene check(s) reproduced locally, derived from ci.yml"
	fi
fi

# ---- 6. group reconciliation: every DECLARED check must have emitted a verdict ----
# The generalisation of the check-4 guard. A block that dies mid-way emits nothing, and a tally
# computed from what emitted cannot see the hole — "ran 3" and "ran 4" are both plausible-looking
# numbers. Reconciling against the DECLARED set is what makes silence detectable (audit G-01).
CURGROUP=""   # reconciliation failures belong to no group
MISSING=""; UNEXPECTED=""
for g in "${EXPECT[@]}"; do [ -z "${EMITTED[$g]:-}" ] && MISSING="$MISSING $g"; done
for g in "${!EMITTED[@]}"; do
	case " ${EXPECT[*]} " in *" $g "*) ;; *) UNEXPECTED="$UNEXPECTED $g" ;; esac
done
if [ -n "$MISSING" ]; then
	fail "check group(s) declared but emitted NO verdict:$MISSING — the gate tested less than it reports (audit G-01)"
fi
if [ -n "$UNEXPECTED" ]; then
	fail "check group(s) emitted but not declared:$UNEXPECTED — update EXPECT so the tally stays meaningful"
fi

# ---- verdict ----
hdr "verdict"
# ONE UNIT: check groups. "6 declared / 4 ran" previously sat next to an unrelated count and a
# reader deciding whether to ship could not tell at a glance whether something was missing.
echo "  check groups: ${#EXPECT[@]} declared -> ${#EMITTED[@]} reported   |   verdicts within them: $PASSED pass, $SKIPPED skip, $FAILED fail"
# The release record now turns on this distinction: "the suite passed" is NOT "the forward is
# gated". Say which of the two actually happened, by name, so neither can be read as the other.
if [ "$BACKEND" = cuda ]; then
	s2a="not run"; s2b="not run"
	[ -n "${EMITTED[suite]:-}" ] && s2a="reported"
	[ -n "${EMITTED[parity]:-}" ] && s2b="reported"
	echo "  of which: kernel-level suite = $s2a   |   resident PARITY gates (forward asserted) = $s2b"
fi
if [ "${#EXPECT[@]}" -ne "${#EMITTED[@]}" ]; then
	echo "  (declared != reported: $(( ${#EXPECT[@]} - ${#EMITTED[@]} )) group(s) produced no verdict — see the FAIL above)"
fi
if [ "$SKIPPED" -gt 0 ]; then
	printf '\n  \033[33mSkipped — a skip is not a pass; this gate does NOT cover:\033[0m\n'
	for n in "${NOTES[@]}"; do printf '    - %s\n' "$n"; done
fi
if [ "$RAN" -eq 0 ]; then
	printf '\n  \033[31mNO GATE\033[0m — nothing actually ran. Do not read this as a pass.\n'
	exit 1
fi
if [ "$FAILED" -gt 0 ]; then
	printf '\n  \033[31mFAIL\033[0m — %s on %s @ %s. Do not tag.\n' "$BACKEND" "$(uname -s)" "$COMMIT$DIRTY"
	exit 1
fi
# THREE STATES, NOT TWO (ported from aikit 17f1517). Every check is green here. A dirty tree is not
# a failure of the CHECKS — it is a failure of PROVENANCE: this verdict names a commit, and an
# uncommitted edit means the verdict does not describe what that commit contains. Collapsing the two
# loses the distinction a reader actually needs: is the CODE broken, or is the EVIDENCE broken?
#
# It used to print "repo <sha>+dirty" in the provenance block and then PASS as normal — so the gate
# could emit a verdict that reads as "PASS at <sha>" for a tree that is not <sha>, with the whole
# distinction carried by a three-character suffix in a different block. Verdicts get pasted into tag
# messages; that is what this script is FOR.
if [ -n "$DIRTY" ]; then
	printf '\n  \033[33mINCONCLUSIVE\033[0m — %s/%s groups green, but the working tree is DIRTY.\n' "${#EXPECT[@]}" "${#EXPECT[@]}"
	printf '  The verdict names %s and the tree is not %s. Commit, then re-run before tagging.\n' "$COMMIT" "$COMMIT"
	git status --porcelain 2>/dev/null | head -10 | sed 's/^/    /'
	exit 1
fi
printf '\n  \033[32mPASS\033[0m — %s on %s @ %s (%s)\n' "$BACKEND" "$(uname -s)" "$COMMIT" "$DATE"
echo "  Paste this block for the tag. The OTHER box must pass its own run: no machine has both GPUs."
exit 0
