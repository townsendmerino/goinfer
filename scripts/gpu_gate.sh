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
RAN=0
NOTES=()

# GROUP ACCOUNTING (audit G-01). The tally used to be computed purely from what emitted, so a check
# that died mid-block simply vanished and the gate still reported PASS — it had tested nothing and
# said so in no way that a reader could notice. Counting what emitted can never detect what did not.
# So the expected groups are DECLARED up front and reconciled at the end: a group that emits no
# verdict, or an unexpected group id, is itself a FAIL.
case "$BACKEND" in
cuda)  EXPECT=(cleangpu seam suite cgofree ptx repo) ;;
metal) EXPECT=(cleangpu seam suite cgofree lifecycle prefill repo) ;;
*)     EXPECT=(cleangpu seam suite repo) ;;
esac
declare -A EMITTED=()
CURGROUP=""
grp() { CURGROUP="$1"; }                                  # set by each hdr, before its checks run
mark() { [ -n "$CURGROUP" ] && EMITTED[$CURGROUP]=1; return 0; }

hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; mark; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=$((FAILED + 1)); mark; }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIPPED=$((SKIPPED + 1)); NOTES+=("SKIPPED: $1"); mark; }

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
if out="$(go test ./decoder/ -run 'TestSeam_' -count=1 2>&1)"; then
	RAN=$((RAN + 1))
	pass "serve↔decoder↔backend seam: residency is actually reached, backend names validate"
else
	fail "seam gate — GPU serve may be silently CPU-only (see 7557723 / 727f198)"
	echo "$out" | tail -8 | sed 's/^/      /'
fi

# ---- 2/3/4. per-backend suites ----
case "$BACKEND" in
cuda)
	grp suite; hdr "2. CUDA kernels + parity (sequential: these contend for VRAM)"
	if out="$(CGO_ENABLED=0 go test -tags cuda -p 1 ./cuda/ -count=1 -short 2>&1)"; then
		RAN=$((RAN + 1))
		pass "full cuda suite"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "cuda suite"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi

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

# ---- 5. shared: the whole non-GPU suite still has to be green ----
grp repo; hdr "5. repo (CPU path, formatting, vet)"
if [ -n "$(gofmt -l . 2>/dev/null)" ]; then
	fail "gofmt: $(gofmt -l . | tr '\n' ' ')"
else
	pass "gofmt clean"
fi
if go vet ./decoder/ ./cmd/... >/dev/null 2>&1; then pass "go vet"; else fail "go vet"; fi

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
echo "  groups declared ${#EXPECT[@]}, emitted ${#EMITTED[@]}; ran $RAN check(s), $SKIPPED skipped, $FAILED failed"
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
printf '\n  \033[32mPASS\033[0m — %s on %s @ %s (%s)\n' "$BACKEND" "$(uname -s)" "$COMMIT$DIRTY" "$DATE"
echo "  Paste this block for the tag. The OTHER box must pass its own run: no machine has both GPUs."
exit 0
