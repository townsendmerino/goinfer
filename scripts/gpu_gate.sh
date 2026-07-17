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

hdr() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=$((FAILED + 1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIPPED=$((SKIPPED + 1)); NOTES+=("SKIPPED: $1"); }

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
hdr "0. clean GPU"
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
hdr "1. seam (runs anywhere — no GPU, no model download)"
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
	hdr "2. CUDA kernels + parity (sequential: these contend for VRAM)"
	if out="$(CGO_ENABLED=0 go test -tags cuda -p 1 ./cuda/ -count=1 -short 2>&1)"; then
		RAN=$((RAN + 1))
		pass "full cuda suite"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "cuda suite"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi

	hdr "3. cgo-free (the whole premise — verify, never assume)"
	if CGO_ENABLED=0 go build -tags cuda -o /tmp/gpu_gate_serve ./cmd/serve 2>/dev/null; then
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
		fail "serve does not build under -tags cuda"
	fi

	hdr "4. PTX is reproducible from committed sources"
	if [ -x cuda/build_ptx.sh ]; then
		BEFORE="$(mktemp -d)"; cp cuda/testdata/*.ptx "$BEFORE"/ 2>/dev/null
		if (cd cuda && ./build_ptx.sh >/dev/null 2>&1); then
			RAN=$((RAN + 1))
			DIFF=0
			for f in cuda/testdata/*.ptx; do cmp -s "$f" "$BEFORE/$(basename "$f")" || DIFF=$((DIFF + 1)); done
			if [ "$DIFF" -eq 0 ]; then
				pass "all $(ls cuda/testdata/*.ptx | wc -l | tr -d ' ') PTX regenerate byte-identically"
			else
				fail "$DIFF PTX file(s) differ from their committed form — the shipped kernels do not match their .cu"
			fi
		else
			skip "PTX rebuild (no NVRTC toolchain here — see cuda/build_ptx.sh)"
		fi
		rm -rf "$BEFORE"
	else
		skip "PTX reproducibility (cuda/build_ptx.sh missing)"
	fi
	;;

metal)
	hdr "2. Metal suite"
	if out="$(go test -p 1 ./metal/ -count=1 -short 2>&1)"; then
		RAN=$((RAN + 1))
		pass "full metal suite"
		echo "$out" | grep -E "^ok" | sed 's/^/      /'
	else
		fail "metal suite"
		echo "$out" | grep -E "^--- FAIL|\.go:[0-9]+:" | head -12 | sed 's/^/      /'
	fi

	hdr "3. cgo-free"
	if CGO_ENABLED=0 go build -o /tmp/gpu_gate_serve ./cmd/serve 2>/dev/null; then
		RAN=$((RAN + 1))
		pass "serve builds CGO_ENABLED=0 (Metal is dlopen'd via purego-objc)"
		rm -f /tmp/gpu_gate_serve
	else
		fail "serve does not build CGO_ENABLED=0"
	fi

	hdr "4. lifecycle"
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
	;;

*)
	hdr "2-4. backend suites"
	skip "no GPU backend detected on this host — only the seam gate ran"
	;;
esac

# ---- 5. shared: the whole non-GPU suite still has to be green ----
hdr "5. repo (CPU path, formatting, vet)"
if [ -n "$(gofmt -l . 2>/dev/null)" ]; then
	fail "gofmt: $(gofmt -l . | tr '\n' ' ')"
else
	pass "gofmt clean"
fi
if go vet ./decoder/ ./cmd/... >/dev/null 2>&1; then pass "go vet"; else fail "go vet"; fi

# ---- verdict ----
hdr "verdict"
echo "  ran $RAN check group(s), $SKIPPED skipped, $FAILED failed"
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
