//go:build cuda

package cuda

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// lintedKernels is the contracted-kernel list TestKernelFMALint checks, and the list
// TestKernelFMALint_coversEmbeddedPTX holds to the set of kernels actually shipped as PTX.
//
// moe.cu is the audited, FROZEN MoE PTX — reviewed separately (editing it needs its own audit),
// deliberately not linted here; it is the single exemption below.
//
// router_f32.cu was added AFTER this lint and never joined the list (audit C-16), so the pure-f32
// Gemma-4 router projection — on the production decode path, and the one path the repo calls "the
// discrete-failure path" because a near-tie flips which expert runs — sat unguarded. A kernel is
// not covered by being contracted; it is covered by being in THIS list.
var lintedKernels = []string{
	"fused_qkv.cu", "glue.cu", "prefill_batched.cu", "decode_splitkv.cu",
	"gemv_w4a8_rn.cu", "gemv_w4a8_batched.cu", "gemv_w4a8_staged.cu", "gemv_fwd.cu",
	"router_f32.cu", "argmax.cu",
}

// fmaLintExempt lists embedded PTX deliberately outside the lint, with the reason. Anything else
// that is embedded MUST be linted — see TestKernelFMALint_coversEmbeddedPTX.
var fmaLintExempt = map[string]string{
	"moe.cu": "frozen audited MoE PTX — changes require their own audit, not a lint pass",
}

// TestKernelFMALint enforces the bit-identity rule at BUILD TIME: no bare float multiply-accumulate
// in any kernel under a bit-identity contract. A bare MAC (`x += a*b`, `= a*b + c`) lets the compiler
// CHOOSE fma vs mul+add, and separately-compiled kernels that share a contract (a decode kernel and
// its batched counterpart) then compile to ~1 ULP-different, DATA-DEPENDENT numerics — invisible on
// uniform fixtures, an 84% token-stream divergence on real weights (docs/task-batched-prefill-
// bitidentity.md). Every MAC must be an explicit intrinsic (__fmaf_rn / __fmul_rn / __fadd_rn) so no
// compiler discretion remains. This catches the CAUSE at compile time — before any numerical test,
// and independent of the NVRTC version that JITs the PTX. A new bare MAC fails the build. (aikit's
// gemv_quant.cu carries the same rule in its own repo — see its header.)
func TestKernelFMALint(t *testing.T) {
	files := lintedKernels
	// Float MAC forms, with the multiply as ` * ` (spaces both sides — so a pointer decl `float* p`
	// with `*` glued to the type is NOT matched):
	//   accumulate:  ident += <expr> * <expr>        (facc/ss/dot/acc — the bug class)
	//   expression:  ... <expr> * <expr> [+-] <expr>  (val = a*b + c ; a*c - b*s)
	macAccum := regexp.MustCompile(`\b\w[\w.]*\s*\+=\s*[^;/]* \* `)
	macExpr := regexp.MustCompile(` \* [^;/]* [+\-] `)
	compliant := regexp.MustCompile(`__fmaf_rn|__fmul_rn|__fadd_rn|__fma_rn|__dp4a|__shfl`)
	arrayIndex := regexp.MustCompile(`\[[^\]]*\]`) // strip array indices (int arithmetic) before matching
	// Lines that TEXTUALLY match a MAC but are integer / pointer / index arithmetic, not a float MAC:
	// a type/pointer declaration, a loop header, or any (long)/(int)/(unsigned) cast (int arithmetic —
	// float MACs cast with (float) or not at all, which the compliant check already covers).
	isDeclOrIndex := regexp.MustCompile(`^\s*(const\s+)?(unsigned\s+)?(int|long|short|char|size_t|void|float\*|double\*|__half\*|const int\*|const unsigned)\b|` +
		`^\s*(const\s+)?[\w]+\s*\*+\s*\w+\s*=|` + // pointer declaration: T* p = ...
		`\(long\)|\(int\)|\(unsigned`) // int-cast arithmetic
	// A for-header is INTEGER arithmetic (the init/cond/incr), but the loop BODY on the same line is
	// not — a single-line `for (…) acc += a[k] * b[k];` carries a real float MAC. Skipping the whole
	// line (the old `\bfor\s*\(` in isDeclOrIndex) let router_f32.cu's two MACs pass unseen (audit
	// R-04). Strip only the `for (…)` header, then lint the remaining body.
	forHeader := regexp.MustCompile(`\bfor\s*\([^)]*\)`)
	// remaining genuine non-MACs: reduce-adds (no '*'), lone float muls (no +/-), int array indices.
	whitelist := []string{
		"red[t] += red[t", "rednf[t] += rednf[t", "qkred[t] += qkred[t", // tree reduce-adds
		"ss += (double)",                                   // qk-norm double reduction — identical expression in both kernels
		"= vm[", "= v[", "= km[", "vc[o", "kc[o", "cache[", // stores with int index
	}
	violations := 0
	for _, fn := range files {
		b, err := os.ReadFile(fn)
		if err != nil {
			t.Fatalf("read %s: %v", fn, err)
		}
		for i, line := range strings.Split(string(b), "\n") {
			code := line
			if k := strings.Index(code, "//"); k >= 0 {
				code = code[:k]
			}
			code = forHeader.ReplaceAllString(code, "") // strip the loop header; keep any single-line body (R-04)
			if strings.TrimSpace(code) == "" || compliant.MatchString(code) || isDeclOrIndex.MatchString(code) {
				continue
			}
			code = arrayIndex.ReplaceAllString(code, "[]") // remove int index arithmetic before MAC match
			if !macAccum.MatchString(code) && !macExpr.MatchString(code) {
				continue
			}
			wl := false
			for _, w := range whitelist {
				if strings.Contains(code, w) {
					wl = true
					break
				}
			}
			if wl {
				continue
			}
			t.Errorf("%s:%d bare float MAC — use __fmaf_rn / __fmul_rn: %s", fn, i+1, strings.TrimSpace(line))
			violations++
		}
	}
	if violations == 0 {
		t.Logf("all float MACs in %d contracted kernels are explicit intrinsics ✓", len(files))
	}
}

// TestKernelFMALint_coversEmbeddedPTX is the gate on the gate (audit C-16).
//
// TestKernelFMALint only checks the kernels someone remembered to list. That is how router_f32.cu
// — production PTX on the Gemma-4 decode path — went unguarded from the day it was added: nothing
// connected "this kernel ships" to "this kernel is linted", so the list silently described a subset
// of reality. Adding router_f32.cu to the list fixes today's gap and nothing else; this test fixes
// the CLASS, by deriving the required set from what kernels.go actually embeds.
//
// A new kernel now fails here the moment it is embedded without being linted or explicitly exempted,
// which is the point at which the author is still holding the context to decide which it should be.
func TestKernelFMALint_coversEmbeddedPTX(t *testing.T) {
	src, err := os.ReadFile("kernels.go")
	if err != nil {
		t.Fatalf("read kernels.go: %v", err)
	}
	embedded := regexp.MustCompile(`//go:embed testdata/([\w]+)\.ptx`).FindAllStringSubmatch(string(src), -1)
	if len(embedded) == 0 {
		t.Fatal("no //go:embed testdata/*.ptx found in kernels.go — this test is no longer measuring anything")
	}
	linted := make(map[string]bool, len(lintedKernels))
	for _, f := range lintedKernels {
		linted[f] = true
	}
	for _, m := range embedded {
		cu := m[1] + ".cu"
		if linted[cu] {
			continue
		}
		if why, ok := fmaLintExempt[cu]; ok {
			t.Logf("exempt: %s — %s", cu, why)
			continue
		}
		t.Errorf("%s is embedded as production PTX but is neither in lintedKernels nor fmaLintExempt — "+
			"a bit-identity-contracted kernel shipping without the FMA lint is exactly audit C-16", cu)
	}
	t.Logf("%d embedded PTX kernels: %d linted, %d exempt", len(embedded), len(lintedKernels), len(fmaLintExempt))
}
