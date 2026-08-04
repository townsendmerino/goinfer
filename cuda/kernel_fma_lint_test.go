//go:build cuda

package cuda

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

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
	// Contracted CUDA kernels the decode and/or batched forward paths reach. moe.cu is the audited,
	// frozen MoE PTX — reviewed separately (editing it needs its own audit), not linted here.
	files := []string{
		"fused_qkv.cu", "glue.cu", "prefill_batched.cu", "decode_splitkv.cu",
		"gemv_w4a8_rn.cu", "gemv_w4a8_batched.cu", "gemv_w4a8_staged.cu", "gemv_fwd.cu",
	}
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
		`\bfor\s*\(|\(long\)|\(int\)|\(unsigned`) // loop headers + int-cast arithmetic
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
