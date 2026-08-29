//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

// What is the MoE exclusion in A3/G24 actually worth?
//
// forwardn.go excludes MoE from --cpu-fast-attention unconditionally, on a stated
// mechanism: "an f32 QK reassociation flips a top-k expert at a near-tie and
// cascades ... so MoE is excluded here rather than trusted to the operator." The
// mechanism is real in kind — routing is discontinuous where a dense MLP is not.
// But it was never MEASURED on a MoE: no MoE appears in the A3 kernel-ratio
// record, and both tests in a3_divergence_test.go load the DENSE bench model,
// including TestA3FastAttentionDivergence, whose doc comment claims it pins "MoE
// excluded" while asserting nothing about MoE at all.
//
// That matters now because the term the exclusion protects is the one that
// dominates. On a Mellum2 slice, attention is 83.2% of prefill work at K=1024 and
// 97.1% at K=8192 — so the excluded lever is aimed at ~97% of the cost while
// expert-major batching competes for what is left.
//
// This measures BOTH halves of the trade at once:
//
//	COST — output cosine and max abs delta, acc64 vs f32, the same statistic
//	       a3_divergence_test.go reports for dense (which ships at 0.9976).
//	GAIN — wall-clock speedup of the same prefill.
//
// It deliberately does NOT decide anything. A cosine is not a routing-flip count,
// and a flip that changes generated TOKENS is the thing that would justify the
// guard; this bounds the perturbation, and says so.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
//	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoEExclusionIsMeasured -v
func TestA3MoEExclusionIsMeasured(t *testing.T) {
	path := os.Getenv("GOINFER_MELLUM_CKPT")
	if path == "" {
		t.Skip("set GOINFER_MELLUM_CKPT to a batched-path MoE checkpoint")
	}
	requireHeavyModel(t)
	K := 2048
	if v := os.Getenv("GOINFER_MELLUM_K"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &K); err != nil {
			t.Fatalf("GOINFER_MELLUM_K=%q: %v", v, err)
		}
	}
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("%s is not a MoE — this test measures the MoE exclusion", path)
	}
	if !m.canBatchN(K) {
		t.Fatalf("canBatchN(%d) = false", K)
	}
	// Varied ids: on a MoE the prompt IS the routing, and a constant-id prompt
	// collapses the top-k to one near-identical set — which would understate
	// exactly the near-tie flips this test exists to provoke.
	vocab := m.w.arch.VocabSize
	ids := make([]int, K)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	run := func(probe bool) ([]float32, time.Duration) {
		t.Helper()
		t.Setenv("GOINFER_CPU_FAST_ATTENTION", map[bool]string{true: "1", false: "0"}[probe])
		start := time.Now()
		out, err := m.forwardLayersN(deadlineCtx(t), ids, m.NewCache(K+8), cpuFastAttention())
		if err != nil {
			t.Fatalf("probe=%v: %v", probe, err)
		}
		return out, time.Since(start)
	}

	fmt.Fprintf(os.Stderr, "A3-MoE: K=%d acc64 arm starting %s\n", K, time.Now().Format("15:04:05"))
	base, tBase := run(false)
	fmt.Fprintf(os.Stderr, "A3-MoE: acc64 %.1fs; f32 arm starting %s\n", tBase.Seconds(), time.Now().Format("15:04:05"))
	fast, tFast := run(true)

	var dot, na, nb, maxAbs float64
	for i := range base {
		a, b := float64(base[i]), float64(fast[i])
		dot += a * b
		na += a * a
		nb += b * b
		if d := math.Abs(a - b); d > maxAbs {
			maxAbs = d
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos == 1.0 && maxAbs == 0 {
		t.Fatalf("K=%d: the probe changed nothing — the seam is not wired, and any conclusion "+
			"drawn from this run would be about a test that did not run", K)
	}
	fmt.Fprintf(os.Stderr,
		"A3-MoE-exclusion: K=%d cosine=%.9f maxAbs=%.4g | acc64 %.1fs -> f32 %.1fs (%.2fx)\n"+
			"  dense ships at cosine 0.9976 behind this same flag\n",
		K, cos, maxAbs, tBase.Seconds(), tFast.Seconds(), tBase.Seconds()/tFast.Seconds())
}
