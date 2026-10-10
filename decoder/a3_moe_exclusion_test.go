//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"
)

// What was the MoE exclusion in A3/G24 worth? Measures both halves of the trade on a MoE checkpoint:
// COST (output cosine and max abs delta, acc64 vs f32, the statistic a3_divergence_test.go reports for dense)
// and GAIN (wall-clock speedup of the same prefill). It decides nothing: a cosine is not a routing-flip count,
// and a flip that changes generated tokens is what would justify a guard (see a3_moe_routeflip_test.go and
// a3_moe_tokenlevel_test.go). cpuFastAttention no longer excludes MoE. It asserts only that the probe changed
// something. Context and figures: docs/code-notes/decoder.md#TestA3MoEExclusionIsMeasured.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
//	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoEExclusionIsMeasured -v
func TestA3MoEExclusionIsMeasured(t *testing.T) {
	path := assetPath(t, "GOINFER_MELLUM_CKPT")
	requireHeavyModel(t)
	K := 2048
	if v := os.Getenv("GOINFER_MELLUM_K"); v != "" {
		if _, err := fmt.Sscanf(v, "%d", &K); err != nil {
			t.Fatalf("GOINFER_MELLUM_K=%q: %v", v, err)
		}
	}
	// Quant is an axis: the attention swap is quant-independent, but its share of prefill is not (int4's faster
	// weight matmul raises attention's fraction and so the speedup). Label which quant you quote: int4 is the
	// operator-facing number (docs/completed/mellum2-resident.md), int8int8 is the one comparable to earlier runs.
	quant := os.Getenv("GOINFER_BENCH_QUANT")
	if quant == "" {
		quant = "int8int8"
	}
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("%s is not a MoE — this test measures the MoE exclusion", path)
	}
	if !m.canBatchN(K) {
		t.Fatalf("canBatchN(%d) = false", K)
	}
	// Varied ids: on a MoE the prompt is the routing, and a constant-id prompt collapses the top-k to one
	// near-identical set, understating the near-tie flips this test exists to provoke.
	vocab := m.w.arch.VocabSize
	ids := make([]int, K)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	run := func(probe bool) ([]float32, time.Duration) {
		t.Helper()
		setKnob(t, m, knobCPUFastAttention, map[bool]string{true: "1", false: "0"}[probe])
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
		"A3-MoE-exclusion: quant=%s K=%d cosine=%.9f maxAbs=%.4g | acc64 %.1fs -> f32 %.1fs (%.2fx)\n"+
			"  dense ships at cosine 0.9976 behind this same flag\n",
		quant, K, cos, maxAbs, tBase.Seconds(), tFast.Seconds(), tBase.Seconds()/tFast.Seconds())
}
