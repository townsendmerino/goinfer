//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

// DOES A ROUTER FLIP EXPLAIN THE A3 MoE DIVERGENCE? — the mechanism experiment for the
// --cpu-fast-attention MoE exclusion.
//
// The state of the argument before this test. forwardn.go excludes MoE from A3/G24
// unconditionally, on a stated mechanism: "an f32 QK reassociation flips a top-k expert at a
// near-tie and cascades". TestA3MoEExclusionIsMeasured now puts the SYMPTOM at cosine
// 0.999787-0.999788 — an order of magnitude tighter than the 0.9976 dense already ships behind
// the same flag. But a cosine cannot tell a routing flip from ordinary numeric drift, and the
// guard is a claim about routing specifically. This measures the mechanism.
//
// THE DECOMPOSITION, which is what makes it decisive. Three arms on one prompt:
//
//	A  acc64 attention, natural routing          — the baseline; its routing is recorded
//	B  f32 attention, natural routing            — total divergence (what the flag would ship)
//	C  f32 attention, A's routing REPLAYED       — divergence with the routing term REMOVED
//
// Arm C uses the existing moeSelOverride seam (built for E2's higher-precision replay). So:
//
//	cos(A,C) ~= cos(A,B)  =>  routing flips contribute ~nothing; the divergence is ordinary
//	                          drift and the guard is aimed at something that is not happening.
//	cos(A,C) >>  cos(A,B)  =>  routing flips ARE the dominant term and the guard has its case.
//
// THE PREDICTION, stated before the run so it can fail: given a symptom this small, arm C should
// land close to arm B. If instead C is far cleaner than B, the exclusion is vindicated on its own
// mechanism and this test says so — which is a real finding, not a failed test.
//
// WHAT A FLIP COSTS is reported too, because "a flip happened" is not "a flip mattered".
// norm_topk_prob renormalizes over the kept k, so swapping the k-th expert perturbs the sum by at
// most its weight; the SMALLEST top-k weight therefore BOUNDS one flip's contribution. That is a
// bound, not a margin — the dropped expert's own score is not in the trace — and it is reported
// as one.
//
// DIAGNOSTIC, NOT A GATE. It asserts only what must hold under either story (the arms are
// comparable, the seams actually fired). The numbers are for a recorded decision.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DIAG=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
//	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoERouteFlips -v -timeout 60m
func TestA3MoERouteFlips(t *testing.T) {
	if os.Getenv("GOINFER_DIAG") == "" {
		t.Skip("DIAGNOSTIC (set GOINFER_DIAG=1): prints evidence for a judgement, asserts only " +
			"what holds under either story. Not a gate.")
	}
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
		t.Fatalf("%s is not a MoE", path)
	}
	// Varied ids: a constant-id prompt collapses the top-k to one near-identical set and
	// cannot produce the near-tie flip this test exists to find.
	vocab := m.w.arch.VocabSize
	ids := make([]int, K)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	// trace=true records this arm's routing; replay!=nil forces it instead.
	run := func(probe, trace bool, replayIdx [][]int, replayWts [][]float32) ([]float32, [][]int, [][]float32) {
		t.Helper()
		moeFastAttnProbe = probe
		t.Setenv("GOINFER_CPU_FAST_ATTENTION", map[bool]string{true: "1", false: "0"}[probe])
		if trace {
			moeSelTrace = make([][]int, 0, 1<<14)
			moeWtsTrace = make([][]float32, 0, 1<<14)
		}
		if replayIdx != nil {
			moeSelOverride, moeWtsOverride, moeOverridePos = replayIdx, replayWts, 0
		}
		out, err := m.forwardLayersN(deadlineCtx(t), ids, m.NewCache(K+8), cpuFastAttention())
		moeFastAttnProbe = false
		moeSelOverride, moeWtsOverride = nil, nil
		gotIdx, gotWts := moeSelTrace, moeWtsTrace
		moeSelTrace, moeWtsTrace = nil, nil
		if err != nil {
			t.Fatalf("probe=%v replay=%v: %v", probe, replayIdx != nil, err)
		}
		return out, gotIdx, gotWts
	}

	cos := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			x, y := float64(a[i]), float64(b[i])
			dot += x * y
			na += x * x
			nb += y * y
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb))
	}

	fmt.Fprintf(os.Stderr, "A3-routeflip: K=%d arm A (acc64, tracing routing)\n", K)
	outA, idxA, wtsA := run(false, true, nil, nil)
	if len(idxA) == 0 {
		t.Fatalf("arm A recorded no routing — the moeSelTrace seam did not fire, and every " +
			"number below would be about a test that did not run")
	}
	fmt.Fprintf(os.Stderr, "A3-routeflip: %d moeMLP calls traced; arm B (f32, natural routing)\n", len(idxA))
	outB, idxB, _ := run(true, true, nil, nil)
	fmt.Fprintf(os.Stderr, "A3-routeflip: arm C (f32, arm A's routing REPLAYED)\n")
	outC, _, _ := run(true, false, idxA, wtsA)

	if len(idxB) != len(idxA) {
		t.Fatalf("arms traced different call counts (%d vs %d) — not comparable", len(idxA), len(idxB))
	}

	// Flip statistics: a call is flipped if its top-k SET differs (order within the set is not
	// a routing difference — the same experts with the same weights produce the same sum).
	set := func(v []int) map[int]bool {
		s := make(map[int]bool, len(v))
		for _, e := range v {
			s[e] = true
		}
		return s
	}
	flips, bounds := 0, []float64{}
	for i := range idxA {
		a, b := set(idxA[i]), set(idxB[i])
		differs := len(a) != len(b)
		for e := range a {
			if !b[e] {
				differs = true
			}
		}
		if !differs {
			continue
		}
		flips++
		wmin := math.Inf(1)
		for _, w := range wtsA[i] {
			if float64(w) < wmin {
				wmin = float64(w)
			}
		}
		bounds = append(bounds, wmin)
	}
	sort.Float64s(bounds)
	pct := func(p float64) float64 {
		if len(bounds) == 0 {
			return 0
		}
		return bounds[min(int(p*float64(len(bounds))), len(bounds)-1)]
	}

	cAB, cAC := cos(outA, outB), cos(outA, outC)
	// (1-cos) is the divergence; how much of it does removing the routing flips remove?
	explained := 0.0
	if 1-cAB > 0 {
		explained = 100 * (1 - (1-cAC)/(1-cAB))
	}
	fmt.Fprintf(os.Stderr,
		"A3-MoE-routeflip: K=%d calls=%d flipped=%d (%.3f%%)\n"+
			"  cos(A,B) natural routing = %.9f   [total divergence]\n"+
			"  cos(A,C) routing replayed = %.9f  [routing term removed]\n"+
			"  => routing flips explain %.1f%% of the divergence\n"+
			"  flip impact BOUND (smallest kept top-k weight): p50=%.4g p90=%.4g max=%.4g\n",
		K, len(idxA), flips, 100*float64(flips)/float64(len(idxA)),
		cAB, cAC, explained, pct(0.5), pct(0.9), pct(1.0))
}
