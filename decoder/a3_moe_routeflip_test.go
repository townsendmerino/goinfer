//go:build goinfer_testhooks

package decoder

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

// Does a router flip explain the A3 MoE divergence? The mechanism experiment for the f32-attention flag on a MoE:
// a cosine cannot tell a routing flip from ordinary numeric drift, so three arms on one prompt separate them.
//
//	A  acc64 attention, natural routing          (baseline; its routing is recorded)
//	B  f32 attention, natural routing            (total divergence)
//	C  f32 attention, A's routing REPLAYED       (divergence with the routing term removed)
//
// Arm C uses the moeSelOverride seam. cos(A,C) ~= cos(A,B) means flips contribute ~nothing; cos(A,C) >> cos(A,B)
// means routing flips dominate. It also reports what a flip costs: norm_topk_prob renormalizes over the kept
// k, so the smallest top-k weight bounds one flip's contribution (a bound, not a margin; the dropped expert's
// score is not in the trace).
//
// DIAGNOSTIC, NOT A GATE: it asserts only what holds under either story (the arms are comparable, the seams
// fired). The prediction and the numbers: docs/code-notes/decoder.md#TestA3MoERouteFlips.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DIAG=1 GOINFER_MELLUM_CKPT=... GOINFER_MELLUM_K=2048 \
//	go test -tags goinfer_testhooks ./decoder/ -run TestA3MoERouteFlips -v -timeout 60m
func TestA3MoERouteFlips(t *testing.T) {
	if os.Getenv("GOINFER_DIAG") == "" {
		t.Skip("DIAGNOSTIC (set GOINFER_DIAG=1): prints evidence for a judgement, asserts only " +
			"what holds under either story. Not a gate.")
	}
	path := assetPath(t, "GOINFER_MELLUM_CKPT")
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
	// Varied ids: a constant-id prompt collapses the top-k to one near-identical set and cannot produce
	// the near-tie flip this test exists to find.
	vocab := m.w.arch.VocabSize
	ids := make([]int, K)
	for i := range ids {
		ids[i] = (i*131 + 7) % vocab
	}

	// trace=true records this arm's routing; replay!=nil forces it instead.
	run := func(probe, trace bool, replayIdx [][]int, replayWts [][]float32) ([]float32, [][]int, [][]float32) {
		t.Helper()
		setKnob(t, m, knobCPUFastAttention, map[bool]string{true: "1", false: "0"}[probe])
		if trace {
			moeSelTrace = make([][]int, 0, 1<<14)
			moeWtsTrace = make([][]float32, 0, 1<<14)
		}
		if replayIdx != nil {
			moeSelOverride, moeWtsOverride, moeOverridePos = replayIdx, replayWts, 0
		}
		out, err := m.forwardLayersN(deadlineCtx(t), ids, m.NewCache(K+8), cpuFastAttention())
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
