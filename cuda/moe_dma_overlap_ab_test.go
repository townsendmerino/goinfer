//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEDMAOverlapAB is the pre-registered A/B for the C′ compute/DMA overlap
// (docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md): one loaded real 26B, the
// two arms flipped between generations on the SAME model and the SAME input sequence, alternating
// ABBA so drift cannot masquerade as an effect. The do-nothing arm is the draining path the model
// shipped with. Bit-identity is asserted (full logits, Float32bits, every token of a generation);
// the perf verdict is LOGGED against the registered bar, not asserted, so a slow box does not turn
// a correctness test red.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestMoEDMAOverlapAB -v -timeout 30m
func TestMoEDMAOverlapAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — real 26B decode")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the 26B with C′ on")
	}
	r := rf.(*cudaResident)
	if !r.overlap {
		t.Fatal("overlap is OFF at load — the A/B needs its resources built")
	}
	t.Logf("loaded 26B .giw in %s, cacheSlots=%d", time.Since(t0).Round(time.Second), r.cacheSlots)

	_, _, _, _, _, _, vocab := m.Dims()
	const nTok = 48
	ids := make([]int, nTok)
	var s uint32 = 24601
	for i := range ids {
		s = s*1664525 + 1013904223
		ids[i] = int(s>>8) % (vocab - 1)
	}
	gen := func(on bool) (time.Duration, []int) {
		if !r.SetOverlapForTest(on) {
			t.Fatal("SetOverlapForTest: no overlap resources")
		}
		r.Reset()
		out := make([]int, nTok)
		start := time.Now()
		for i, id := range ids {
			nextID, err := r.ForwardArgmax(m.EmbedResidentForTest(id), i)
			if err != nil {
				t.Fatalf("overlap=%v step %d: %v", on, i, err)
			}
			out[i] = nextID
		}
		return time.Since(start), out
	}
	logits := func(on bool) [][]uint32 {
		if !r.SetOverlapForTest(on) {
			t.Fatal("SetOverlapForTest: no overlap resources")
		}
		r.Reset()
		out := make([][]uint32, nTok)
		for i, id := range ids {
			lg, err := r.Forward(m.EmbedResidentForTest(id), i)
			if err != nil {
				t.Fatalf("overlap=%v Forward %d: %v", on, i, err)
			}
			bits := make([]uint32, len(lg))
			for k, v := range lg {
				bits[k] = math.Float32bits(v)
			}
			out[i] = bits
		}
		return out
	}

	// Warm-up, both arms, discarded (first-touch JIT, cache fill).
	gen(false)
	gen(true)

	// Bit-identity: full logits, every token, both arms on the same inputs from the same cache state.
	off, on := logits(false), logits(true)
	for i := range off {
		for k := range off[i] {
			if off[i][k] != on[i][k] {
				t.Fatalf("NOT bit-identical: token %d logit %d: off=%08x on=%08x", i, k, off[i][k], on[i][k])
			}
		}
	}
	t.Logf("bit-identical: %d tokens x %d logits, Float32bits equal", nTok, vocab)

	// Paired perf: ABBA alternation, pairs of (off, on) generations.
	const pairs = 8
	ratios := make([]float64, 0, pairs)
	var sumOff, sumOn time.Duration
	for p := 0; p < pairs; p++ {
		var dOff, dOn time.Duration
		var aOff, aOn []int
		if p%2 == 0 {
			dOff, aOff = gen(false)
			dOn, aOn = gen(true)
		} else {
			dOn, aOn = gen(true)
			dOff, aOff = gen(false)
		}
		for i := range aOff {
			if aOff[i] != aOn[i] {
				t.Fatalf("pair %d: argmax differs at token %d (off=%d on=%d)", p, i, aOff[i], aOn[i])
			}
		}
		ratio := dOff.Seconds() / dOn.Seconds()
		ratios = append(ratios, ratio)
		sumOff += dOff
		sumOn += dOn
		t.Logf("pair %d: off %.2f tok/s | on %.2f tok/s | on/off speedup %.3fx",
			p, nTok/dOff.Seconds(), nTok/dOn.Seconds(), ratio)
	}
	sort.Float64s(ratios)
	median := ratios[len(ratios)/2]
	if len(ratios)%2 == 0 {
		median = (ratios[len(ratios)/2-1] + ratios[len(ratios)/2]) / 2
	}
	verdict := "KILL (<1.5%)"
	switch {
	case median >= 1.03:
		verdict = "SHIP (>=3%)"
	case median >= 1.015:
		verdict = "PARK (1.5-3%)"
	}
	if median >= 1.03*0.95 && median < 1.03 || median >= 1.015*0.95 && median < 1.015 {
		verdict += " — AMBIGUOUS (within 5% of a threshold): parked"
	}
	t.Logf("paired median speedup %.3fx (min %.3f, max %.3f) over %d pairs; pooled off %.2f vs on %.2f tok/s — %s",
		median, ratios[0], ratios[len(ratios)-1], pairs,
		float64(pairs*nTok)/sumOff.Seconds(), float64(pairs*nTok)/sumOn.Seconds(), verdict)
}
