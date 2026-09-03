//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGptOssExpertCacheAB is the assertion the 2026-08-31 index-space fix was owed.
//
// THE BUG IT DISCRIMINATES. gpt-oss's gate‖up bias table is [nExpert][2*I], uploaded once for
// all experts and never moved. expIdx() returns SLOT ids when expert caching is on and EXPERT
// ids otherwise, so binding it to index that table selected the wrong expert's biases whenever
// caching was enabled — silently, with plausible logits. The two index spaces COINCIDE with
// caching off, which is why nothing caught it.
//
// THE INVARIANT. Expert caching is a RESIDENCY strategy, not an arithmetic one: which slot a
// weight was streamed into cannot change what the model computes. So the cached and uncached
// runs must agree BIT FOR BIT, and any difference at all is a bug rather than a tolerance
// question. This asserts equality, not a cosine — a near-match would be exactly the symptom of
// the wrong bias row, and a loose bar would let it through.
//
// It is also the only form of this test that can discriminate: a device-free unit test cannot,
// because gpu.Buffer's fields are unexported, so two zero buffers compare equal and such a test
// passes on the buggy code too.
func TestGptOssExpertCacheAB(t *testing.T) {
	if !decoder.ResidentBackendFeatures("cuda")[decoder.FeatAttnSink] {
		t.Skip("cuda does not declare FeatAttnSink/FeatOutBias — gpt-oss cannot go resident here yet")
	}
	path := "../decoder/testdata/gptoss_tiny.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gpt-oss fixture at %s — run scripts/gptoss_tiny_golden.py", path)
	}
	seed := []int{3, 14, 7, 42, 1, 99, 5, 60} // gptoss_tiny_golden.json input_ids
	const steps = 8

	// topK+1 for the fixture (nE=4, topK=2): honoured, below nE, and forces at least one
	// eviction so slot ≠ expert id.
	const wantSlots = 3
	run := func(cache bool) [][]float32 {
		opts := decoder.Options{Backend: "cuda", Quant: "int4", MoECacheExperts: cache}
		if cache {
			// G-07: topK+1 = 3, not 2. The fixture is nE=4/topK=2, and a request of 2 was NOT
			// honoured — `req > topK` was false, so cacheSlots stayed at min(8·topK, nE) = 4,
			// i.e. one permanent slot per expert. This gate's whole premise is that slot ≠
			// expert id, and it was getting the identity mapping; it discriminated on the
			// 2026-08-31 run by routing luck. 3 is honoured, is below nE=4, and forces at
			// least one eviction.
			opts.MoECacheSlots = wantSlots
		}
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("load (cache=%v): %v", cache, err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Skipf("gpt-oss not resident on cuda (%s) — declare FeatAttnSink+FeatOutBias to run this", m.ResidentDecline())
		}
		// G-07: ASSERT THE EFFECTIVE SLOT COUNT. Nothing did, and the request was being
		// silently floored — with topK=2 the old `req > topK` was false for a request of 2, so
		// cacheSlots stayed at min(8·topK, nE) = 4 = nE: one permanent slot per expert, and
		// slot ≠ expert only by first-admit order. This gate's premise is that the two index
		// spaces diverge, so the premise has to be checked rather than requested.
		if cache {
			if cr, ok := rf.(interface{ CacheSlotsForTest() int }); ok {
				if got := cr.CacheSlotsForTest(); got != wantSlots {
					t.Fatalf("cacheSlots = %d, requested %d — the request was not honoured, so "+
						"slot and expert id may still coincide and this A/B proves nothing (G-07)",
						got, wantSlots)
				}
			} else {
				t.Fatal("resident does not expose CacheSlotsForTest; the slot count cannot be " +
					"asserted and this gate would run on an unknown configuration (G-07)")
			}
		}
		out := make([][]float32, 0, steps)
		for i, tok := range seed[:steps] {
			l, err := rf.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward %d (cache=%v): %v", i, cache, err)
			}
			out = append(out, append([]float32(nil), l...))
		}
		return out
	}

	off, on := run(false), run(true)
	for i := range off {
		if len(off[i]) != len(on[i]) {
			t.Fatalf("step %d: length %d vs %d", i, len(off[i]), len(on[i]))
		}
		for j := range off[i] {
			if off[i][j] != on[i][j] {
				t.Fatalf("step %d logit %d: cached %.9g != uncached %.9g — expert caching changed "+
					"the ARITHMETIC, which it must never do. The 2026-08-31 defect was gpt-oss "+
					"indexing its per-expert bias table by SLOT id under caching; this is what "+
					"that looks like.", i, j, on[i][j], off[i][j])
			}
		}
	}
	t.Logf("gpt-oss expert cache A/B: %d steps bit-identical with slots=%d (asserted) vs all-resident", steps, wantSlots)
}
