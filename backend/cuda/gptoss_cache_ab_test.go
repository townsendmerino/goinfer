//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGptOssExpertCacheAB pins that expert caching does not change which expert's biases are used.
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
	path := "../../decoder/testdata/gptoss_tiny.gguf"
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
			// topK+1 = 3, not 2. The fixture is nE=4/topK=2, and a request of 2 is NOT honoured — `req > topK`
			// is false, so cacheSlots stays at min(8·topK, nE) = 4, one permanent slot per expert, and the
			// identity mapping hides the bug this gate exists for (slot ≠ expert id). 3 is honoured, is below
			// nE=4, and forces at least one eviction.
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
		// ASSERT THE EFFECTIVE SLOT COUNT: this gate's premise is that the two index spaces diverge, so the
		// premise has to be checked rather than requested (a request can be silently floored; see wantSlots).
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

// TestFlintResidentParityCUDA is the CUDA resident and expert-cache parity gate for Flint:
//   - testdata/flint-tiny (6 layers: 2 dense full/NoPE + 3 MoE sliding/RoPE + 1 MoE
//     full/NoPE, sliding_window=8, learned per-head attention sinks, QK-norm, top-2 of 8
//     sigmoid-normalized router + ungated shared expert) goes resident on CUDA at int4;
//   - sequential decode across 12 positions (past the 8-token sliding window) agrees with
//     the CPU int4 forward and with CUDA's batched prefill;
//   - MoECacheExperts=true with MoECacheSlots=3 (< 8 experts, forcing LRU eviction and
//     asserting CacheSlotsForTest()==3) produces bit-identical decode logits to the
//     all-resident run.
func TestFlintResidentParityCUDA(t *testing.T) {
	const ckpt = "../../testdata/flint-tiny"
	requireDeviceAndFixture(t, ckpt)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()

	if !mRes.HasAttnSinksResident() {
		t.Fatalf("flint-tiny HasAttnSinksResident() = false, want true")
	}
	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("flint-tiny did not go CUDA-resident (%T) — decline: %s", rf, mRes.ResidentDecline())
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	const ntok = 12 // > sliding_window=8
	_, _, _, _, _, _, vocab := mCPU.Dims()
	prompt := make([]int, ntok)
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
	}

	rf.Reset()
	cache := mCPU.NewCache(ntok)
	worstCos, worstRel := 1.0, 0.0
	var allResLogits [][]float32
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lr, lc)
		worstCos = math.Min(worstCos, cos)
		worstRel = math.Max(worstRel, relL2(lr, lc))
		allResLogits = append(allResLogits, append([]float32(nil), lr...))
	}
	t.Logf("flint-tiny CUDA decode (%d positions across window=8) vs CPU int4: worst cosine %.6f, worst relL2 %.4e",
		ntok, worstCos, worstRel)
	if worstCos < 0.995 || worstRel > 0.10 {
		t.Errorf("CUDA decode vs CPU int4: worst cosine %.6f (want >= 0.995), worst relL2 %.4f (want <= 0.10)",
			worstCos, worstRel)
	}

	rf.Reset()
	embs := make([][]float32, ntok)
	for i, tok := range prompt {
		embs[i] = mRes.EmbedResidentForTest(tok)
	}
	gotPre, err := cr.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	preCos, _ := cosF32(gotPre, allResLogits[ntok-1])
	preRel := relL2(gotPre, allResLogits[ntok-1])
	t.Logf("flint-tiny CUDA batched prefill vs sequential decode: cosine %.6f relL2 %.4e", preCos, preRel)
	if preCos < 0.995 || preRel > 0.10 {
		t.Errorf("CUDA batched prefill vs sequential decode: cosine %.6f relL2 %.4f", preCos, preRel)
	}

	const wantSlots = 3
	mCached, err := decoder.Load(ckpt, decoder.Options{
		Backend:         "cuda",
		Quant:           "int4",
		MoECacheExperts: true,
		MoECacheSlots:   wantSlots,
	})
	if err != nil {
		t.Fatalf("load cuda (MoECacheExperts): %v", err)
	}
	defer mCached.Close()
	rfCached := mCached.ResidentForwardForTest()
	crCached, ok := rfCached.(*cudaResident)
	if !ok {
		t.Fatalf("flint-tiny (MoECacheExperts) did not go CUDA-resident: %s", mCached.ResidentDecline())
	}
	if got := crCached.CacheSlotsForTest(); got != wantSlots {
		t.Fatalf("CacheSlotsForTest() = %d, want %d", got, wantSlots)
	}
	rfCached.Reset()
	for i, tok := range prompt {
		lr, err := rfCached.Forward(mCached.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cached forward[%d]: %v", i, err)
		}
		for j := range lr {
			if lr[j] != allResLogits[i][j] {
				t.Fatalf("step %d logit %d: cached %.9g != all-resident %.9g", i, j, lr[j], allResLogits[i][j])
			}
		}
	}
	t.Logf("flint-tiny CUDA expert cache (slots=%d vs all 8 experts): %d steps bit-identical", wantSlots, ntok)
}
