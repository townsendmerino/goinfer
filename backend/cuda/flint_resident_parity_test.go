//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestFlintResidentParityCUDA is the CUDA resident + C′ expert-cache parity gate for
// Recursive's Flint (docs/completed/task-flint-2026-10.md F2/F3):
//   - testdata/flint-tiny (6 layers: 2 dense full/NoPE + 3 MoE sliding/RoPE + 1 MoE
//     full/NoPE, sliding_window=8, learned per-head attention sinks, QK-norm, top-2 of 8
//     sigmoid-normalized router + ungated shared expert) goes resident on CUDA at int4;
//   - sequential decode across 12 positions (past the 8-token sliding window) agrees with
//     the CPU int4 forward (cosine >= 0.995, relL2 <= 0.10) and with CUDA's batched
//     prefill (cosine >= 0.995, relL2 <= 0.10);
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

	// C′ expert cache A/B: slots=3 (< nE=8, > topK=2) must be bit-identical to all-resident.
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
	t.Logf("flint-tiny CUDA C′ expert cache (slots=%d vs all 8 experts): %d steps bit-identical", wantSlots, ntok)
}
