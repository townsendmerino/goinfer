//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
)

// TestFlintResidentParityMetal is the Metal resident + paged-MoE parity gate for
// Recursive's Flint (docs/completed/task-flint-2026-10.md F2/F3):
//   - testdata/flint-tiny (6 layers: 2 dense full/NoPE + 3 MoE sliding/RoPE + 1 MoE
//     full/NoPE, sliding_window=8, learned per-head attention sinks, QK-norm, top-2 of 8
//     sigmoid-normalized router + ungated shared expert) goes resident on Metal at int4;
//   - sequential decode across 12 positions (past the 8-token sliding window) agrees with
//     the CPU int4 forward (cosine >= 0.999) and with Metal's batched prefill
//     (cosine >= 0.999);
//   - a .giw bundle loaded with MoECacheExperts=true, MoECacheSlots=3 (< 8 experts,
//     forcing LRU eviction) produces bit-identical decode logits to the all-resident run.
func TestFlintResidentParityMetal(t *testing.T) {
	const ckpt = "../../testdata/flint-tiny"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	if !mRes.HasAttnSinksResident() {
		t.Fatalf("flint-tiny HasAttnSinksResident() = false, want true")
	}
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("flint-tiny did not go Metal-resident (decode path %q; decline: %s)",
			mRes.DecodePath(), mRes.ResidentDecline())
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	const ntok = 12 // > sliding_window=8
	_, _, _, _, _, _, vocab := mRes.Dims()
	prompt := make([]int, ntok)
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
	}

	rf.Reset()
	cache := mCPU.NewCache(ntok)
	worstCos := 1.0
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("metal forward[%d]: %v", i, err)
		}
		for j, v := range lr {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("metal forward[%d]: logit[%d] = %v", i, j, v)
			}
		}
		worstCos = math.Min(worstCos, cosF(lr, lc))
	}
	t.Logf("flint-tiny Metal decode (%d positions across window=8) vs CPU int4: worst cosine %.6f", ntok, worstCos)
	if worstCos < 0.995 {
		t.Errorf("Metal decode vs CPU int4 worst cosine %.6f < 0.995", worstCos)
	}

	// Write a metal-target .giw and verify MoECacheExperts=true (slots=3 < nE=8) is bit-identical to unpaged.
	giwPath := filepath.Join(t.TempDir(), "flint-tiny.int4.metal.giw")
	if err := prequant.Transcode(t.Context(), ckpt, giwPath, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("prequant.Transcode: %v", err)
	}

	runGIW := func(paged bool) [][]float32 {
		opts := decoder.Options{Backend: "metal", MoECacheExperts: paged}
		if paged {
			opts.MoECacheSlots = 3
		}
		m, err := decoder.Load(giwPath, opts)
		if err != nil {
			t.Fatalf("Load(.giw, paged=%v): %v", paged, err)
		}
		defer m.Close()
		r := m.ResidentForwardForTest()
		if r == nil {
			t.Fatalf(".giw (paged=%v) did not go resident: %s", paged, m.ResidentDecline())
		}
		r.Reset()
		out := make([][]float32, ntok)
		for i, tok := range prompt {
			lg, err := r.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf(".giw forward[%d] (paged=%v): %v", i, paged, err)
			}
			out[i] = append([]float32(nil), lg...)
		}
		return out
	}

	unpaged, paged := runGIW(false), runGIW(true)
	for i := range unpaged {
		for j := range unpaged[i] {
			if unpaged[i][j] != paged[i][j] {
				t.Fatalf("step %d logit %d: paged %.9g != unpaged %.9g", i, j, paged[i][j], unpaged[i][j])
			}
		}
	}
	t.Logf("flint-tiny Metal .giw MoE pager (slots=3 vs all 8 experts): %d steps bit-identical", ntok)
}
