//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestBatchedVsDecodeGap documents a latent numerics gap surfaced by D1 (speculative decode). The
// batched forward (PrefillLast/PrefillLastN) and the decode-step forward (Forward) are BYTE-IDENTICAL
// at startPos=0 — the only regime production ever used PrefillLast in (whole-prompt prefill), and what
// TestPrefillLast_e2e gates — but diverge by ~1e-6 at startPos>0 (appending to an existing KV cache).
// Spec-decode's verify calls the batched path at startPos>0, so this gap is what makes the batched
// verify NOT bit-identical to the decode path (it IS self-consistent / lossless w.r.t. the batched
// forward — see TestSpecDecode). rope_kv and rope_kv_batched are mathematically identical and K-after-
// rope is byte-identical (the KV gate), so the cause is elsewhere in attention-over-primed-KV; unpinned.
//
// This test ASSERTS the byte-identity at startPos=0 (a real invariant) and REPORTS the startPos>0 gap
// (informational — it does not fail, so it stands as a live repro for the fix). Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestBatchedVsDecodeGap -v
func TestBatchedVsDecodeGap(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(id int) []float32 { return mc.EmbedResidentForTest(((id % vocab) + vocab) % vocab) }

	// compare Forward(tok, startPos) vs PrefillLast([tok], startPos), both over the same primed KV.
	compare := func(startPos int) (mism int, maxΔ float32) {
		if startPos > 0 {
			se := make([][]float32, startPos)
			for i := range se {
				se[i] = emb(i*7 + 3)
			}
			if _, e := rf.PrefillLast(se, 0); e != nil {
				t.Fatalf("prime: %v", e)
			}
		}
		f, e := rf.Forward(emb(555), startPos)
		if e != nil {
			t.Fatalf("Forward: %v", e)
		}
		ff := append([]float32(nil), f...)
		if startPos > 0 {
			se := make([][]float32, startPos)
			for i := range se {
				se[i] = emb(i*7 + 3)
			}
			if _, e := rf.PrefillLast(se, 0); e != nil {
				t.Fatalf("re-prime: %v", e)
			}
		}
		pl, e := rf.PrefillLast([][]float32{emb(555)}, startPos)
		if e != nil {
			t.Fatalf("PrefillLast: %v", e)
		}
		for j := range ff {
			if d := ff[j] - pl[j]; d != 0 {
				if d < 0 {
					d = -d
				}
				if d > maxΔ {
					maxΔ = d
				}
				mism++
			}
		}
		return
	}

	mism0, max0 := compare(0)
	t.Logf("startPos=0:  mism=%d/%d maxΔ=%g", mism0, vocab, max0)
	if mism0 != 0 {
		t.Fatalf("INVARIANT BROKEN: batched vs decode forward differ at startPos=0 (mism=%d) — a regression", mism0)
	}
	mism32, max32 := compare(32)
	t.Logf("startPos=32: mism=%d/%d maxΔ=%g  (the latent gap — informational, blocks bit-identical spec-decode)", mism32, vocab, max32)
}
