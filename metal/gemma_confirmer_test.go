//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemmaConfirmer_MatchedInput is the Metal half of the matched-input confirmer (the CUDA box's
// TestGemmaConfirmerReference assembles the reference; this injects it). It settles the last cut on
// Gemma's dormant crater: int4-direct proved the WEIGHTS are not the cause (L0=1.0 but L1 still
// craters to 0.640 with byte-identical weights), leaving two candidates —
//
//	Metal's L1 context MATCHES the target on injected input -> the crater is accumulated
//	  f16/precision DRIFT in the residual+KV feeding attention (fix: f32 KV / f32 attn-accumulate).
//	Metal's L1 context STILL inflates on injected input     -> Metal's per-layer attention BLOCK
//	  (norm/QKV/RoPE/softmax/accumulate) has a real bug, independent of drift.
//
// The reference is goinfer's own CPU-int4 state (== CUDA-int4 to decimals) over the byte-identical
// Q4_K_M gguf, assembled from the same seams the CUDA box used: residual entering L1 =
// ForwardCapture(...,[0]); post-RoPE K / raw V = cache.LayerKVForTest(1); target = ForwardSubCapture
// -> ctx[1]. attnConfirmForTest sets Metal's r.x + r.kc[1]/r.vc[1] and runs L1's attention.
func TestGemmaConfirmer_MatchedInput(t *testing.T) {
	if testing.Short() {
		t.Skip("loads a real model")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	const probe, injectLayer = 5, 1
	prompt := []int{2, 669, 5279, 529, 7001, 563}

	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if !r.sandwich {
		t.Fatal("gemma3 resident not sandwich")
	}

	// Reference bundle from goinfer CPU-int4 (same assembly as TestGemmaConfirmerReference).
	capCache := m.NewCache(len(prompt) + 1)
	var residIntoL1 []float32
	for i, tok := range prompt {
		_, hid, err := m.ForwardCapture(tok, capCache, []int{injectLayer - 1})
		if err != nil {
			t.Fatalf("ForwardCapture pos %d: %v", i, err)
		}
		if i == probe {
			residIntoL1 = append([]float32(nil), hid[0]...)
		}
	}
	kL1, vL1, kvBase := capCache.LayerKVForTest(injectLayer)

	subCache := m.NewCache(len(prompt) + 1)
	var ctxL1 []float32
	for i, tok := range prompt {
		if i == probe {
			_, _, ctx, err := m.ForwardSubCapture(tok, subCache)
			if err != nil {
				t.Fatalf("ForwardSubCapture: %v", err)
			}
			ctxL1 = append([]float32(nil), ctx[injectLayer]...)
		} else if _, err := m.ForwardForTest(tok, subCache); err != nil {
			t.Fatalf("warm pos %d: %v", i, err)
		}
	}
	l2 := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s)
	}
	t.Logf("reference: resid L2=%.2f | K L2=%.2f (base %d) | V L2=%.2f | target ctx L2=%.2f",
		l2(residIntoL1), l2(kL1), kvBase, l2(vL1), l2(ctxL1))

	cosTo := func(got []float32) (float64, float64) {
		var dot, na, nb float64
		for i := range ctxL1 {
			dot += float64(got[i]) * float64(ctxL1[i])
			na += float64(got[i]) * float64(got[i])
			nb += float64(ctxL1[i]) * float64(ctxL1[i])
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), math.Sqrt(na)
	}

	// (A) Matched residual + matched K/V — isolates the attention BLOCK itself.
	cA, nA := cosTo(r.attnConfirmForTest(residIntoL1, kL1, vL1, injectLayer, probe, true))
	t.Logf("(A) matched resid + matched KV: cos=%.4f |Metal|=%.2f |target|=%.2f", cA, nA, l2(ctxL1))

	// (B) Matched residual + Metal's OWN walked f16 KV history — isolates the KV drift. A Metal
	// walk populates r.kc[1] with Metal's drifted f16 KV for positions 0..4; the confirmer then
	// computes pos-5 K/V from the matched residual and attends over that mixed history.
	for i := 0; i < probe; i++ {
		r.forwardTrunkForTest(m.EmbedResidentForTest(prompt[i]), i, r.nL)
	}
	cB, nB := cosTo(r.attnConfirmForTest(residIntoL1, nil, nil, injectLayer, probe, false))
	t.Logf("(B) matched resid + Metal WALKED f16 KV: cos=%.4f |Metal|=%.2f |target|=%.2f", cB, nB, l2(ctxL1))

	if cA > 0.99 {
		t.Logf("VERDICT: attention BLOCK is FAITHFUL (A=%.4f). The full-forward crater is INPUT drift.", cA)
		if cB < 0.95 {
			t.Logf("  -> and it's the KV: matched residual + Metal's walked f16 KV alone craters (B=%.4f). "+
				"The f16 KV cache is the drift source — fix is f32 KV for Gemma.", cB)
		} else {
			t.Logf("  -> KV history is NOT the dominant source (B=%.4f); the residual entering L1 is.", cB)
		}
	} else {
		t.Errorf("attention block diverges on matched input (A=%.4f) — unexpected after the QK-norm fix", cA)
	}
}
