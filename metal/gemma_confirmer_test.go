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

	// Inject into Metal and run L1 attention over the matched residual + K/V.
	got := r.attnConfirmForTest(residIntoL1, kL1, vL1, injectLayer, probe)

	var dot, na, nb float64
	for i := range ctxL1 {
		dot += float64(got[i]) * float64(ctxL1[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ctxL1[i]) * float64(ctxL1[i])
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
	t.Logf("MATCHED-INPUT L%d context: cos(Metal, target)=%.4f  |Metal|=%.2f |target|=%.2f", injectLayer, cos, math.Sqrt(na), math.Sqrt(nb))
	if cos > 0.99 && math.Sqrt(na) < 1.3*math.Sqrt(nb) {
		t.Logf("VERDICT: MATCH on injected input — Metal's L1 attention block is FAITHFUL. The crater is " +
			"accumulated f16/precision DRIFT in the residual+KV feeding attention (fix: f32 KV / f32 attn-accumulate).")
	} else {
		t.Logf("VERDICT: STILL DIVERGES on matched input (cos %.4f, |Metal|/|target|=%.2f) — Metal's per-layer "+
			"attention BLOCK has a real bug (norm/QKV/RoPE/softmax), independent of drift.", cos, math.Sqrt(na)/math.Sqrt(nb))
	}
}
