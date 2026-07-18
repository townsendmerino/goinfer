package decoder

import (
	"math"
	"os"
	"testing"
)

// TestGemmaConfirmerReference assembles the matched-input confirmer's reference bundle from
// existing seams — no new capture code — so the Metal box can inject goinfer's exact L1 state
// into Metal's attention and isolate the crater (0.9994 @ L0 -> 0.640 @ L1) to one of:
//
//	Metal's L1 context MATCHES this reference given matched input  -> the crater is accumulated
//	  f16/precision drift in the residual+KV feeding attention (fix: f32 KV / f32 attention
//	  accumulate for Gemma's low-magnitude contexts).
//	Metal's L1 context STILL inflates on matched input            -> Metal's attention op
//	  (softmax/scale/accumulate) has a real bug, independent of drift.
//
// The bundle at layer L1, position 5 ("The capital of France is"):
//
//	residual entering L1 = ForwardCapture(prompt, [0])[0]   (layer-0 output = L1 input)
//	K/V at L1            = cache.Keys(1) / cache.Vals(1)     (post-RoPE K, raw V, f32)
//	target context       = ForwardSubCapture -> subCtx[1]
//
// Runs on CPU int4 over the byte-identical Q4_K_M gguf (sha 882e8d2d), which reproduces the
// CUDA resident context to decimals — so the Metal box gets the same reference either box would
// produce, and the injection stays entirely on its side (naturally-f32 KV, no faked state).
func TestGemmaConfirmerReference(t *testing.T) {
	gguf := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, e := os.Stat(gguf); e != nil {
		t.Skipf("no gguf at %s", gguf)
	}
	prompt := []int{2, 669, 5279, 529, 7001, 563}
	const probe, injectLayer = 5, 1

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	defer m.Close()

	// Pass 1: residual entering L1 (= layer-0 output) + the populated KV cache.
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
	// L1 is a sliding-window (ring) layer, so its K/V is NOT in Keys/Vals (the global store) —
	// LayerKVForTest unwraps the ring into absolute f32 order (dequant if int8).
	kL1, vL1, kvBase := capCache.LayerKVForTest(injectLayer) // post-RoPE K, raw V, f32

	// Pass 2 (deterministic, same state): the target L1 context.
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
	head := func(v []float32, n int) []float32 {
		if len(v) < n {
			n = len(v)
		}
		return v[:n]
	}
	t.Logf("=== matched-input confirmer reference: layer L%d, pos %d ===", injectLayer, probe)
	t.Logf("residual into L%d : len=%d  L2=%.4f  head=%.4f", injectLayer, len(residIntoL1), l2(residIntoL1), head(residIntoL1, 6))
	t.Logf("K @ L%d           : len=%d  base=%d  L2=%.4f  head=%.4f", injectLayer, len(kL1), kvBase, l2(kL1), head(kL1, 6))
	t.Logf("V @ L%d           : len=%d  L2=%.4f  head=%.4f", injectLayer, len(vL1), l2(vL1), head(vL1, 6))
	t.Logf("target context L%d: len=%d  L2=%.4f  head=%.4f", injectLayer, len(ctxL1), l2(ctxL1), head(ctxL1, 6))
	t.Logf("recipe for Metal: set r.x=residual, write r.kc[%d]=K r.vc[%d]=V, run L%d attention for pos %d, cos(context, target) — match => precision drift, inflate => attention-op bug",
		injectLayer, injectLayer, injectLayer, probe)
}
