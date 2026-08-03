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
	requireHeavyModel(t)
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
	var residIntoL1, residIntoL1P0 []float32
	for i, tok := range prompt {
		_, hid, err := m.ForwardCapture(tok, capCache, []int{injectLayer - 1})
		if err != nil {
			t.Fatalf("ForwardCapture pos %d: %v", i, err)
		}
		if i == 0 {
			residIntoL1P0 = append([]float32(nil), hid[0]...)
		}
		if i == probe {
			residIntoL1 = append([]float32(nil), hid[0]...)
		}
	}
	// Is Metal's L0 OUTPUT for the BOS (residual entering L1 at pos 0) already wrong? If so the
	// bug is L0's BOS processing; if it matches, the pos-0 K-compute itself is the culprit (but
	// positions 1..4 share those kernels and are faithful).
	emb0 := m.EmbedResidentForTest(prompt[0])
	{
		var s float64
		for _, x := range emb0 {
			s += float64(x) * float64(x)
		}
		t.Logf("BOS input embedding: |emb|=%.2f (if ~12491 the massive activation is in the embedding and L0 keeps it; if ~1461 L0 must build it)", math.Sqrt(s))
	}
	// L0 attn-vs-mlp contribution at the BOS (CUDA box: attn grows it to ~642, MLP builds ~12583 at
	// ch 443 = 96% of the massive activation). Confirm Metal's MLP contribution is ~10x too small.
	{
		sa, sm, smPre, _, _ := r.forwardSubCaptureForTest(emb0, 0)
		l2 := func(v []float32) float64 {
			var s float64
			for _, x := range v {
				s += float64(x) * float64(x)
			}
			return math.Sqrt(s)
		}
		t.Logf("L0 BOS: attn ch443=%.1f | mlp POST-norm ch443=%.1f | mlp PRE-norm(down out) ch443=%.1f (CUDA f32 post~12583)",
			sa[0][443], sm[0][443], smPre[0][443])
		t.Logf("  compute-vs-norm: if PRE-norm down out is already large -> MLP compute bug; if near-zero & post-norm blows it up -> sandwich-norm RMS (|mlpPre|=%.1f |mlp|=%.1f)",
			l2(smPre[0]), l2(sm[0]))
	}
	// geglu-vs-down cut: Metal's L0 BOS gate|up (f32) -> f32 geglu vs the int8 round-trip the
	// down-proj consumes. If Metal's f32 geglu matches CUDA's but the int8 round-trip diverges,
	// it's the crush (absmax scale dominated by an outlier crushing the sink's tiny channels).
	{
		gateUp, geglu8, gSc := r.l0GegluForTest(emb0, 0)
		I := len(geglu8)
		gelu := func(x float32) float32 {
			xf := float64(x)
			return float32(0.5 * xf * (1 + math.Tanh(0.7978845608028654*(xf+0.044715*xf*xf*xf))))
		}
		var f32n, i8n, amax float64
		var crushed, amaxCh int
		for i := 0; i < I; i++ {
			s := gelu(gateUp[i]) * gateUp[I+i] // f32 geglu
			f32n += float64(s) * float64(s)
			if a := math.Abs(float64(s)); a > amax {
				amax, amaxCh = a, i
			}
			i8n += float64(geglu8[i]) * float64(geglu8[i])
			if geglu8[i] == 0 {
				crushed++
			}
		}
		t.Logf("L0 BOS geglu: |f32|=%.3f |int8-rt|=%.3f  int8 scale=%.4g (kernel absmax=%.3f, my-f32 absmax=%.3f) crushed=%d/%d",
			math.Sqrt(f32n), math.Sqrt(i8n), gSc, gSc*127, amax, crushed, I)
		g, u := gateUp[amaxCh], gateUp[I+amaxCh]
		t.Logf("  max-geglu ch %d: gate=%.4g up=%.4g my-gelu(gate)=%.4g geglu=%.4g int8-rt=%.4g",
			amaxCh, g, u, gelu(g), gelu(g)*u, geglu8[amaxCh])
	}
	mResid0 := r.forwardTrunkForTest(emb0, 0, 1)
	{
		var dot, na, nb float64
		for i := range residIntoL1P0 {
			dot += float64(mResid0[i]) * float64(residIntoL1P0[i])
			na += float64(mResid0[i]) * float64(mResid0[i])
			nb += float64(residIntoL1P0[i]) * float64(residIntoL1P0[i])
		}
		t.Logf("BOS residual entering L1 (pos 0): cos(Metal, goinfer)=%.5f |Metal|=%.2f |goinfer|=%.2f",
			dot/(math.Sqrt(na)*math.Sqrt(nb)+1e-30), math.Sqrt(na), math.Sqrt(nb))
	}
	kL1, vL1, kvBase := capCache.LayerKVForTest(injectLayer)

	subCache := m.NewCache(len(prompt) + 1)
	var ctxL1 []float32
	for i, tok := range prompt {
		if i == probe {
			_, _, ctx, _, err := m.ForwardSubCapture(tok, subCache)
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
	// Diagnostic: is Metal's WALKED K (now f32) close to goinfer's K, or does the KV VALUE drift?
	// If they match, the crater is NOT the KV storage/compute — it's the residual. If they differ,
	// Metal computes different K/V during the walk (from drifted residuals), which f32 storage
	// cannot fix. Compare the walked positions (0..probe-1) that LayerKVForTest returns.
	{
		kvDim := r.layers[injectLayer].geom.kvDim
		mk := make([]float32, probe*kvDim)
		if r.kvF32 {
			copy(mk, r.kc[injectLayer].Floats()[:probe*kvDim])
		} else {
			for i := range mk {
				mk[i] = f16ToF32(r.kc[injectLayer].U16s()[i])
			}
		}
		// Per-position: pos 0's input is the IDENTICAL embedding, so a faithful K-compute MUST
		// match goinfer there. If pos 0 drifts too, the K-compute itself is wrong; if only later
		// positions drift, it's residual drift feeding a faithful K-compute.
		for p := 0; p < probe; p++ {
			var dot, na, nb float64
			for j := 0; j < kvDim; j++ {
				i := p*kvDim + j
				dot += float64(mk[i]) * float64(kL1[i])
				na += float64(mk[i]) * float64(mk[i])
				nb += float64(kL1[i]) * float64(kL1[i])
			}
			t.Logf("  walked K pos %d: cos=%.5f |metal|=%.2f |goinfer|=%.2f", p, dot/(math.Sqrt(na)*math.Sqrt(nb)+1e-30), math.Sqrt(na), math.Sqrt(nb))
		}
		t.Logf("  (kvF32=%v)", r.kvF32)
	}
	cB, nB := cosTo(r.attnConfirmForTest(residIntoL1, nil, nil, injectLayer, probe, false))
	t.Logf("(B) matched resid + Metal WALKED KV: cos=%.4f |Metal|=%.2f |target|=%.2f", cB, nB, l2(ctxL1))

	// (C) Same as (B) but OVERWRITE just position 0's K/V with goinfer's (the BOS, cos 0.40). If
	// (C) recovers, the whole crater is Metal's position-0/BOS K being wrong.
	for i := 0; i < probe; i++ {
		r.forwardTrunkForTest(m.EmbedResidentForTest(prompt[i]), i, r.nL)
	}
	kvDim := r.layers[injectLayer].geom.kvDim
	if r.kvF32 {
		copy(r.kc[injectLayer].Floats()[:kvDim], kL1[:kvDim])
		copy(r.vc[injectLayer].Floats()[:kvDim], vL1[:kvDim])
	} else {
		kc, vc := r.kc[injectLayer].U16s(), r.vc[injectLayer].U16s()
		for j := 0; j < kvDim; j++ {
			kc[j], vc[j] = f32ToF16(kL1[j]), f32ToF16(vL1[j])
		}
	}
	cC, nC := cosTo(r.attnConfirmForTest(residIntoL1, nil, nil, injectLayer, probe, false))
	t.Logf("(C) (B) but with goinfer's BOS K/V at pos 0: cos=%.4f |Metal|=%.2f |target|=%.2f", cC, nC, l2(ctxL1))

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
