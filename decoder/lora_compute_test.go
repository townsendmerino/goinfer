package decoder

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestLoRACompute_matchesMerge is the core correctness gate for compute-time LoRA
// (#7): applying the low-rank delta in the forward (applyLoRA, added to the base
// output W·x) must equal folding it into the weight first (merge, then W'·x). The
// two are algebraically identical — (W+s·B·A)·x = W·x + s·B·(A·x) — but the f32
// reduction is split differently, so the gate is closeness, not bit-exactness.
func TestLoRACompute_matchesMerge(t *testing.T) {
	const out, in, r = 6, 8, 3
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32((i*7+seed*3)%13)*0.1 - 0.6 // deterministic, signed, varied
		}
		return d
	}
	w := fill(out*in, 1)
	a := fill(r*in, 2)
	b := fill(out*r, 3)
	x := fill(in, 4)
	const scale = 1.7
	d := &loraDelta{a: a, b: b, r: r, in: in, out: out, scale: scale}

	// Merge path: fold the delta into a copy of W, then matmul.
	wm := append([]float32(nil), w...)
	ad := &loraAdapter{deltas: map[string]loraDelta{"w": *d}}
	if err := ad.merge("w", wm, out, in); err != nil {
		t.Fatal(err)
	}
	yMerged := make([]float32, out)
	linalg.MatmulBT(x, wm, yMerged, 1, in, out)

	// Compute-time path: base matmul, then add the delta in place.
	yCT := make([]float32, out)
	linalg.MatmulBT(x, w, yCT, 1, in, out)
	applyLoRA(d, x, yCT, &decodeScratch{})

	for o := range out {
		if diff := math.Abs(float64(yMerged[o] - yCT[o])); diff > 1e-4*(1+math.Abs(float64(yMerged[o]))) {
			t.Errorf("y[%d]: compute-time %v vs merged %v (diff %g)", o, yCT[o], yMerged[o], diff)
		}
	}
	// A nil delta is a no-op (untargeted projection).
	yNil := append([]float32(nil), yCT...)
	applyLoRA(nil, x, yCT, &decodeScratch{})
	for i := range yNil {
		if yNil[i] != yCT[i] {
			t.Fatalf("nil delta mutated y at %d", i)
		}
	}
}

// TestLoRACompute_forwardParity loads one synthetic llama base two ways — adapter
// MERGED at load vs the same adapter applied at COMPUTE time on the immutable base
// — and asserts the last-position prefill logits match (argmax + tight cosine).
// The adapter targets attention (q,v) and the MLP (gate,down), so it exercises both
// causalAttention's and gatedMLP's wiring, plus the forced-sequential prefill path.
func TestLoRACompute_forwardParity(t *testing.T) {
	const hidden, heads, headDim, inter, vocab, layers = 8, 2, 4, 16, 16, 2
	qDim := heads * headDim // 8
	base := t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":2,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n, seed int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32((i*5+seed)%11)*0.07 - 0.35
		}
		return d
	}
	ts := map[string]stTensor{
		"model.embed_tokens.weight": {[]int{vocab, hidden}, fill(vocab*hidden, 1)},
		"model.norm.weight":         {[]int{hidden}, fill(hidden, 2)},
		"lm_head.weight":            {[]int{vocab, hidden}, fill(vocab*hidden, 3)},
	}
	for l := range layers {
		p := func(s string) string { return "model.layers." + itoa(l) + s }
		ts[p(".self_attn.q_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 10+l)}
		ts[p(".self_attn.k_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 20+l)}
		ts[p(".self_attn.v_proj.weight")] = stTensor{[]int{qDim, hidden}, fill(qDim*hidden, 30+l)}
		ts[p(".self_attn.o_proj.weight")] = stTensor{[]int{hidden, qDim}, fill(hidden*qDim, 40+l)}
		ts[p(".mlp.gate_proj.weight")] = stTensor{[]int{inter, hidden}, fill(inter*hidden, 50+l)}
		ts[p(".mlp.up_proj.weight")] = stTensor{[]int{inter, hidden}, fill(inter*hidden, 60+l)}
		ts[p(".mlp.down_proj.weight")] = stTensor{[]int{hidden, inter}, fill(hidden*inter, 70+l)}
		ts[p(".input_layernorm.weight")] = stTensor{[]int{hidden}, fill(hidden, 80+l)}
		ts[p(".post_attention_layernorm.weight")] = stTensor{[]int{hidden}, fill(hidden, 90+l)}
	}
	writeSafetensors(t, filepath.Join(base, "model.safetensors"), ts)

	// Adapter on q,v,gate,down across both layers; r=2, alpha=4 → scale 2.
	const r = 2
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"),
		[]byte(`{"r":2,"lora_alpha":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	at := map[string]stTensor{}
	pfx := "base_model.model.model.layers."
	for l := range layers {
		add := func(mod string, inDim, outDim int) {
			at[pfx+itoa(l)+mod+".lora_A.weight"] = stTensor{[]int{r, inDim}, fill(r*inDim, 100+l)}
			at[pfx+itoa(l)+mod+".lora_B.weight"] = stTensor{[]int{outDim, r}, fill(outDim*r, 200+l)}
		}
		add(".self_attn.q_proj", hidden, qDim)
		add(".self_attn.v_proj", hidden, qDim)
		add(".mlp.gate_proj", hidden, inter)
		add(".mlp.down_proj", inter, hidden)
	}
	writeSafetensors(t, filepath.Join(adapter, "adapter_model.safetensors"), at)

	be, _ := NewBackend("")
	newModel := func(lo *loraAdapter) *Model {
		w, err := loadWeights(base, quantNone, false, lo)
		if err != nil {
			t.Fatalf("loadWeights: %v", err)
		}
		return &Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs()}
	}

	// Merged reference.
	lo, err := loadLoRA(adapter)
	if err != nil {
		t.Fatal(err)
	}
	defer lo.close()
	mMerged := newModel(lo)
	defer mMerged.Close()

	// Compute-time: immutable base + the adapter applied in the forward.
	mBase := newModel(nil)
	defer mBase.Close()
	if err := mBase.LoadAdapter("a", adapter); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	prompt := []int{1, 5, 3, 9, 2, 7}
	cm := mMerged.NewCache(len(prompt))
	lMerged, err := mMerged.prefillLogits(prompt, cm)
	if err != nil {
		t.Fatalf("merged prefill: %v", err)
	}
	cb := mBase.NewCache(len(prompt))
	cb.lora = mBase.adapter("a")
	lCT, err := mBase.prefillLogits(prompt, cb)
	if err != nil {
		t.Fatalf("compute-time prefill: %v", err)
	}
	// Sanity: the adapter is not vacuous — base-with-NO-adapter logits differ.
	cn := mBase.NewCache(len(prompt))
	lNone, err := mBase.prefillLogits(prompt, cn)
	if err != nil {
		t.Fatalf("base prefill: %v", err)
	}

	if argmax(lCT) != argmax(lMerged) {
		t.Errorf("argmax compute-time %d != merged %d", argmax(lCT), argmax(lMerged))
	}
	if c := cosine(lCT, lMerged); c < 0.99999 {
		t.Errorf("compute-time vs merged cosine %v, want ~1", c)
	}
	if cosine(lNone, lMerged) > 0.999999 {
		t.Error("base (no adapter) logits indistinguishable from merged — adapter has no effect, test is vacuous")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
}
