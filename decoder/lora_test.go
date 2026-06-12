package decoder

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// TestLoRAMerge checks the in-place merge computes W + scale·B·A exactly.
func TestLoRAMerge(t *testing.T) {
	const out, in, r = 3, 4, 2
	a := []float32{ // [r, in]
		1, 0, 2, 0,
		0, 3, 0, 1,
	}
	b := []float32{ // [out, r]
		1, 0,
		0, 1,
		1, 1,
	}
	const scale = 0.5
	base := make([]float32, out*in) // zeros, so the result is purely the delta
	adapter := &loraAdapter{deltas: map[string]loraDelta{
		"w": {a: a, b: b, r: r, in: in, out: out, scale: scale},
	}}
	if err := adapter.merge("w", base, out, in); err != nil {
		t.Fatal(err)
	}
	// want[o,j] = scale · Σ_k B[o,k]·A[k,j]
	want := make([]float32, out*in)
	for o := range out {
		for j := range in {
			var s float64
			for k := range r {
				s += float64(b[o*r+k]) * float64(a[k*in+j])
			}
			want[o*in+j] = float32(scale * s)
		}
	}
	for i := range want {
		if math.Abs(float64(base[i]-want[i])) > 1e-6 {
			t.Errorf("merge[%d] = %v, want %v", i, base[i], want[i])
		}
	}
	// A non-target tensor is left untouched.
	other := []float32{9, 9}
	_ = adapter.merge("missing", other, 1, 2)
	if other[0] != 9 || other[1] != 9 {
		t.Errorf("non-target tensor mutated: %v", other)
	}
	// Shape mismatch is an error.
	if err := adapter.merge("w", base, out+1, in); err == nil {
		t.Error("expected shape-mismatch error")
	}
}

func TestLoRABaseName(t *testing.T) {
	got := loraBaseName("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", ".lora_A.weight")
	if want := "model.layers.0.self_attn.q_proj.weight"; got != want {
		t.Errorf("loraBaseName = %q, want %q", got, want)
	}
}

func TestLoRAValidateTargets(t *testing.T) {
	a := &loraAdapter{deltas: map[string]loraDelta{
		"model.layers.0.self_attn.q_proj.weight": {},
	}}
	if err := a.validateTargets(1, &llamaTensorSchema); err != nil {
		t.Errorf("q_proj should be a valid target: %v", err)
	}
	bad := &loraAdapter{deltas: map[string]loraDelta{
		"model.embed_tokens.weight": {},
	}}
	if err := bad.validateTargets(1, &llamaTensorSchema); err == nil {
		t.Error("embed_tokens is not a supported merge target — want error")
	}
}

// TestLoadLoRA writes a tiny PEFT adapter (config + safetensors) and loads it,
// checking the config-derived scale, name mapping, and A/B pairing.
func TestLoadLoRA(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"r":8,"lora_alpha":16,"use_rslora":false,"target_modules":["q_proj"]}`
	if err := os.WriteFile(filepath.Join(dir, "adapter_config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	// q_proj: out=6, in=4, r=8 → A [8,4], B [6,8].
	writeSafetensors(t, filepath.Join(dir, "adapter_model.safetensors"), map[string]stTensor{
		"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {[]int{8, 4}, make([]float32, 8*4)},
		"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": {[]int{6, 8}, make([]float32, 6*8)},
	})
	a, err := loadLoRA(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer a.close()
	d, ok := a.deltas["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatalf("delta not mapped to base name; have %v", keysOf(a.deltas))
	}
	if d.scale != 16.0/8.0 {
		t.Errorf("scale = %v, want 2.0 (alpha/r)", d.scale)
	}
	if d.r != 8 || d.in != 4 || d.out != 6 {
		t.Errorf("dims r=%d in=%d out=%d, want 8/4/6", d.r, d.in, d.out)
	}
}

// TestLoRA_mergeAtLoad exercises the full Load → loadProj → merge wiring on a
// tiny synthetic llama: loading with the adapter must yield q_proj = base +
// scale·B·A, and leave every other tensor (here k_proj) untouched.
func TestLoRA_mergeAtLoad(t *testing.T) {
	const hidden, heads, headDim, inter, vocab = 8, 2, 4, 16, 16
	qDim := heads * headDim // 8
	base := t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":1,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n int) []float32 { // deterministic, varied
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(i%7)*0.1 - 0.3
		}
		return d
	}
	ts := map[string]stTensor{
		"model.embed_tokens.weight":                      {[]int{vocab, hidden}, fill(vocab * hidden)},
		"model.norm.weight":                              {[]int{hidden}, fill(hidden)},
		"lm_head.weight":                                 {[]int{vocab, hidden}, fill(vocab * hidden)},
		"model.layers.0.self_attn.q_proj.weight":         {[]int{qDim, hidden}, fill(qDim * hidden)},
		"model.layers.0.self_attn.k_proj.weight":         {[]int{qDim, hidden}, fill(qDim * hidden)},
		"model.layers.0.self_attn.v_proj.weight":         {[]int{qDim, hidden}, fill(qDim * hidden)},
		"model.layers.0.self_attn.o_proj.weight":         {[]int{hidden, qDim}, fill(hidden * qDim)},
		"model.layers.0.mlp.gate_proj.weight":            {[]int{inter, hidden}, fill(inter * hidden)},
		"model.layers.0.mlp.up_proj.weight":              {[]int{inter, hidden}, fill(inter * hidden)},
		"model.layers.0.mlp.down_proj.weight":            {[]int{hidden, inter}, fill(hidden * inter)},
		"model.layers.0.input_layernorm.weight":          {[]int{hidden}, fill(hidden)},
		"model.layers.0.post_attention_layernorm.weight": {[]int{hidden}, fill(hidden)},
	}
	writeSafetensors(t, filepath.Join(base, "model.safetensors"), ts)

	// Adapter on q_proj only: r=2, alpha=4 → scale 2.
	const r = 2
	aData := []float32{0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4, 0.0, 0.1, -0.3, 0.2, 0.1, 0.0, -0.2, 0.3} // [2,8]
	bData := []float32{0.2, 0.1, -0.1, 0.3, 0.0, 0.2, 0.4, -0.2, 0.1, 0.1, -0.3, 0.0, 0.2, 0.1, 0.3, -0.1} // [8,2]
	adapter := t.TempDir()
	if err := os.WriteFile(filepath.Join(adapter, "adapter_config.json"),
		[]byte(`{"r":2,"lora_alpha":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSafetensors(t, filepath.Join(adapter, "adapter_model.safetensors"), map[string]stTensor{
		"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {[]int{r, hidden}, aData},
		"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": {[]int{qDim, r}, bData},
	})

	w0, err := loadWeights(base, quantNone, nil)
	if err != nil {
		t.Fatalf("load base: %v", err)
	}
	lo, err := loadLoRA(adapter)
	if err != nil {
		t.Fatal(err)
	}
	defer lo.close()
	w1, err := loadWeights(base, quantNone, lo)
	if err != nil {
		t.Fatalf("load merged: %v", err)
	}

	q0, q1 := tF32(&w0.Layers[0].QProj), tF32(&w1.Layers[0].QProj)
	const scale = 2.0
	for o := range qDim {
		for j := range hidden {
			var s float64
			for k := range r {
				s += float64(bData[o*r+k]) * float64(aData[k*hidden+j])
			}
			want := q0[o*hidden+j] + float32(scale*s)
			if math.Abs(float64(q1[o*hidden+j]-want)) > 1e-5 {
				t.Fatalf("q_proj[%d,%d] = %v, want base+delta %v", o, j, q1[o*hidden+j], want)
			}
		}
	}
	// k_proj (not a target) must be identical.
	for i := range tF32(&w0.Layers[0].KProj) {
		if tF32(&w0.Layers[0].KProj)[i] != tF32(&w1.Layers[0].KProj)[i] {
			t.Fatalf("k_proj changed at %d (should be untouched)", i)
		}
	}
}

func keysOf(m map[string]loraDelta) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}

type stTensor struct {
	shape []int
	data  []float32
}

// writeSafetensors writes a minimal F32 .safetensors file (header + blob).
func writeSafetensors(t *testing.T, path string, tensors map[string]stTensor) {
	t.Helper()
	type meta struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]meta{}
	var blob []byte
	off := 0
	for _, n := range names {
		d := tensors[n]
		b := make([]byte, len(d.data)*4)
		for i, f := range d.data {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
		}
		header[n] = meta{"F32", d.shape, [2]int{off, off + len(b)}}
		blob = append(blob, b...)
		off += len(b)
	}
	hjson, _ := json.Marshal(header)
	var buf []byte
	lenHdr := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenHdr, uint64(len(hjson)))
	buf = append(buf, lenHdr...)
	buf = append(buf, hjson...)
	buf = append(buf, blob...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}
