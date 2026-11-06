package fit

import "testing"

// pricing drives both estimators over a synthetic tensor table, so each branch is asserted on bytes it alone can produce.
func TestWeightBytes_branches(t *testing.T) {
	dims := map[string][]int{
		"blk.0.attn_q.weight":        {10, 10}, // matrix: 100 * Matrix
		"blk.0.attn_norm.weight":     {10},     // 1-D: f32
		"blk.0.ssm_in.weight":        {10, 10}, // f32 pinned (ssm_), or Int8 on a quantized Nemotron-H
		"blk.0.attn_q_a.weight":      {10, 10}, // f32 pinned MLA, or Int8 with MLAProjInt8
		"blk.0.ffn_gate_exps.weight": {10, 10}, // matrix and counted as experts
		"blk.0.kept.weight":          {10, 10}, // Kept overrides Matrix
		"blk.0.empty.weight":         {10, 0},  // zero elements: skipped
	}
	names := []string{}
	for n := range dims {
		names = append(names, n)
	}
	get := func(n string) ([]int, bool) { d, ok := dims[n]; return d, ok }
	r := Rates{Matrix: 0.5, Int8: 1, Embedding: 1}

	total, exp := GGUFWeightBytes(names, get, "llama", r)
	// attn_q 50 + norm 40 + ssm_in f32 400 + attn_q_a f32 400 + experts 50 + kept 50 (no Kept yet)
	if want := int64(50 + 40 + 400 + 400 + 50 + 50); total != want || exp != 50 {
		t.Errorf("plain: total %d experts %d, want %d and 50", total, exp, want)
	}

	r.MambaProjInt8, r.MLAProjInt8 = true, true
	r.Kept = func(name string, _ []int) (float64, bool) { return 7, name == "blk.0.kept.weight" }
	total, _ = GGUFWeightBytes(names, get, "nemotron_h", r)
	// ssm_in at Int8 (100), attn_q_a at Int8 (100), kept 7
	if want := int64(50 + 40 + 100 + 100 + 50 + 7); total != want {
		t.Errorf("nemotron_h int8: total %d, want %d", total, want)
	}
	total, _ = GGUFWeightBytes(names, get, "llama", r)
	// Mamba int8 applies only to the nemotron architectures; MLA int8 applies anywhere
	if want := int64(50 + 40 + 400 + 100 + 50 + 7); total != want {
		t.Errorf("llama with MLA int8: total %d, want %d", total, want)
	}
}

func TestSafetensorsWeightBytes_branches(t *testing.T) {
	shapes := map[string][]int{
		"model.layers.0.q.weight":             {10, 10},
		"model.layers.0.norm.weight":          {10},
		"model.embed_tokens.weight":           {10, 10},
		"lm_head.weight":                      {10, 10},
		"model.layers.0.x.qweight":            {10, 10}, // packed: 8x
		"model.layers.0.x.scales":             {10, 10}, // GPTQ aux: zero
		"vision_tower.blocks.0.w":             {10, 10}, // f32
		"model.layers.0.mixer.in_proj.weight": {10, 10}, // f32 pinned, or Int8 with MambaProjInt8
	}
	names := []string{}
	for n := range shapes {
		names = append(names, n)
	}
	get := func(n string) ([]int, bool) { s, ok := shapes[n]; return s, ok }
	r := Rates{Matrix: 0.5, Embedding: 2, Int8: 1}
	got := SafetensorsWeightBytes(names, get, r)
	// q 50 + norm 40 + embed 200 + head 200 + qweight 100*8*0.5=400 + vision 400 + mixer f32 400
	if want := int64(50 + 40 + 200 + 200 + 400 + 400 + 400); got != want {
		t.Errorf("total %d, want %d", got, want)
	}
	r.MambaProjInt8 = true
	if got, want := SafetensorsWeightBytes(names, get, r), int64(50+40+200+200+400+400+100); got != want {
		t.Errorf("mamba int8: total %d, want %d", got, want)
	}
}

func TestKVBytesPerPosition(t *testing.T) {
	for _, c := range []struct {
		layers, heads, dim int
		f16, i8            bool
		want               int64
	}{
		{2, 4, 8, false, false, 2 * 4 * 32 * 2}, // K+V * f32 * kvDim 32 * 2 layers
		{2, 4, 8, true, false, 2 * 2 * 32 * 2},
		{2, 4, 8, true, true, int64(2 * 1.125 * 32 * 2)}, // int8 wins over f16
		{0, 4, 8, false, false, 0},
		{2, 0, 8, false, false, 0},
	} {
		if got := KVBytesPerPosition(c.layers, c.heads, c.dim, c.f16, c.i8); got != c.want {
			t.Errorf("%+v: got %d", c, got)
		}
	}
}

func TestTensorNameClassifiers(t *testing.T) {
	for name, want := range map[string]bool{
		"blk.3.ssm_a": true, "model.layers.0.mixer.conv1d.weight": true, "blk.0.attn_kv_b.weight": true,
		"blk.0.ffn_gate_exps.bias": true, "model.layers.0.mixer.up_proj.weight": false, "linear_attn.in_proj_qkv.weight": false,
		"blk.0.attn_q.weight": false,
	} {
		if got := F32PinnedTensorName(name); got != want {
			t.Errorf("F32PinnedTensorName(%q) = %v", name, got)
		}
	}
	if MLAInt8TensorName("blk.0.attn_kv_b.weight") || !MLAInt8TensorName("blk.0.attn_kv_a_mqa.weight") {
		t.Error("MLAInt8TensorName: kv_b stays f32, kv_a_mqa is int8")
	}
	if !embeddingTensorName("model.embed_tokens.weight") || !embeddingTensorName("output.weight") || embeddingTensorName("model.layers.0.embed_tokens.weight.bias") {
		t.Error("embeddingTensorName")
	}
}
