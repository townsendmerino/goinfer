package fit

import "testing"

// Every case prices ONE tensor with its own element count and checks the bytes that branch alone produces, so a swapped or dropped
// branch moves that case's number. Rates: matrix 0.5, embedding 2, int8 1, f32 4 bytes per element.
type tensorCase struct {
	name  string
	dims  []int // nil: the tensor is unknown to the reader (ok false)
	rates func(*Rates)
	want  int64
}

var rates = Rates{Matrix: 0.5, Embedding: 2, Int8: 1}

func (c tensorCase) rate() Rates {
	r := rates
	if c.rates != nil {
		c.rates(&r)
	}
	return r
}

func TestGGUFWeightBytes_branches(t *testing.T) {
	kept := func(r *Rates) {
		r.Kept = func(name string, _ []int) (float64, bool) { return 7, name == "blk.0.attn_q.weight" }
	}
	mamba := func(r *Rates) { r.MambaProjInt8 = true }
	mla := func(r *Rates) { r.MLAProjInt8 = true }
	for _, c := range []struct {
		tensorCase
		arch    string
		experts int64
	}{
		{tensorCase{"blk.0.attn_q.weight", []int{2, 3}, nil, 3}, "llama", 0},               // matrix: 6 * 0.5
		{tensorCase{"blk.0.attn_norm.weight", []int{7}, nil, 28}, "llama", 0},              // 1-D: f32
		{tensorCase{"blk.0.ssm_in.weight", []int{5, 3}, nil, 60}, "nemotron_h", 0},         // f32 pinned: 15 * 4
		{tensorCase{"blk.0.ssm_in.weight", []int{6, 7}, mamba, 42}, "nemotron_h", 0},       // int8: 42 * 1
		{tensorCase{"blk.0.ssm_out.weight", []int{3, 11}, mamba, 33}, "nemotron_h_moe", 0}, // int8: 33 * 1
		{tensorCase{"blk.0.ssm_in.weight", []int{6, 7}, mamba, 168}, "llama", 0},           // not Nemotron-H: stays f32
		{tensorCase{"blk.0.attn_q_a.weight", []int{4, 6}, mla, 24}, "deepseek2", 0},        // MLA int8: 24 * 1
		{tensorCase{"blk.0.attn_q_a.weight", []int{4, 6}, nil, 96}, "deepseek2", 0},        // MLA f32 pinned: 24 * 4
		{tensorCase{"blk.0.attn_kv_b.weight", []int{3, 5}, mla, 60}, "deepseek2", 0},       // kv_b stays f32 under MLAProjInt8
		{tensorCase{"blk.0.ffn_gate_exps.weight", []int{4, 5, 2}, nil, 20}, "llama", 20},   // matrix 40 * 0.5, all expert
		{tensorCase{"blk.0.ffn_gate_exps.bias", []int{3, 4}, nil, 48}, "llama", 0},         // f32 pinned, not counted as expert
		{tensorCase{"blk.0.attn_q.weight", []int{2, 3}, kept, 7}, "llama", 0},              // Kept overrides the matrix price
		{tensorCase{"blk.0.attn_k.weight", []int{2, 3}, kept, 3}, "llama", 0},              // Kept false: matrix price
		{tensorCase{"blk.0.zero.weight", []int{10, 0}, nil, 0}, "llama", 0},                // zero elements
		{tensorCase{"blk.0.neg.weight", []int{-1, 4}, nil, 0}, "llama", 0},                 // non-positive dimension
		{tensorCase{"blk.0.unknown.weight", nil, nil, 0}, "llama", 0},                      // shape unreadable
	} {
		get := func(n string) ([]int, bool) { return c.dims, c.dims != nil }
		total, exp := GGUFWeightBytes([]string{c.name}, get, c.arch, c.rate())
		if total != c.want || exp != c.experts {
			t.Errorf("%s %v arch %s: total %d experts %d, want %d and %d", c.name, c.dims, c.arch, total, exp, c.want, c.experts)
		}
	}
}

func TestSafetensorsWeightBytes_branches(t *testing.T) {
	mamba := func(r *Rates) { r.MambaProjInt8 = true }
	mla := func(r *Rates) { r.MLAProjInt8 = true }
	for _, c := range []tensorCase{
		{"model.layers.0.q.weight", []int{2, 3}, nil, 3},                // matrix: 6 * 0.5
		{"model.layers.0.norm.weight", []int{7}, nil, 28},               // 1-D: f32
		{"model.embed_tokens.weight", []int{4, 5}, nil, 40},             // embedding: 20 * 2
		{"lm_head.weight", []int{2, 11}, nil, 44},                       // embedding: 22 * 2
		{"model.layers.0.x.qweight", []int{3, 2}, nil, 24},              // packed: 6 * 8 * 0.5
		{"model.layers.0.x.scales", []int{9, 9}, nil, 0},                // GPTQ aux: never resident
		{"model.layers.0.x.qzeros", []int{9, 9}, nil, 0},                // GPTQ aux
		{"model.layers.0.x.g_idx", []int{9, 9}, nil, 0},                 // GPTQ aux
		{"vision_tower.blocks.0.w", []int{3, 13}, nil, 156},             // f32: 39 * 4
		{"multi_modal_projector.w", []int{3, 17}, nil, 204},             // f32: 51 * 4
		{"model.layers.0.mixer.conv1d.weight", []int{5, 3}, nil, 60},    // f32 pinned: 15 * 4
		{"model.layers.0.mixer.in_proj.weight", []int{6, 7}, mamba, 42}, // int8: 42 * 1
		{"model.layers.0.mixer.out_proj.weight", []int{3, 11}, mamba, 33},
		{"model.layers.0.mixer.in_proj.weight", []int{6, 7}, nil, 168},      // not armed: f32 pinned
		{"model.layers.0.self_attn.q_a_proj.weight", []int{4, 6}, mla, 24},  // MLA int8: 24 * 1
		{"model.layers.0.self_attn.q_a_proj.weight", []int{4, 6}, nil, 96},  // MLA f32 pinned: 24 * 4
		{"model.layers.0.self_attn.kv_b_proj.weight", []int{3, 5}, mla, 60}, // kv_b stays f32
		{"model.layers.0.zero.weight", []int{10, 0}, nil, 0},                // zero elements
		{"model.layers.0.neg.weight", []int{-1, 4}, nil, 0},                 // non-positive dimension
		{"model.layers.0.unknown.weight", nil, nil, 0},                      // shape unreadable
	} {
		get := func(n string) ([]int, bool) { return c.dims, c.dims != nil }
		if got := SafetensorsWeightBytes([]string{c.name}, get, c.rate()); got != c.want {
			t.Errorf("%s %v: got %d, want %d", c.name, c.dims, got, c.want)
		}
	}
}

// TestSafetensorsWeightBytes_sums checks that tensors add, with distinct counts so no two orderings agree.
func TestSafetensorsWeightBytes_sums(t *testing.T) {
	shapes := map[string][]int{"a.weight": {2, 3}, "b.weight": {7}, "model.embed_tokens.weight": {4, 5}}
	get := func(n string) ([]int, bool) { s, ok := shapes[n]; return s, ok }
	names := []string{"a.weight", "b.weight", "model.embed_tokens.weight", "missing.weight"}
	if got, want := SafetensorsWeightBytes(names, get, rates), int64(3+28+40); got != want {
		t.Errorf("total %d, want %d", got, want)
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
		{2, 4, 0, false, false, 0},
		{2, 4, -8, false, false, 0}, // unknown geometry is 0, never negative
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
	if mlaInt8TensorName("blk.0.attn_kv_b.weight") || !mlaInt8TensorName("blk.0.attn_kv_a_mqa.weight") {
		t.Error("mlaInt8TensorName: kv_b stays f32, kv_a_mqa is int8")
	}
	if !embeddingTensorName("model.embed_tokens.weight") || !embeddingTensorName("output.weight") || embeddingTensorName("model.layers.0.embed_tokens.weight.bias") {
		t.Error("embeddingTensorName")
	}
}

// TestGGUFWeightBytes_sums checks that total and the expert share accumulate across tensors, with distinct counts.
func TestGGUFWeightBytes_sums(t *testing.T) {
	dims := map[string][]int{
		"blk.0.attn_q.weight":        {2, 3},    // 6 * 0.5 = 3
		"blk.0.attn_norm.weight":     {7},       // 28
		"blk.0.ffn_gate_exps.weight": {4, 5, 2}, // 40 * 0.5 = 20, expert
		"blk.1.ffn_up_exps.weight":   {2, 9},    // 18 * 0.5 = 9, expert
		"blk.0.ssm_in.weight":        {5, 3},    // 60, f32 pinned, not expert
		"blk.0.scalar":               {},        // 4
	}
	names := []string{"blk.0.attn_q.weight", "blk.0.attn_norm.weight", "blk.0.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.0.ssm_in.weight", "blk.0.scalar", "blk.0.missing.weight"}
	get := func(n string) ([]int, bool) { d, ok := dims[n]; return d, ok }
	total, exp := GGUFWeightBytes(names, get, "llama", rates)
	if total != 3+28+20+9+60+4 || exp != 20+9 {
		t.Errorf("total %d experts %d, want %d and %d", total, exp, 3+28+20+9+60+4, 20+9)
	}
}
