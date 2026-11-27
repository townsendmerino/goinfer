package loraload

import (
	"math"
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
	adapter := &Adapter{Deltas: map[string]Delta{
		"w": {A: a, B: b, R: r, In: in, Out: out, Scale: scale},
	}}
	if err := adapter.Merge("w", base, out, in); err != nil {
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
	_ = adapter.Merge("missing", other, 1, 2)
	if other[0] != 9 || other[1] != 9 {
		t.Errorf("non-target tensor mutated: %v", other)
	}
	// Shape mismatch is an error.
	if err := adapter.Merge("w", base, out+1, in); err == nil {
		t.Error("expected shape-mismatch error")
	}
}

func TestLoRABaseName(t *testing.T) {
	got := BaseName("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", ".lora_A.weight")
	if want := "model.layers.0.self_attn.q_proj.weight"; got != want {
		t.Errorf("BaseName = %q, want %q", got, want)
	}
}
