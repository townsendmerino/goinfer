//go:build darwin

package metal

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemvExt_bitIdentical is D-B04's identity gate (docs/audit-metal-2026-09-30.md): teacher-forced decode on the tiny
// Qwen3.5 hybrid (DeltaNet layers, and MoE layers with a gated shared expert) with gemvExtOn against the same tokens
// with it off, each on its own freshly built resident. Every logit at every position must match bit for bit, and the
// on arm must have dispatched the coal rows kernel (DeltaNet qkv/z, the shared expert's down) and the SA rows kernel
// (the shared expert's gate|up), or the comparison is of a path with itself.
func TestGemvExt_bitIdentical(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const steps = 24
	run := func(on bool) ([][]float32, [3]int) {
		gemvExtOn = on
		defer func() { gemvExtOn = false }()
		m, err := decoder.Load(filepath.Join("..", "testdata", "qwen35-tiny"), decoder.Options{Quant: "int4", ResidentContext: 256, ResidentKVSlots: 1})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer r.Close()
		var out [][]float32
		e := make([]float32, r.H)
		for pos := range steps {
			r.embed.Row((pos*131+7)%r.V, e)
			out = append(out, append([]float32(nil), r.ForwardEmb(e, pos)...))
		}
		return out, r.gemvExtRows
	}
	ref, off := run(false)
	got, on := run(true)
	if off != [3]int{} {
		t.Fatalf("the off arm dispatched rows kernels: %v", off)
	}
	if on[gemvExtKCoal] == 0 || on[gemvExtKSA] == 0 {
		t.Fatalf("the on arm dispatched coal/sa/resid rows kernels %v: the fixture did not reach the coal and SA sites", on)
	}
	diff := 0
	for p := range ref {
		for i := range ref[p] {
			if math.Float32bits(ref[p][i]) != math.Float32bits(got[p][i]) {
				diff++
			}
		}
	}
	t.Logf("%d positions x %d logits: %d differ; rows-form dispatches coal/sa/resid %v", steps, len(ref[0]), diff, on)
	if diff != 0 {
		t.Fatalf("%d logits differ with gemvExtOn: the rows kernels are not bit-identical to the ones they replace", diff)
	}
}
