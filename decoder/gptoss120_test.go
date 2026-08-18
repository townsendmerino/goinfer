//go:build realckpt

// gpt-oss-120b loader gate — does the safetensors path generalize past the 20b it was
// written against?
//
// WHAT THIS DOES AND DOES NOT CHECK, stated because the difference decides its value. The
// forward math is already proven end to end on 20b (argmax-identical, cosine 0.999121 vs
// the T3-validated GGUF reader). What 120b changes is SHAPE, not math: 36 layers instead
// of 24, 128 experts instead of 32, and 14 shards instead of 2. So the risk this gate is
// pointed at is EXPERT INDEXING at four times the count and tensors spanning many shards —
// an off-by-N in the expert stride reads the wrong expert's weights while every shape check
// still passes.
//
// It deliberately does NOT build an HF reference. Four layers of 120b dequantize to ~51GB
// in f32, which does not fit; claiming a numeric oracle here would mean pretending to a
// check that cannot run. Instead it asserts geometry, spot-checks the HIGHEST expert index
// (where a stride bug shows first and a low-index check would miss), and requires finite,
// non-degenerate logits.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestGptOss120 -v -timeout 60m
package decoder

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestGptOss120_sliceLoads(t *testing.T) {
	requireHeavyModel(t)
	const ckpt = "testdata/gptoss120-slice"
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no 120b slice at %s — build it from a partial openai/gpt-oss-120b download", ckpt)
	}

	m, err := Load(ckpt, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load 120b slice: %v", err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "gpt-oss" || a.gptoss == nil {
		t.Fatalf("arch = %q (gptoss=%v), want gpt-oss", a.Name, a.gptoss != nil)
	}
	if a.HiddenDim != 2880 || a.NumHeads != 64 || a.NumKVHeads != 8 {
		t.Errorf("geometry = %dh/%dq/%dkv, want 2880/64/8", a.HiddenDim, a.NumHeads, a.NumKVHeads)
	}
	// The thing 120b actually changes: four times the experts.
	if a.MoE == nil || a.MoE.NumExperts != 128 {
		t.Fatalf("experts = %v, want 128 — this is the shape 20b could not exercise", a.MoE)
	}
	for i := range a.NumLayers {
		if got := len(m.w.Layers[i].Experts); got != 128 {
			t.Fatalf("layer %d has %d experts, want 128", i, got)
		}
		if got := len(m.w.Layers[i].AttnSinks); got != a.NumHeads {
			t.Errorf("layer %d sinks = %d, want %d", i, got, a.NumHeads)
		}
	}

	// THE HIGHEST EXPERT INDEX. An off-by-N in the expert stride still yields a fully
	// populated, correctly-shaped WeightMat — it just contains a neighbour's weights, and
	// the error grows with the index, so expert 127 is where it shows and expert 0 is where
	// it hides. Distinctness across widely-separated experts is the cheap proxy: real
	// trained experts differ, and a stride that wraps or repeats produces duplicates.
	l0 := &m.w.Layers[0]
	rowOf := func(e int) []float32 {
		row := make([]float32, a.HiddenDim)
		l0.Experts[e].Gate.Row(0, row)
		return row
	}
	e0, e63, e127 := rowOf(0), rowOf(63), rowOf(127)
	for _, tc := range []struct {
		name string
		v    []float32
	}{{"expert 0", e0}, {"expert 63", e63}, {"expert 127", e127}} {
		nz, bad := 0, 0
		for _, x := range tc.v {
			if x != 0 {
				nz++
			}
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				bad++
			}
		}
		if bad > 0 {
			t.Errorf("%s gate row 0 has %d NaN/Inf values", tc.name, bad)
		}
		if nz == 0 {
			t.Errorf("%s gate row 0 is all zeros — the expert stride likely read past the tensor", tc.name)
		}
	}
	same := func(a, b []float32) bool {
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	if same(e0, e63) || same(e0, e127) || same(e63, e127) {
		t.Error("distinct expert indices returned IDENTICAL weights — the expert stride is wrong " +
			"(shapes stay valid under this bug, which is why it is checked by value)")
	}

	// A forward through the 4-layer slice: not a correctness oracle (no reference fits), but
	// it proves the whole loaded stack runs and produces finite, non-degenerate logits.
	cache := m.NewCache(8)
	var logits []float32
	for _, id := range []int{15496, 11, 616, 1438, 318} {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	finite, nz := 0, 0
	for _, v := range logits {
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			finite++
		}
		if v != 0 {
			nz++
		}
	}
	if finite != len(logits) || nz == 0 {
		t.Fatalf("logits: %d/%d finite, %d non-zero", finite, len(logits), nz)
	}
	t.Logf("gpt-oss-120b slice: %d layers x %d experts loaded; argmax %d over %d finite logits",
		a.NumLayers, a.MoE.NumExperts, argmax(logits), finite)
}
