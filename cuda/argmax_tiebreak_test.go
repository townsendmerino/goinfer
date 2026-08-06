//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestArgmaxTieBreak is the C-14 gate: the on-device argmax_reduce must return the LOWEST index on an
// exact tie, matching the CPU reference (decoder.argmax / argmaxF, strict >). The tree reduction pairs
// by thread index, and a left thread's strided max can sit at a higher absolute index than the right
// thread's — so before the fix an exact tie returned the higher index, silently diverging greedy decode
// from the CPU golden. Crafted ties exercise the multi-thread reduction the real forward almost never
// hits. Heavy (needs a resident for the compiled kernel + buffers).
func TestArgmaxTieBreak(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 0.5B model for the compiled argmax kernel)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("model did not go resident")
	}

	// build a V-length logits vector that is `lo` everywhere except `hi` at the given indices.
	craft := func(V int, at ...int) []float32 {
		v := make([]float32, V)
		for i := range v {
			v[i] = -1e9
		}
		for _, i := range at {
			v[i] = 0.0
		}
		return v
	}
	cases := []struct {
		name string
		V    int
		at   []int
		want int
	}{
		{"tie 1 vs 256 (left thread holds the higher index)", 512, []int{1, 256}, 1},
		{"tie 5 vs 300 across threads", 1000, []int{300, 5}, 5},
		{"three-way tie", 800, []int{600, 40, 271}, 40},
		{"all-equal picks 0", 300, allIdx(300), 0},
		{"single max control", 700, []int{423}, 423},
	}
	for _, c := range cases {
		got, err := rf.ArgmaxForTest(craft(c.V, c.at...))
		if err != nil {
			t.Fatalf("%s: ArgmaxForTest: %v", c.name, err)
		}
		if got != c.want {
			t.Errorf("%s: argmax = %d, want %d (lowest tied index)", c.name, got, c.want)
		}
	}
}

func allIdx(n int) []int {
	a := make([]int, n)
	for i := range a {
		a[i] = i
	}
	return a
}
