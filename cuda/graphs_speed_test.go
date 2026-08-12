//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGraphsDecodeSpeedup measures the real decode tok/s of graph replay vs live launch on a real
// model, validating the ~1.4–1.7× dispatch-elimination prediction end-to-end through the safe-gate.
// Graphs are enabled via the UNSAFE override (this idle box is DEFAULT compute mode; with no churn,
// replay is bit-exact — the self-test in admitGraphs confirms it). Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestGraphsDecodeSpeedup -v
func TestGraphsDecodeSpeedup(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}

	measure := func(graphs bool) (float64, bool) {
		if graphs {
			t.Setenv("GOINFER_CUDA_GRAPHS", "1")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
		} else {
			t.Setenv("GOINFER_CUDA_GRAPHS", "")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
		}
		mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer mc.Close()
		rf := mc.ResidentForwardForTest().(*cudaResident)
		if rf.graphs != graphs {
			t.Fatalf("graphs=%v requested, resident has %v", graphs, rf.graphs)
		}
		_, _, _, _, _, _, vocab := mc.Dims()
		emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }
		// warm + establish KV over a short prefix
		for i := range 16 {
			if _, err := rf.ForwardArgmax(emb(i), i); err != nil {
				t.Fatalf("warm: %v", err)
			}
		}
		const steps = 256
		best := time.Hour
		for range 3 {
			t0 := time.Now()
			for i := range steps {
				if _, err := rf.ForwardArgmax(emb(1000+i), 16+i); err != nil {
					t.Fatalf("decode: %v", err)
				}
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(steps) / best.Seconds(), rf.graphs
	}

	base, _ := measure(false)
	g, on := measure(true)
	if !on {
		t.Fatal("graphs did not enable under the UNSAFE override")
	}
	t.Logf("decode tok/s (1.5B, greedy): live %.1f  |  graphs %.1f  →  %.2f×", base, g, g/base)
}
