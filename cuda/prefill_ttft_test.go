//go:build cuda

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillTTFT measures the batched PrefillLast vs the sequential ForwardNoLogits loop on a real
// dense model, at the prompt lengths that bracket the Ollama crossover (128/512/2048). It is the
// milestone-2 speedup number: goinfer's sequential prefill reads every weight once per prompt token
// (weight-bandwidth-bound), so its TTFT grows ~linearly; the batched path reads each weight once for
// all M tokens. Heavy (loads a 1.5B model); gated on GOINFER_HEAVY_TESTS + a GPU.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillTTFT -v
func TestPrefillTTFT(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	const path = "/home/francis/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident is not *cudaResident")
	}
	if !rf.prefillReady {
		t.Fatal("batched prefill kernels did not load")
	}
	_, _, _, _, _, _, vocab := mc.Dims()

	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = 12345
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}
	// Confirm the batched path accepts this dense model rather than declining.
	if _, e := rf.PrefillLast(build(8), 0); e != nil {
		t.Fatalf("PrefillLast declined a dense qwen2.5 model: %v", e)
	}

	median := func(f func(), reps int) time.Duration {
		ds := make([]time.Duration, reps)
		for i := range ds {
			t0 := time.Now()
			f()
			ds[i] = time.Since(t0)
		}
		// simple min (best-of) — prefill is compute/bandwidth bound, min is the cleanest signal
		best := ds[0]
		for _, d := range ds {
			if d < best {
				best = d
			}
		}
		return best
	}

	t.Logf("%-6s %12s %12s %8s", "N", "sequential", "batched", "speedup")
	for _, n := range []int{128, 512, 2048} {
		embs := build(n)
		seq := median(func() {
			for i := 0; i < n-1; i++ {
				if e := rf.ForwardNoLogits(embs[i], i); e != nil {
					t.Fatalf("seq pos %d: %v", i, e)
				}
			}
			if _, e := rf.Forward(embs[n-1], n-1); e != nil {
				t.Fatalf("seq last: %v", e)
			}
		}, 3)
		bat := median(func() {
			if _, e := rf.PrefillLast(embs, 0); e != nil {
				t.Fatalf("batched n=%d: %v", n, e)
			}
		}, 3)
		t.Logf("%-6d %12v %12v %7.2fx", n, seq, bat, float64(seq)/float64(bat))
	}
}
