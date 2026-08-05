//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillCrossover measures goinfer's TOTAL request time (batched prefill + 64 greedy decode) on
// the real dense qwen2.5-coder-1.5b, at the prompt lengths that bracket the Ollama crossover. Paired
// with the pinned-Ollama-0.5.7 numbers (docs/benchmarks.md §B2), it locates where goinfer's decode/
// overhead advantage stops covering its slower prefill. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillCrossover -v
func TestPrefillCrossover(t *testing.T) {
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
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()

	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = uint32(n*2654435761 + 1)
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}

	t.Logf("%-6s %12s %12s %12s", "promptN", "prefill", "decode64", "TOTAL")
	for _, n := range []int{128, 512, 1024, 2048} {
		embs := build(n)
		best := time.Hour
		var pf, dc time.Duration
		for rep := 0; rep < 3; rep++ {
			t0 := time.Now()
			lg, e := rf.PrefillLast(embs, 0)
			if e != nil {
				t.Fatalf("prefill n=%d: %v", n, e)
			}
			pfx := time.Since(t0)
			t1 := time.Now()
			cur := lg
			for i := 0; i < 64; i++ {
				tok := argmaxF(cur)
				l, e := rf.Forward(mc.EmbedResidentForTest(tok), n+i)
				if e != nil {
					t.Fatalf("decode: %v", e)
				}
				cur = l
			}
			dcx := time.Since(t1)
			if pfx+dcx < best {
				best, pf, dc = pfx+dcx, pfx, dcx
			}
		}
		t.Logf("%-6d %12v %12v %12v", n, pf, dc, best)
	}
}
