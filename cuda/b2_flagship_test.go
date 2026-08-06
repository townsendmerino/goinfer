//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestB2DenseFlagship measures a fitting dense flagship (qwen2.5-7B int4, resident on the 2070 SUPER
// with real KV headroom) end-to-end, §B2 method: TTFT at 128/512/2048 + all-in and decode-only tok/s,
// best of 3 warm with the first discarded. Pair the goinfer column with pinned Ollama 0.5.7 for the
// published §B2 row. The claim here is the honest one: faster decode, prefill within a stated multiple,
// crossover at a measured prompt length.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestB2DenseFlagship -v -timeout 900s
func TestB2DenseFlagship(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 7B model)")
	}
	path := modelPath("qwen2.5-7b-instruct-q4_k_m.gguf")
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
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("7B did not go cuda-resident")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	if _, e := rf.PrefillLast(make([][]float32, 8), 0); e == nil {
		t.Logf("PrefillLast accepts (batched)") // sanity: dense qwen2 is covered
	}

	build := func(n int) [][]float32 {
		embs := make([][]float32, n)
		var s uint32 = uint32(n*2654435761 + 1)
		for i := range embs {
			s = s*1664525 + 1013904223
			embs[i] = append([]float32(nil), mc.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
		}
		return embs
	}
	const gen = 64
	t.Logf("%-6s %12s %12s %14s %14s", "promptN", "TTFT", "decode64", "all-in tok/s", "decode tok/s")
	for _, n := range []int{128, 512, 2048} {
		embs := build(n)
		var bestPre, bestDec time.Duration
		best := time.Hour
		for rep := 0; rep < 3; rep++ { // best of 3, first (rep 0) is the warm-up and discarded by "best"
			t0 := time.Now()
			lg, e := rf.PrefillLast(embs, 0)
			if e != nil {
				t.Fatalf("prefill n=%d: %v", n, e)
			}
			pre := time.Since(t0)
			t1 := time.Now()
			cur := lg
			for i := 0; i < gen; i++ {
				tok := argmaxF(cur)
				l, e := rf.Forward(mc.EmbedResidentForTest(tok), n+i)
				if e != nil {
					t.Fatalf("decode: %v", e)
				}
				cur = l
			}
			dec := time.Since(t1)
			if rep > 0 && pre+dec < best {
				best, bestPre, bestDec = pre+dec, pre, dec
			}
		}
		allIn := float64(gen) / best.Seconds()
		decOnly := float64(gen) / bestDec.Seconds()
		t.Logf("%-6d %12v %12v %14.1f %14.1f", n, bestPre, bestDec, allIn, decOnly)
	}
}
