//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestSplitKVCrossover finds the KV depth where split-KV overtakes attn_batched(M=1), so default-on
// can gate on it: split-KV loses at shallow ctx (3-launch overhead vs a cheap single-block attention)
// and wins at long ctx (occupancy). Measures both modes at several depths on the real 1.5B by toggling
// the resident's splitkvAttn flag. Prints the per-depth ratio; the threshold is the smallest depth
// where split-KV is a clear win. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSplitKVCrossover -v
func TestSplitKVCrossover(t *testing.T) {
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
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }

	measure := func(depth int, useSK bool) float64 {
		rf.splitkvAttn = useSK
		embs := make([][]float32, depth)
		for i := range embs {
			embs[i] = emb(i)
		}
		if _, e := rf.PrefillLast(embs, 0); e != nil {
			t.Fatalf("prefill(%d): %v", depth, e)
		}
		const steps = 96
		best := time.Hour
		for rep := 0; rep < 3; rep++ {
			t0 := time.Now()
			for i := 0; i < steps; i++ {
				if _, e := rf.ForwardArgmax(emb(70000+i), depth+i); e != nil {
					t.Fatalf("decode@%d: %v", depth, e)
				}
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(steps) / best.Seconds()
	}

	t.Logf("%-8s %-12s %-12s %-8s", "depth", "attn_batched", "split-KV", "ratio")
	for _, depth := range []int{128, 256, 384, 512, 768, 1024, 2048} {
		base := measure(depth, false)
		sk := measure(depth, true)
		t.Logf("%-8d %-12.1f %-12.1f %.3f×", depth, base, sk, sk/base)
	}
}
