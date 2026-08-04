//go:build cuda

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestDecodeDepthThroughput measures real decode tok/s at a shallow (128) and a deep (2048) KV
// depth on the 1.5B. §B2 recorded the deep number collapsing to ~97 tok/s vs ~221 shallow — the
// long-context deficit ncu traced to the uncoalesced glue decode-attention K read. This is the
// A/B instrument for the coalesced (attn_batched M=1) decode swap: run it on the coalesced build,
// then `git stash` resident.go and run it on the glue build, to attribute the recovery.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestDecodeDepthThroughput -v
func TestDecodeDepthThroughput(t *testing.T) {
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
	emb := func(i int) []float32 { return mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }

	measureAt := func(depth int) float64 {
		// Prime the KV to `depth` via batched prefill (fast; not the thing under test).
		embs := make([][]float32, depth)
		for i := range embs {
			embs[i] = emb(i)
		}
		if _, err := rf.PrefillLast(embs, 0); err != nil {
			t.Fatalf("prefill(%d): %v", depth, err)
		}
		const steps = 128
		best := time.Hour
		for rep := 0; rep < 3; rep++ {
			t0 := time.Now()
			for i := 0; i < steps; i++ {
				if _, err := rf.ForwardArgmax(emb(100000+i), depth+i); err != nil {
					t.Fatalf("decode@%d: %v", depth, err)
				}
			}
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return float64(steps) / best.Seconds()
	}

	sh := measureAt(128)
	dp := measureAt(2048)
	t.Logf("decode tok/s (1.5B, greedy): depth128 %.1f  |  depth2048 %.1f  →  deep/shallow %.2f×", sh, dp, dp/sh)
}
