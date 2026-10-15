//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// GV-C(b) and G7-C (docs/measurements/kv2-cuda-2026-10-15.md): a copy allocates nothing on the device, and what it costs.
// The tiny case grades the allocation; the real cases (GOINFER_HEAVY_TESTS=1: the 1.5B; GOINFER_KV2_COPYCOST_7B=1 adds the
// 7B) also time 20 copies of 18,176 rows, the sharing gate's hit, each to the stream's end (CopyKV synchronises).

func kv2CopyCost(t *testing.T, g *kv2Rig, rows, reps int) {
	t.Helper()
	var seen, calls int
	allocSeenForTest = func(b int) { seen += b; calls++ }
	t.Cleanup(func() { allocSeenForTest = nil })
	free0, _, _ := g.r.dev.Context().MemInfo()
	times := make([]time.Duration, 0, reps)
	for range reps {
		t0 := time.Now()
		if err := g.r.CopyKV(0, 1, 0, rows); err != nil {
			t.Fatalf("CopyKV [0, %d): %v", rows, err)
		}
		times = append(times, time.Since(t0))
	}
	free1, _, _ := g.r.dev.Context().MemInfo()
	allocSeenForTest = nil
	if seen != 0 || calls != 0 {
		t.Errorf("%d copies allocated %d device buffers, %d bytes; a copy must allocate nothing", reps, calls, seen)
	}
	var bytes int
	for _, b := range g.bufs() {
		bytes += rows * b.row
	}
	slices.Sort(times)
	fmt.Fprintf(os.Stderr, "[kv2-g7] %d rows, %d layers, KV f16=%v: %.1f MB a copy; median %v, min %v, max %v over %d; free VRAM %d -> %d MB (reported, not graded)\n",
		rows, len(g.r.layers), g.r.kvF16, float64(bytes)/1e6, times[len(times)/2], times[0], times[len(times)-1], reps, free0>>20, free1>>20)
}

func TestKV2_copyAllocatesNothing_tiny(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	g := kv2Load(t, dir, decoder.Options{ResidentKVSlots: 2, ResidentContext: 2048, KVPrecision: "f16"})
	kv2CopyCost(t, g, 1024, 20)
}

func TestKV2_copyCost_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B, and the 7B with GOINFER_KV2_COPYCOST_7B=1)")
	}
	const rows = 18176
	models := []string{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"}
	if os.Getenv("GOINFER_KV2_COPYCOST_7B") != "" {
		models = append(models, "qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	for _, name := range models {
		path := modelPath(name)
		requireDeviceAndFile(t, path)
		kvs := []string{"f16", "f32"}
		if name != models[0] {
			kvs = kvs[:1] // two slots of 18,240 positions at f32 do not fit beside the 7B's weights on an 8 GB card
		}
		for _, kv := range kvs {
			t.Run(name+"/"+kv, func(t *testing.T) {
				g := kv2Load(t, path, decoder.Options{ResidentKVSlots: 2, ResidentContext: rows + 64, KVPrecision: kv})
				kv2CopyCost(t, g, rows, 20)
			})
		}
	}
}
