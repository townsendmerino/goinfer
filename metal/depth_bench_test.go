//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestZZ_metalDepthBench — the Metal side of benchmarks.md's greedy-decode-by-KV-depth curve, which
// was CUDA-only (§B6/§B7) and pre-P6a on Metal (§B3 was short-prompt). Anchored on qwen2.5-coder-1.5b
// with the CURRENT binary: builds one resident, warms the KV incrementally, and measures steady
// greedy decode (ForwardArgmax — the on-device-argmax path) at each depth. Depth axis matches the
// CUDA curve {128,512,2048,4000}; 4000 is near metalCtxCap=4096, so the top cell's measurement is
// clamped to stay inside the resident KV.
//
// NOTE ON SPLIT-KV: split-KV attention was never enabled on Metal (built, measured a regression,
// reverted — ollama-chase §A2-Metal), so unlike the CUDA 0.5B curve these rows carry NO split-KV
// caveat — this is the single attention path Metal ships.
//
// Opt-in timing diagnostic, not a gate.
func TestZZ_metalDepthBench(t *testing.T) {
	if os.Getenv("GOINFER_METAL_DEPTH_BENCH") == "" {
		t.Skip("Metal depth A/B (timing diagnostic, not a gate); set GOINFER_METAL_DEPTH_BENCH=1")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present: %v", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"}) // Metal W4A8 decode path (matches §B3)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}

	const tok = 100 // any valid token id; content is decode-timing-irrelevant
	pos := 0
	warmTo := func(d int) {
		for ; pos < d; pos++ {
			r.ForwardArgmax(tok, pos)
		}
	}
	warmTo(8) // initial warmup

	type row struct {
		depth int
		toks  float64
	}
	var rows []row
	for _, d := range []int{128, 512, 2048, 4000} {
		warmTo(d)
		// Measure steady decode: best (min) of several fixed-size batches. Clamp the batch so pos
		// never reaches metalCtxCap (the 4000 cell has only ~96 positions of headroom).
		iters, batches := 50, 5
		if room := (metalCtxCap - 8 - d) / batches; room < iters {
			iters = room
		}
		if iters < 5 {
			iters = 5
			batches = (metalCtxCap - 8 - d) / iters
		}
		best := time.Hour
		for b := 0; b < batches && pos+iters < metalCtxCap; b++ {
			t0 := time.Now()
			for i := 0; i < iters; i++ {
				r.ForwardArgmax(tok, pos)
				pos++
			}
			if el := time.Since(t0); el < best {
				best = el
			}
		}
		rows = append(rows, row{d, float64(iters) / best.Seconds()})
	}

	t.Logf("Metal greedy decode by KV depth — qwen2.5-coder-1.5b (W4A8), M1 Pro, current binary:")
	for i, rw := range rows {
		coef := ""
		if i > 0 {
			dMs := 1e3/rw.toks - 1e3/rows[i-1].toks
			usPer := dMs * 1e3 / float64(rw.depth-rows[i-1].depth)
			coef = fmt.Sprintf("  (%+.3f µs/pos vs previous)", usPer)
		}
		t.Logf("  depth %5d: %6.1f tok/s%s", rw.depth, rw.toks, coef)
	}
}
