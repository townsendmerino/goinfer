//go:build cuda && goinfer_testhooks

package cuda

import (
	"fmt"
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

// TestGraphsDecode26B is the one measurement CUDA graphs were owed before deciding their fate
// (docs/cuda-graphs-investigation.md: ~1.01x on the dense 1.5B; the 26B MoE was never measured). On
// the C′ path graphs are not free to turn on: they force the DMA overlap off (a captured segment
// cannot wait per miss) and block compute-time LoRA. So the comparison is the trade itself —
// today's default (graphs off, overlap on) against graphs on (overlap off) — not graphs vs a
// strawman. Greedy decode feeds each argmax back, so MoE routing and the expert cache see a real
// continuation. Arms are separate loads (two 26B residents do not fit 8 GB), interleaved ABBA.
//
// Pre-registered decision rule (2026-09-24, before running): graphs >= 1.05x default → worth
// keeping; <= 1.00x → remove graphs; between → ambiguous, back to the owner.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestGraphsDecode26B -v -timeout 30m
func TestGraphsDecode26B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the real 26B four times)")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		path = modelPath("gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	start := time.Now()

	const warm, steps = 32, 128
	run := func(graphs bool, arm int) (float64, []int) {
		if graphs {
			t.Setenv("GOINFER_CUDA_GRAPHS", "1")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
		} else {
			t.Setenv("GOINFER_CUDA_GRAPHS", "")
			t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "")
		}
		fmt.Fprintf(os.Stderr, "[graphs26b %s] arm %d/4 graphs=%v: loading (elapsed %s)\n",
			time.Now().Format("15:04:05"), arm, graphs, time.Since(start).Round(time.Second))
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf == nil {
			t.Fatalf("cuda resident declined the 26B: %s", m.ResidentDecline())
		}
		if rf.graphs != graphs {
			t.Fatalf("graphs=%v requested, resident has %v", graphs, rf.graphs)
		}
		if graphs && rf.overlap {
			t.Fatal("graphs on but the DMA overlap is still on — the trade this measures is not in effect")
		}
		_, _, _, _, _, _, vocab := m.Dims()
		id := 2
		for i := range warm { // a fixed pseudo-random prompt, teacher-forced
			next, err := rf.ForwardArgmax(m.EmbedResidentForTest((i*2654435761+7)%(vocab-1)), i)
			if err != nil {
				t.Fatalf("warm: %v", err)
			}
			id = next
		}
		out := make([]int, steps)
		t0 := time.Now()
		for i := range steps { // greedy continuation: routing follows the model's own output
			next, err := rf.ForwardArgmax(m.EmbedResidentForTest(id), warm+i)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			out[i], id = next, next
		}
		tps := float64(steps) / time.Since(t0).Seconds()
		fmt.Fprintf(os.Stderr, "[graphs26b %s] arm %d/4 graphs=%v overlap=%v slots=%d: %.2f tok/s\n",
			time.Now().Format("15:04:05"), arm, graphs, rf.overlap, rf.cacheSlots, tps)
		return tps, out
	}

	d1, outD := run(false, 1)
	g1, outG := run(true, 2)
	g2, _ := run(true, 3)
	d2, _ := run(false, 4)
	def, gr := (d1+d2)/2, (g1+g2)/2
	same := len(outD) == len(outG)
	for i := range outD {
		if outD[i] != outG[i] {
			same = false
			break
		}
	}
	verdict := "AMBIGUOUS — back to the owner"
	switch {
	case gr >= 1.05*def:
		verdict = "KEEP (>= 1.05x)"
	case gr <= def:
		verdict = "REMOVE (<= 1.00x)"
	}
	t.Logf("26B C′ decode tok/s, ABBA: default(overlap on) %.2f / %.2f, graphs(overlap off) %.2f / %.2f → graphs/default %.3fx; token streams identical=%v; rule: %s",
		d1, d2, g1, g2, gr/def, same, verdict)
}
