//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDFlashDispatchAmortization settles the ONE assumption gate 3's draft term still rests on.
//
// The 6.6 ms draft in docs/spec/08 is `5.33 × per-layer(M=16)`, where per-layer comes from
// dividing the 36-layer target's batched verify by 36. That silently assumes **a 5-layer model
// costs 5/36ths of a 36-layer one** — i.e. that per-layer cost is independent of how many layers
// are in the stack. It need not be: a 36-layer forward has 36 dispatches to hide launch latency
// behind, and five have far less. If per-layer cost RISES as the stack shortens, the drafter is
// more expensive than 6.6 ms and every projected speedup drops.
//
// This is not hypothetical in this repo. It is the mechanism that landed the CUDA-graphs
// projection (1.4–1.7×) at a measured 1.01× — CPU dispatch overlaps GPU compute differently at
// different scales — and the mechanism behind Lever 2's "the draft was the wall, not the verify".
//
// METHOD: run the resident forward with the layer loop truncated (`r.nLayers`), at both M=1
// (`launchToken`) and M=16 (`PrefillLast`, the regime the block draft actually runs in), and
// compare per-layer cost across stack depths. The outputs are numerically meaningless — a
// truncated stack is not a model — but the TIMING is exactly the quantity in question, and the
// weights for every layer are already resident so no reload is involved.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashDispatchAmortization -v
func TestDFlashDispatchAmortization(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := mc.EmbedResidentForTest(1234 % (vocab - 1))

	const depth = 1024
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(warm, 0); e != nil {
		t.Fatalf("warm: %v", e)
	}
	full := r.nLayers
	defer func() { r.nLayers = full }() // restore, whatever happens

	block := make([][]float32, 16)
	for i := range block {
		block[i] = mc.EmbedResidentForTest((i*7919 + 13) % (vocab - 1))
	}

	best := func(f func() error) time.Duration {
		b := time.Hour
		for range 7 {
			t0 := time.Now()
			if err := f(); err != nil {
				t.Fatalf("run: %v", err)
			}
			if d := time.Since(t0); d < b {
				b = d
			}
		}
		return b
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

	t.Logf("target has %d layers; probing truncated stacks at depth %d", full, depth)
	t.Logf("%7s | %10s %12s | %10s %12s", "layers", "M=1 ms", "per-layer", "M=16 ms", "per-layer")
	type row struct {
		n         int
		pl1, pl16 float64
	}
	var rows []row
	for _, n := range []int{5, 8, 12, 18, 36} {
		if n > full {
			continue
		}
		r.nLayers = n
		d1 := best(func() error {
			return r.do(func() error {
				if e := r.launchToken(emb, depth, false); e != nil { // no head: the drafter has none
					return e
				}
				return r.stream.Sync()
			})
		})
		d16 := best(func() error {
			_, e := r.PrefillLast(block, depth)
			return e
		})
		pl1, pl16 := ms(d1)/float64(n), ms(d16)/float64(n)
		rows = append(rows, row{n, pl1, pl16})
		t.Logf("%7d | %10.3f %12.4f | %10.3f %12.4f", n, ms(d1), pl1, ms(d16), pl16)
	}
	r.nLayers = full

	// The verdict: per-layer cost at 5 layers against per-layer cost at the full stack. The
	// 6.6 ms draft assumes this ratio is 1.0.
	var short, long row
	for _, x := range rows {
		if x.n == 5 {
			short = x
		}
		if x.n == full {
			long = x
		}
	}
	if short.n == 0 || long.n == 0 {
		t.Skip("need both a 5-layer and a full-stack point")
	}
	t.Logf("")
	t.Logf("per-layer INFLATION at 5 layers vs %d:  M=1 %.2fx   M=16 %.2fx",
		full, short.pl1/long.pl1, short.pl16/long.pl16)
	t.Logf("=> drafter (5 layers + fc) at M=16: %.2f ms  (the 6.6 ms projection assumes %.2f)",
		5.33*short.pl16, 5.33*long.pl16)
}
