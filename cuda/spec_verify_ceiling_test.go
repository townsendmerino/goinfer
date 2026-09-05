//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestSpecVerifyCeiling measures the go/no-go for D1 (speculative decoding) on the CUDA backend:
// the whole win hinges on the VERIFY of k drafted tokens being much cheaper than k sequential
// decodes. Decode is weight-bandwidth-bound (all weights read per token); a batched M=k forward
// reads the weights ONCE for k tokens (the prefill-batching win). This times, at a realistic KV
// depth, one batched PrefillLast(M=k) against k sequential Forward calls, for k∈{2,4,6,8}. The ratio
// is the spec-decode speedup CEILING (before the accept-rate discount): if batched-M=k ≈ k× cheaper,
// D1 is worth building; if it barely amortizes at small k, D1 can't win and we don't build it.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestSpecVerifyCeiling -v
func TestSpecVerifyCeiling(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	// GOINFER_CUDA_MODEL overrides the default fixture so the ceiling can be measured on the
	// pairing that matters rather than only on the 1.5B: the P10 projection needs the REAL
	// target (Qwen3-4B), where a fixed-size drafter is relatively cheaper.
	path := os.Getenv("GOINFER_CUDA_MODEL")
	if path == "" {
		path = modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
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

	const depth = 1024 // realistic mid-context; the batched-vs-sequential ratio is depth-insensitive
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = emb(i)
	}
	if _, e := rf.PrefillLast(context.Background(), warm, 0); e != nil {
		t.Fatalf("warm prefill: %v", e)
	}

	timeIt := func(f func()) time.Duration {
		best := time.Hour
		for range 5 {
			t0 := time.Now()
			f()
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}

	// single M=1 decode as the unit
	one := timeIt(func() {
		if _, e := rf.Forward(emb(5), depth); e != nil {
			t.Fatal(e)
		}
	})
	t.Logf("single decode (M=1) @depth %d: %.3f ms", depth, float64(one.Microseconds())/1000)
	t.Logf("%-4s %-14s %-16s %-12s %-s", "k", "batched(M=k)", "k×sequential", "cheaper by", "→ spec ceiling")
	for _, k := range []int{2, 4, 6, 8, 16} { // 16 = DFlash's block width (P10)
		ek := make([][]float32, k)
		for i := range ek {
			ek[i] = emb(100 + i)
		}
		batched := timeIt(func() {
			if _, e := rf.PrefillLast(context.Background(), ek, depth); e != nil {
				t.Fatal(e)
			}
		})
		seq := timeIt(func() {
			for i := range k {
				if _, e := rf.Forward(ek[i], depth+i); e != nil {
					t.Fatal(e)
				}
			}
		})
		ratio := float64(seq) / float64(batched) // how many k-decodes the batched verify costs
		t.Logf("%-4d %-14.3f %-16.3f %-12.2f× (ceiling ≈ %.2f× if 100%% accept)",
			k, float64(batched.Microseconds())/1000, float64(seq.Microseconds())/1000, ratio, ratio)
	}
	t.Log("Read: spec-decode replaces k sequential decodes with 1 batched verify + 1 draft (n-gram=free).")
	t.Log("If 'cheaper by' ≫ 1 at k=4-8, D1 can win (× accept rate); if ≈ 1, batching doesn't amortize → no D1.")
}
