package decoder

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// The decode-matmul parallelism threshold sweep. int4ParThreshold (weightmat.go, 1<<20)
// and DefaultDecodeParallelThreshold (tune.go, 300K) are the fan-out crossovers, in MACs,
// below which a decode matmul runs SERIAL. Both were tuned on a Ryzen 7 3700X (8 desktop
// cores); the M1 Pro (6 P + 2 E, very different memory latency) is Phase 5's rig and was
// never swept — a wrong threshold understates any benchmark we eventually publish. This
// isolates the crossover from the model: it drives the SAME kernels (MatmulBTW4A8Into /
// MatmulBTW8A8Into) matmul() dispatches, at Gemma-4-26B-A4B's real decode shapes (M=1),
// across a bracket of thresholds, with a per-call Workspace exactly as the forward does.
//
// Run + read (per-shape ns/op; lower is better; the winning threshold is the largest one
// whose ns/op is still ~serial-beating for every shape it must fan out):
//
//	go test ./decoder -run '^$' -bench 'ParThresholdSweep' -benchtime 200ms
//
// Interpreting it: for each shape the ns/op is flat until the threshold rises ABOVE that
// shape's MAC count, at which point it jumps to the serial cost. The optimum is the
// threshold that keeps every shape you want parallel below the jump while leaving truly
// tiny ops (which over-parallelize — thread spawn > work) serial. Compare the M1 Pro
// optimum against the constants; if it differs materially, the constant wants to be
// per-platform (GOOS/arch or GOMAXPROCS-derived), not a universal default.

// gemma4-26b-a4b decode matmul shapes at M=1 (K=cols/in, N=rows/out), MACs = K*N:
//   down 1.98M is the SMALLEST — the one int4ParThreshold (1.05M) must let through;
//   attn 11.5M the largest. hidden=2816, moe_inter=704, dense_inter=2112, nH*hd(global)=4096.
var decodeShapes = []struct {
	name string
	K, N int // K = input dim (dot length), N = output rows (partitioned in 8-wide groups)
}{
	{"down_1.98M", 704, 2816},
	{"gate_up_3.96M", 2816, 1408},
	{"dense_5.9M", 2816, 2112},
	{"attn_11.5M", 2816, 4096},
}

// The bracket straddles both constants (300K, 1<<20) and the down/gate_up/dense/attn MAC
// counts, plus the endpoints: 0 fans out everything (measures the over-parallelize cost)
// and 1<<24 is aikit's serial default (measures the all-serial cost the constants fixed).
var sweepThresholds = []int{0, 300_000, 700_000, 1 << 20, 1_500_000, 2_000_000, 4_000_000, 6_000_000, 12_000_000, 1 << 24}

func BenchmarkInt4ParThresholdSweep(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, sh := range decodeShapes {
		// Build the int4 weight [N, K] and f32 activation [1, K] once — same quantizer
		// matmul() feeds (linalg.QuantizeInt4), so the codes/scales are production-shaped.
		wf := make([]float32, sh.N*sh.K)
		for i := range wf {
			wf[i] = float32(rng.NormFloat64()) * 0.05
		}
		wm := linalg.QuantizeInt4(wf, sh.N, sh.K, int4GroupSize)
		q4, q4s, group, ok := wm.Int4()
		if !ok {
			b.Fatalf("%s: QuantizeInt4 did not yield an int4 WeightMat", sh.name)
		}
		a := make([]float32, sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64()) * 0.5
		}
		dst := make([]float32, sh.N)
		for _, thr := range sweepThresholds {
			b.Run(fmt.Sprintf("%s/thr=%d", sh.name, thr), func(b *testing.B) {
				var ws linalg.Workspace
				ws.SetThreshold(thr)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					linalg.MatmulBTW4A8Into(&ws, a, q4, q4s, dst, 1, sh.K, sh.N, group)
				}
			})
		}
	}
}

func BenchmarkInt8ParThresholdSweep(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	for _, sh := range decodeShapes {
		wf := make([]float32, sh.N*sh.K)
		for i := range wf {
			wf[i] = float32(rng.NormFloat64()) * 0.05
		}
		wm := linalg.QuantizeInt8(wf, sh.N, sh.K, true) // w8a8
		q8, scales, _, ok := wm.Int8()
		if !ok {
			b.Fatalf("%s: QuantizeInt8 did not yield an int8 WeightMat", sh.name)
		}
		a := make([]float32, sh.K)
		for i := range a {
			a[i] = float32(rng.NormFloat64()) * 0.5
		}
		dst := make([]float32, sh.N)
		for _, thr := range sweepThresholds {
			b.Run(fmt.Sprintf("%s/thr=%d", sh.name, thr), func(b *testing.B) {
				var ws linalg.Workspace
				ws.SetThreshold(thr)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					linalg.MatmulBTW8A8Into(&ws, a, q8, scales, dst, 1, sh.K, sh.N)
				}
			})
		}
	}
}
