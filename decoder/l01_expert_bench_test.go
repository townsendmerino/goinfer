package decoder

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// L01 design pass (docs/task-l01-hybrid-moe-cpu-gpu.md §9 item 1): the isolated CPU
// expert-compute measurement the design doc's §2 derived estimate (1.98ms/expert, folded in
// with attention/router/shared-expert cost from a whole-decode-step measurement) needs before
// trusting it for a funding decision. Geometry is gemma-4-26B's REAL values, probed directly
// via Model.Dims()/MoEResidentParams() on the real checkpoint (nobara-pc, 2026-09-10) rather
// than guessed: hidden=2816, expert inter=704, nExperts=128, topK=8, no shared expert.
// Synthetic int4 weights via quantizeWM (the SAME repackW4A8IfEligible/maybeF16RoundInt4Scales
// chain production loading uses — not a bare linalg.QuantizeInt4, and not f32, since either
// would exercise a different code path than the "existing W4A8/W8A8... expert kernels" the
// audit's L-01 mechanism specifically names.
const (
	l01Hidden = 2816
	l01Inter  = 704
)

func l01SyntheticExpert(rng *rand.Rand) *expertWeights {
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	q4 := func(data []float32, rows, cols int) linalg.WeightMat {
		return quantizeWM(linalg.WrapF32(data, rows, cols), quantInt4)
	}
	return &expertWeights{
		Gate: q4(rnd(l01Inter*l01Hidden), l01Inter, l01Hidden),
		Up:   q4(rnd(l01Inter*l01Hidden), l01Inter, l01Hidden),
		Down: q4(rnd(l01Hidden*l01Inter), l01Hidden, l01Inter),
	}
}

// BenchmarkL01_singleExpert isolates ONE expert's SwiGLU MLP — the direct measurement the
// design doc's derived 1.98ms estimate needs to confirm or refute.
func BenchmarkL01_singleExpert(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	ex := l01SyntheticExpert(rng)
	be := &cpuBackend{}
	h := make([]float32, l01Hidden)
	for i := range h {
		h[i] = rng.Float32()*2 - 1
	}
	dst := make([]float32, l01Hidden)
	gate := make([]float32, l01Inter)
	up := make([]float32, l01Inter)

	b.ResetTimer()
	for range b.N {
		swiGLUExpert(ex, h, dst, l01Inter, be, gate, up)
	}
}

// l01Sequential times n experts computed ONE AT A TIME through shared scratch — the same
// pattern moeMLP's real sequential loop uses today (decoder/mlp.go: "the experts run
// sequentially, so k pairs were never simultaneously live"). The do-nothing-extra baseline for
// however many experts a layer happened to miss.
func l01Sequential(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(2))
	experts := make([]*expertWeights, n)
	for i := range experts {
		experts[i] = l01SyntheticExpert(rng)
	}
	be := &cpuBackend{}
	h := make([]float32, l01Hidden)
	for i := range h {
		h[i] = rng.Float32()*2 - 1
	}
	dst := make([]float32, l01Hidden)
	gate := make([]float32, l01Inter)
	up := make([]float32, l01Inter)

	b.ResetTimer()
	for range b.N {
		for _, ex := range experts {
			swiGLUExpert(ex, h, dst, l01Inter, be, gate, up)
		}
	}
}

func BenchmarkL01_sequential1(b *testing.B) { l01Sequential(b, 1) }
func BenchmarkL01_sequential2(b *testing.B) { l01Sequential(b, 2) }
func BenchmarkL01_sequential4(b *testing.B) { l01Sequential(b, 4) }
func BenchmarkL01_sequential5(b *testing.B) { l01Sequential(b, 5) }
func BenchmarkL01_sequential8(b *testing.B) { l01Sequential(b, 8) }

// l01Parallel times n experts computed CONCURRENTLY, one goroutine per expert, own scratch
// each — the restructuring moeMLP's sequential design does not do today. This is the design
// doc's open question (docs/task-l01-hybrid-moe-cpu-gpu.md §4/§8): does dividing the machine's
// core pool across the missed-expert set beat sequential compute at the high-m tail (m=5-8,
// ~6.5% of decisions per the trace-distribution finding), given each expert's own matmul
// already claims most/all cores via aikit's parallelCols (so this deliberately creates NESTED
// parallelism/oversubscription — that contention, if any, is exactly what's being measured,
// not an artifact to avoid).
func l01Parallel(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(3))
	experts := make([]*expertWeights, n)
	for i := range experts {
		experts[i] = l01SyntheticExpert(rng)
	}
	be := &cpuBackend{}
	h := make([]float32, l01Hidden)
	for i := range h {
		h[i] = rng.Float32()*2 - 1
	}

	b.ResetTimer()
	for range b.N {
		var wg sync.WaitGroup
		for _, ex := range experts {
			wg.Add(1)
			go func(ex *expertWeights) {
				defer wg.Done()
				dst := make([]float32, l01Hidden)
				gate := make([]float32, l01Inter)
				up := make([]float32, l01Inter)
				swiGLUExpert(ex, h, dst, l01Inter, be, gate, up)
			}(ex)
		}
		wg.Wait()
	}
}

func BenchmarkL01_parallel2(b *testing.B) { l01Parallel(b, 2) }
func BenchmarkL01_parallel4(b *testing.B) { l01Parallel(b, 4) }
func BenchmarkL01_parallel5(b *testing.B) { l01Parallel(b, 5) }
func BenchmarkL01_parallel8(b *testing.B) { l01Parallel(b, 8) }

// l01Concurrent simulates `streams` INDEPENDENT decode streams each offloading
// expertsPerStream missed experts to CPU at the SAME time — the multi-tenant contention
// question docs/task-l01-hybrid-moe-cpu-gpu.md §5/§9 named as unmeasured: does a second
// concurrent request degrade "always offload" badly enough to matter? Measures wall-clock for
// ALL streams' goroutines to finish (the tail, since that's what either stream's own latency
// depends on), not aggregate throughput.
func l01Concurrent(b *testing.B, streams, expertsPerStream int) {
	rng := rand.New(rand.NewSource(4))
	allExperts := make([][]*expertWeights, streams)
	for s := range allExperts {
		allExperts[s] = make([]*expertWeights, expertsPerStream)
		for i := range allExperts[s] {
			allExperts[s][i] = l01SyntheticExpert(rng)
		}
	}
	be := &cpuBackend{}
	h := make([]float32, l01Hidden)
	for i := range h {
		h[i] = rng.Float32()*2 - 1
	}

	b.ResetTimer()
	for range b.N {
		var wg sync.WaitGroup
		for _, streamExperts := range allExperts {
			for _, ex := range streamExperts {
				wg.Add(1)
				go func(ex *expertWeights) {
					defer wg.Done()
					dst := make([]float32, l01Hidden)
					gate := make([]float32, l01Inter)
					up := make([]float32, l01Inter)
					swiGLUExpert(ex, h, dst, l01Inter, be, gate, up)
				}(ex)
			}
		}
		wg.Wait()
	}
}

// BenchmarkL01_concurrent2streams8 is the worst-case pairing: two streams, each at m=8 (the
// rarest, most expensive miss count per docs/measurements/g33-routing-trace.json's own
// distribution), offloading at exactly the same instant.
func BenchmarkL01_concurrent2streams8(b *testing.B) { l01Concurrent(b, 2, 8) }
func BenchmarkL01_concurrent3streams8(b *testing.B) { l01Concurrent(b, 3, 8) }

// BenchmarkL01_concurrent2streamsMean pairs two streams at m=2, the closest integer to G33's
// own measured mean (1.915) — the realistic case, not the worst case.
func BenchmarkL01_concurrent2streamsMean(b *testing.B) { l01Concurrent(b, 2, 2) }
