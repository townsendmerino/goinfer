//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// BenchmarkMoERoute is D-P04's probe (docs/audit-metal-2026-09-30.md, "D-P04"): the GPU time of one (1,1) moe_route /
// route_gptoss dispatch, net of a (1,1) copy_u32 dispatched the same number of times in the same serial encoder, so
// what is left is the router's own one-thread work rather than dispatch overhead. The audit's kill line is 5 us per
// layer. Shapes: Qwen1.5-MoE (60 experts, top 4), OLMoE (64, 8), Qwen3-MoE and Gemma 4 (128, 8, renormalised),
// DeepSeek-V3 (256, 8, sigmoid, bias, 8 groups keep 4), gpt-oss-20b (32, 4).
func BenchmarkMoERoute(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			b.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	pRoute, pGptOss, pCopy := pipe("moe_route"), pipe("route_gptoss"), pipe("copy_u32")
	type shape struct {
		name                 string
		nE, k, sigmoid, norm int
		scale                float32
		nGroup, topkGroup    int
		gptoss, bias         bool
	}
	shapes := []shape{
		{name: "qwen15moe-60x4", nE: 60, k: 4},
		{name: "olmoe-64x8", nE: 64, k: 8},
		{name: "qwen3moe-128x8", nE: 128, k: 8, norm: 1},
		{name: "deepseekv3-256x8", nE: 256, k: 8, sigmoid: 1, norm: 1, scale: 2.5, nGroup: 8, topkGroup: 4, bias: true},
		{name: "gptoss-32x4", nE: 32, k: 4, gptoss: true, bias: true},
	}
	const reps = 64
	q := d.NewCommandQueue()
	for _, s := range shapes {
		b.Run(s.name, func(b *testing.B) {
			rng := rand.New(rand.NewSource(int64(s.nE*31 + s.k)))
			logits, bias := make([]float32, s.nE), make([]float32, s.nE)
			for i := range logits {
				logits[i] = float32(rng.NormFloat64())
				if s.bias {
					bias[i] = float32(rng.NormFloat64()) * 0.1
				}
			}
			dL, dB := NewBufferFloats(d, logits), NewBufferFloats(d, bias)
			dIdx, dW := d.NewBufferLen(s.k), d.NewBufferLen(s.k)
			uNE, uK := NewBufferU32(d, uint32(s.nE)), NewBufferU32(d, uint32(s.k))
			uSig, uNorm := NewBufferU32(d, uint32(s.sigmoid)), NewBufferU32(d, uint32(s.norm))
			uScale := NewBufferFloats(d, []float32{s.scale})
			uNG, uTG := NewBufferU32(d, uint32(s.nGroup)), NewBufferU32(d, uint32(s.topkGroup))
			dSrc, dDst := NewBufferUint32s(d, []uint32{1}), d.NewBufferLen(1)
			time1 := func(route bool) time.Duration {
				e := q.Begin()
				for range reps {
					switch {
					case !route:
						e.Dispatch(pCopy, 1, 1, dSrc, dDst)
					case s.gptoss:
						e.Dispatch(pGptOss, 1, 1, dL, dB, dIdx, dW, uNE, uK)
					default:
						e.Dispatch(pRoute, 1, 1, dL, dB, dIdx, dW, uNE, uK, uSig, uNorm, uScale, uNG, uTG)
					}
				}
				t0 := time.Now()
				e.End()
				return time.Since(t0)
			}
			for range 4 {
				time1(true)
				time1(false)
			}
			b.ResetTimer()
			bestR, bestC := time.Hour, time.Hour
			for i := range b.N {
				// alternate which arm goes first, so a drift in clock or load lands on both
				for _, route := range []bool{i%2 == 0, i%2 != 0} {
					dt := time1(route)
					if route {
						bestR = min(bestR, dt)
					} else {
						bestC = min(bestC, dt)
					}
				}
			}
			net := float64(bestR-bestC) / reps / float64(time.Microsecond)
			b.ReportMetric(float64(bestR)/reps/float64(time.Microsecond), "us/route(best)")
			b.ReportMetric(float64(bestC)/reps/float64(time.Microsecond), "us/copy(best)")
			b.ReportMetric(net, "us/route-net")
			if testing.Verbose() {
				fmt.Printf("D-P04 %s: route %.2f us, copy %.2f us, net %.2f us per dispatch (best of %d, %d per buffer)\n",
					s.name, float64(bestR)/reps/1e3, float64(bestC)/reps/1e3, net, b.N, reps)
			}
		})
	}
}
