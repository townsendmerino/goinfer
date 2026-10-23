//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestKSD2_steelHeadDims is D2-G2 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D2 gates"): attention_prefill_steel's
// head-dim 64, 80 and 96 instances and attention_prefill_fused, each against an f64 attention over the same half-precision
// inputs, at random q, k and v; M = 100 and 300; start 0 and 37; window 0 and 64; 14 query heads over 2 KV heads. PASS iff
// every output is finite and, in every cell, steel's max-abs error is at most 2x the fused kernel's and its mean-abs error
// at most 1.25x.
func TestKSD2_steelHeadDims(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	pFused := pipe("attention_prefill_fused")
	q := d.NewCommandQueue()
	const nH, nKV = 14, 2
	rng := rand.New(rand.NewSource(29))
	for _, hd := range []int{64, 80, 96} {
		pSteel := pipe(fmt.Sprintf("attention_prefill_steel_hd%d", hd))
		qDim, kvDim := nH*hd, nKV*hd
		stride := qDim + 2*kvDim
		scale := float32(1 / math.Sqrt(float64(hd)))
		for _, M := range []int{100, 300} {
			for _, start := range []int{0, 37} {
				for _, window := range []int{0, 64} {
					Mpad, ctx := (M+7)/8*8, start+M
					qkv := make([]uint16, Mpad*stride)
					for i := range qkv {
						qkv[i] = f32ToF16(float32(rng.NormFloat64()))
					}
					kc, vc := make([]uint16, ctx*kvDim), make([]uint16, ctx*kvDim)
					for i := range kc {
						kc[i], vc[i] = f32ToF16(float32(rng.NormFloat64())), f32ToF16(float32(rng.NormFloat64()))
					}
					ref := steelRefAttention(qkv, kc, vc, M, nH, nKV, hd, stride, start, window, float64(scale))
					bQ, bK, bV := NewBufferU16s(d, qkv), NewBufferU16s(d, kc), NewBufferU16s(d, vc)
					run := func(steel bool) []uint16 {
						out := NewBufferU16s(d, make([]uint16, Mpad*qDim))
						args := []Buffer{bQ, bK, bV, out, NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, uint32(hd)),
							NewBufferU32(d, uint32(start)), NewBufferFloats(d, []float32{scale}), NewBufferU32(d, uint32(stride)),
							NewBufferU32(d, uint32(window)), NewBufferU32(d, uint32(M))}
						e := q.Begin()
						if steel {
							e.Dispatch(pSteel, nH*((M+31)/32)*128, 128, args...)
						} else {
							total := (nH*((M+7)/8) + 3) / 4 * 4 * 32
							e.Dispatch(pFused, total, 4*32, args...)
						}
						e.End()
						return out.U16s()[:M*qDim]
					}
					errs := func(got []uint16) (maxE, meanE float64, finite bool) {
						finite = true
						for i, h := range got {
							v := float64(f16ToF32(h))
							if math.IsNaN(v) || math.IsInf(v, 0) {
								finite = false
							}
							e := math.Abs(v - ref[i])
							maxE = max(maxE, e)
							meanE += e
						}
						return maxE, meanE / float64(len(got)), finite
					}
					sMax, sMean, sFin := errs(run(true))
					fMax, fMean, fFin := errs(run(false))
					fmt.Fprintf(os.Stderr, "[D2-G2] hd=%d M=%d start=%d window=%d: steel max %.3g mean %.3g, fused max %.3g mean %.3g (steel/fused %.2f, %.2f)\n",
						hd, M, start, window, sMax, sMean, fMax, fMean, sMax/fMax, sMean/fMean)
					if !sFin || !fFin || sMax > 2*fMax || sMean > 1.25*fMean {
						t.Errorf("hd=%d M=%d start=%d window=%d: steel max %.3g mean %.3g (finite %v) against fused max %.3g mean %.3g (finite %v)",
							hd, M, start, window, sMax, sMean, sFin, fMax, fMean, fFin)
					}
				}
			}
		}
	}
}

// steelRefAttention is causal (optionally windowed) attention in float64 over half inputs: row m of M sits at position
// start+m and attends to keys [lo, start+m], lo = start+m+1-window when a window is set; output [M x nH*hd].
func steelRefAttention(qkv, kc, vc []uint16, M, nH, nKV, hd, stride, start, window int, scale float64) []float64 {
	kvDim, out := nKV*hd, make([]float64, M*nH*hd)
	for m := range M {
		p := start + m
		lo := 0
		if window > 0 && p+1 > window {
			lo = p + 1 - window
		}
		for h := range nH {
			kvh := h / (nH / nKV)
			sc := make([]float64, p+1-lo)
			mx := math.Inf(-1)
			for j := lo; j <= p; j++ {
				var dot float64
				for c := range hd {
					dot += float64(f16ToF32(qkv[m*stride+h*hd+c])) * float64(f16ToF32(kc[j*kvDim+kvh*hd+c]))
				}
				sc[j-lo] = dot * scale
				mx = max(mx, sc[j-lo])
			}
			var sum float64
			for i := range sc {
				sc[i] = math.Exp(sc[i] - mx)
				sum += sc[i]
			}
			for c := range hd {
				var acc float64
				for j := lo; j <= p; j++ {
					acc += sc[j-lo] * float64(f16ToF32(vc[j*kvDim+kvh*hd+c]))
				}
				out[m*nH*hd+h*hd+c] = acc / sum
			}
		}
	}
	return out
}
