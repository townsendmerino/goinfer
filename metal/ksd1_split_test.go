//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// TestKSD1_splitKernel is D1-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D1 gates"): attention_prefill_steel_split at head
// dims 256 and 512 against an f64 attention over the same half inputs, at random q, k, v; M = 100 and 300; start 0 and 37;
// window 0 and 64 (at 256); 8 query heads over 1 KV head. PASS iff every output is finite and max-abs <= 4e-3, mean-abs <=
// 2e-4 in every cell. It also checks (not a registered gate) that a 300-row pass equals the same rows run as two passes of 150,
// byte for byte, which the blocked prefill relies on.
func TestKSD1_splitKernel(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	const nH, nKV = 8, 1
	rng := rand.New(rand.NewSource(43))
	q := d.NewCommandQueue()
	for _, hd := range []int{256, 512} {
		p, err := d.NewComputePipeline(lib, fmt.Sprintf("attention_prefill_steel_hd%d", hd))
		if err != nil {
			t.Fatal(err)
		}
		bq := 32
		if hd == 512 {
			bq = 16
		}
		qDim, kvDim := nH*hd, nKV*hd
		stride := qDim + 2*kvDim
		scale := float32(1 / math.Sqrt(float64(hd)))
		windows := []int{0}
		if hd == 256 {
			windows = append(windows, 64)
		}
		for _, M := range []int{100, 300} {
			for _, start := range []int{0, 37} {
				for _, window := range windows {
					Mpad, ctx := (M+7)/8*8, start+M
					qkv := make([]uint16, Mpad*stride)
					for i := range qkv {
						qkv[i] = f32ToF16(float32(rng.NormFloat64()))
					}
					kc, vc := make([]uint16, ctx*kvDim), make([]uint16, ctx*kvDim)
					for i := range kc {
						kc[i], vc[i] = f32ToF16(float32(rng.NormFloat64())), f32ToF16(float32(rng.NormFloat64()))
					}
					ref := ksd1RefAttention(qkv, kc, vc, M, nH, nKV, hd, stride, start, window, float64(scale))
					bQ, bK, bV := NewBufferU16s(d, qkv), NewBufferU16s(d, kc), NewBufferU16s(d, vc)
					run := func(rowOff, rows, st int) []uint16 {
						out := NewBufferU16s(d, make([]uint16, (rows+7)/8*8*qDim))
						e := q.Begin()
						e.Dispatch(p, nH*((rows+bq-1)/bq)*256, 256, bQ.At(rowOff*stride*2), bK, bV, out, NewBufferU32(d, nH),
							NewBufferU32(d, nKV), NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(st)), NewBufferFloats(d, []float32{scale}),
							NewBufferU32(d, uint32(stride)), NewBufferU32(d, uint32(window)), NewBufferU32(d, uint32(rows)))
						e.End()
						return slices.Clone(out.U16s()[:rows*qDim])
					}
					got := run(0, M, start)
					var maxE, meanE float64
					finite := true
					for i, h := range got {
						v := float64(f16ToF32(h))
						if math.IsNaN(v) || math.IsInf(v, 0) {
							finite = false
						}
						e := math.Abs(v - ref[i])
						maxE = max(maxE, e)
						meanE += e
					}
					meanE /= float64(len(got))
					split := append(run(0, M/2, start), run(M/2, M-M/2, start+M/2)...)
					chunkDiff := 0
					for i := range got {
						if got[i] != split[i] {
							chunkDiff++
						}
					}
					fmt.Fprintf(os.Stderr, "[D1-G1] hd=%d M=%d start=%d window=%d: max-abs %.3g, mean-abs %.3g, finite %v; two passes: %d of %d differ\n",
						hd, M, start, window, maxE, meanE, finite, chunkDiff, len(got))
					if !finite || maxE > 4e-3 || meanE > 2e-4 {
						t.Errorf("hd=%d M=%d start=%d window=%d: max-abs %.3g mean-abs %.3g finite %v", hd, M, start, window, maxE, meanE, finite)
					}
					if chunkDiff != 0 {
						t.Errorf("hd=%d M=%d start=%d window=%d: %d outputs differ when the rows run as two passes", hd, M, start, window, chunkDiff)
					}
				}
			}
		}
	}
}

// ksd1RefAttention is causal (optionally windowed) attention in float64 over half inputs: row m sits at position start+m and
// attends to keys [lo, start+m], lo = start+m+1-window when a window is set; output [M x nH*hd].
func ksd1RefAttention(qkv, kc, vc []uint16, M, nH, nKV, hd, stride, start, window int, scale float64) []float64 {
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
