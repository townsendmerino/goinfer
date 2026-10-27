//go:build darwin

package kernels

import (
	"fmt"
	"math"
	"testing"
)

// gptossRouteRef is the gpt-oss router's selection on the host, with the rule the three Metal kernels share: top-k by
// repeated strict-> maximum over the biased logits (ties to the lowest index), and when no remaining score is above
// -Inf, the lowest expert not yet chosen. It returns the indices only: the weights' bits are the device's exp.
func gptossRouteRef(logits, bias []float32, k int) []uint32 {
	nE := len(logits)
	taken := make([]bool, nE)
	out := make([]uint32, k)
	for j := range k {
		best, bv := -1, float32(math.Inf(-1))
		for i := range nE {
			if s := logits[i] + bias[i]; !taken[i] && s > bv {
				bv, best = s, i
			}
		}
		if best < 0 {
			best = 0
			for i := range nE {
				if !taken[i] {
					best = i
					break
				}
			}
		}
		taken[best] = true
		out[j] = uint32(best)
	}
	return out
}

// TestGptOssRoute_neverEmitsAnOutOfRangeExpert: when no remaining score is above -Inf (every score NaN or -Inf, from a
// crafted router or an activation that overflowed), the three gpt-oss router kernels (route_gptoss on one thread,
// route_gptoss_sg on one simdgroup, route_gptoss_batch one row a thread) left their best index at the sentinel and
// emitted expert 0xFFFFFFFF, which the indexed expert GEMVs use as a row offset about 2^32 rows past the weights.
// Whatever the scores, every index must be a valid expert, no expert may repeat, every weight must be finite, the
// indices must be the host rule's, and the three kernels must write the same indices and the same weight bits. The
// last rows of each size are ordinary scores: the controls that the fallback changes nothing when a score is there.
func TestGptOssRoute_neverEmitsAnOutOfRangeExpert(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	plib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefill kernels: %v", err)
	}
	pSer, err := d.NewComputePipeline(lib, "route_gptoss")
	if err != nil {
		t.Fatalf("pipeline route_gptoss: %v", err)
	}
	pSG, err := d.NewComputePipeline(lib, "route_gptoss_sg")
	if err != nil {
		t.Fatalf("pipeline route_gptoss_sg: %v", err)
	}
	pBatch, err := d.NewComputePipeline(plib, "route_gptoss_batch")
	if err != nil {
		t.Fatalf("pipeline route_gptoss_batch: %v", err)
	}
	q := d.NewCommandQueue()
	nan, ninf := float32(math.NaN()), float32(math.Inf(-1))

	type row struct {
		name   string
		logits func(i int) float32
		bias   func(i int) float32
	}
	zero := func(int) float32 { return 0 }
	rows := []row{
		{"every score NaN", func(int) float32 { return nan }, zero},
		{"every score -Inf", func(int) float32 { return ninf }, zero},
		{"every score below -1e30", func(int) float32 { return -3e30 }, zero},
		{"fewer finite scores than k", func(i int) float32 {
			if i < 2 {
				return float32(i)
			}
			return nan
		}, zero},
		{"the only finite scores sit at high indices", func(i int) float32 {
			if i == 5 || i == 7 {
				return float32(i)
			}
			return ninf
		}, zero},
		{"NaN and -Inf mixed, one finite score", func(i int) float32 {
			switch {
			case i == 3:
				return -2
			case i%2 == 0:
				return nan
			}
			return ninf
		}, zero},
		{"finite logits, a NaN bias on every expert", func(i int) float32 { return float32(i%7) * 0.5 }, func(int) float32 { return nan }},
		{"control: ordinary scores", func(i int) float32 { return float32(math.Sin(float64(i)*0.9)) * 2 }, func(i int) float32 { return float32(i%5-2) * 0.05 }},
		{"control: every score equal", func(int) float32 { return 0.5 }, zero},
	}
	for _, sz := range []struct{ nE, k int }{{32, 4}, {40, 6}, {128, 8}, {8, 8}} {
		for _, r := range rows {
			name := fmt.Sprintf("nE%d_k%d/%s", sz.nE, sz.k, r.name)
			logits, bias := make([]float32, sz.nE), make([]float32, sz.nE)
			for i := range logits {
				logits[i], bias[i] = r.logits(i), r.bias(i)
			}
			want := gptossRouteRef(logits, bias, sz.k)
			run := func(p Pipeline, threads, tg int, batch bool) ([]uint32, []uint32) {
				const M = 3 // the batched kernel: the row under test between two ordinary rows
				n := 1
				in := logits
				if batch {
					n = M
					in = make([]float32, 0, M*sz.nE)
					for m := range M {
						for i := range sz.nE {
							if m == 1 {
								in = append(in, logits[i])
							} else {
								in = append(in, float32(math.Cos(float64(i+m)*0.7)))
							}
						}
					}
				}
				dIdx := NewBufferUint32s(d, make([]uint32, n*sz.k))
				dW := d.NewBufferLen(n * sz.k)
				bufs := []Buffer{NewBufferFloats(d, in), NewBufferFloats(d, bias), dIdx, dW, NewBufferU32(d, uint32(sz.nE)), NewBufferU32(d, uint32(sz.k))}
				if batch {
					bufs = append(bufs, NewBufferU32(d, uint32(M)))
				}
				q.Run1D(p, threads, tg, bufs...)
				idx, w := dIdx.U32s(), dW.Floats()
				if batch {
					idx, w = idx[sz.k:2*sz.k], w[sz.k:2*sz.k]
				}
				wb := make([]uint32, len(w))
				for i, x := range w {
					wb[i] = math.Float32bits(x)
				}
				return append([]uint32(nil), idx...), wb
			}
			type out struct {
				name   string
				idx, w []uint32
			}
			serIdx, serW := run(pSer, 1, 1, false)
			sgIdx, sgW := run(pSG, 32, 32, false)
			bIdx, bW := run(pBatch, 3*32, 32, true)
			for _, kern := range []out{{"route_gptoss", serIdx, serW}, {"route_gptoss_sg", sgIdx, sgW}, {"route_gptoss_batch", bIdx, bW}} {
				seen := map[uint32]bool{}
				for j := range sz.k {
					if kern.idx[j] >= uint32(sz.nE) {
						t.Errorf("%s: %s: slot %d routes to expert %#x (nE %d)", name, kern.name, j, kern.idx[j], sz.nE)
					}
					if seen[kern.idx[j]] {
						t.Errorf("%s: %s: expert %d chosen twice: %v", name, kern.name, kern.idx[j], kern.idx)
					}
					seen[kern.idx[j]] = true
					if w := float64(math.Float32frombits(kern.w[j])); math.IsNaN(w) || math.IsInf(w, 0) {
						t.Errorf("%s: %s: slot %d weight %v is not finite", name, kern.name, j, w)
					}
				}
				if fmt.Sprint(kern.idx) != fmt.Sprint(want) {
					t.Errorf("%s: %s: experts %v, the rule gives %v", name, kern.name, kern.idx, want)
				}
				if fmt.Sprint(kern.idx) != fmt.Sprint(serIdx) || fmt.Sprint(kern.w) != fmt.Sprint(serW) {
					t.Errorf("%s: %s differs from route_gptoss: experts %v against %v, weight bits %08x against %08x", name, kern.name, kern.idx, serIdx, kern.w, serW)
				}
			}
		}
	}
}
