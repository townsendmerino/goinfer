//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// TestAttentionPrefillSteelMatchesFloat64 (F-G01, docs/audit-metal-2026-09-30.md) runs attention_prefill_steel, the
// prefill attention for every head-dim-128 model, against float64 attention over the same f16 Q, K and V. No test that
// runs by default asserted anything about this kernel before; its evidence was one pooled fidelity run at startPos 0.
// The cases cover what that run did not: startPos > 0 (every chunk after the first under chunked prefill, and every
// prefix-reuse turn), M on both sides of the 32-row tile, a sliding window, three GQA groups, and peaked scores that
// move the running max between key blocks. The launch, the fused-QKV row stride and the 8-row padding of Q and the
// output are PrefillLast's; the K/V cache holds only the keys the call may read, rounded to 8 rows, not a context's
// worth.
//
// The test can fail: in every case whose earliest query row sees at most 64 keys, the answer with one more key per row
// (a causal limit off by one) must miss the max-abs bar.
func TestAttentionPrefillSteelMatchesFloat64(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefillKernels: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention_prefill_steel")
	if err != nil {
		t.Fatalf("pipeline attention_prefill_steel: %v", err)
	}
	cq := d.NewCommandQueue()
	// maxAbs is the audit's 0.05 (the fused kernel's bar) tightened to what this kernel measures: 4.1e-4 at worst, about
	// two f16 rounding steps of a unit-scale output (2026-10-01).
	const hd, minCos, maxAbs = 128, 0.9999, 1e-3
	scale := float32(1 / math.Sqrt(hd))
	worstCos, worstAbs, minMiss := 1.0, 0.0, math.Inf(1)
	var cases, discriminating int
	for _, g := range []struct{ nH, nKV int }{{12, 2}, {14, 2}, {32, 8}} {
		for _, M := range []int{1, 7, 31, 33, 100} {
			for _, startPos := range []int{0, 5, 16, 1000} {
				for _, window := range []int{0, 64} {
					for _, peaked := range []bool{false, true} {
						name := fmt.Sprintf("nH=%d nKV=%d M=%d startPos=%d window=%d peaked=%v", g.nH, g.nKV, M, startPos, window, peaked)
						qDim, kvDim := g.nH*hd, g.nKV*hd
						stride := qDim + 2*kvDim
						mPad, rows := (M+7)/8*8, (startPos+M+7)/8*8
						rng := rand.New(rand.NewSource(int64(cases)*7919 + 20261001))
						amp := float32(1)
						if peaked {
							amp = 8 // scores several units apart, so the running max moves between key blocks
						}
						qkv := randHalves(rng, mPad*stride, amp)
						kc, vc := randHalves(rng, rows*kvDim, 1), randHalves(rng, rows*kvDim, 1)
						out := make([]uint16, mPad*qDim)
						for i := range out {
							out[i] = 0x7e00 // f16 NaN: an element the kernel does not write fails the NaN check
						}
						outB := NewBufferU16s(d, out)
						u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
						e := cq.Begin()
						e.Dispatch(pipe, g.nH*((M+31)/32)*128, 128, NewBufferU16s(d, qkv), NewBufferU16s(d, kc), NewBufferU16s(d, vc),
							outB, u(g.nH), u(g.nKV), u(hd), u(startPos), NewBufferFloats(d, []float32{scale}), u(stride), u(window), u(M))
						e.End()
						if err := e.Err(); err != nil {
							t.Fatalf("%s: dispatch: %v", name, err)
						}
						got := append([]uint16(nil), outB.U16s()[:M*qDim]...)
						d.ReleaseAll()

						ref := steelRef64(qkv, kc, vc, g.nH, g.nKV, M, startPos, window, stride, float64(scale), 0)
						cos, abs, nans := steelCompare(got, ref, g.nH, M)
						worstCos, worstAbs = math.Min(worstCos, cos), math.Max(worstAbs, abs)
						if nans > 0 || !(cos >= minCos) || !(abs <= maxAbs) {
							t.Errorf("%s: %d NaN, worst row-head cosine %.7f (want >= %g), max |kernel - float64| %.3g (want <= %g)",
								name, nans, cos, minCos, abs, maxAbs)
						}
						firstKeys := startPos + 1 // the keys query row 0 sees
						if window > 0 {
							firstKeys = min(firstKeys, window)
						}
						if firstKeys <= 64 {
							off := steelRef64(qkv, kc, vc, g.nH, g.nKV, M, startPos, window, stride, float64(scale), 1)
							_, miss, _ := steelCompareF64(off, ref, g.nH, M)
							minMiss = math.Min(minMiss, miss)
							if miss <= maxAbs {
								t.Errorf("%s: a causal limit off by one moves the answer by only %.3g; the case cannot catch it", name, miss)
							}
							discriminating++
						}
						cases++
					}
				}
			}
		}
	}
	t.Logf("%d cases: worst row-head cosine %.7f, max |kernel - float64| %.3g; an off-by-one causal limit moves the "+
		"answer by at least %.3g in the %d cases that check it", cases, worstCos, worstAbs, minMiss, discriminating)
}

func randHalves(rng *rand.Rand, n int, amp float32) []uint16 {
	v := make([]uint16, n)
	for i := range v {
		v[i] = f32ToF16((rng.Float32()*2 - 1) * amp)
	}
	return v
}

// steelRef64 is causal prefill attention in float64: query row i sits at position startPos+i and sees keys
// [max(0, p+1-window), p+1) with p = startPos+i+extra (window 0 = no lower bound), clipped to the cache. extra is 0
// for the reference and 1 for the off-by-one answer the test must be able to tell apart from it.
func steelRef64(qkv, kc, vc []uint16, nH, nKV, M, startPos, window, stride int, scale float64, extra int) []float64 {
	const hd = 128
	kvDim, rows := nKV*hd, len(kc)/(nKV*hd)
	kf, vf := halvesToF64(kc), halvesToF64(vc)
	out := make([]float64, M*nH*hd)
	var wg sync.WaitGroup
	for h := range nH {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()
			kvh := h / (nH / nKV)
			q := make([]float64, hd)
			sc := make([]float64, 0, startPos+M+1)
			for i := range M {
				for dd := range hd {
					q[dd] = float64(f16ToF32(qkv[i*stride+h*hd+dd]))
				}
				end := min(startPos+i+1+extra, rows)
				from := 0
				if window > 0 && end > window {
					from = end - window
				}
				sc = sc[:0]
				mx := math.Inf(-1)
				for s := from; s < end; s++ {
					var a float64
					for dd, k := range kf[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd] {
						a += q[dd] * k
					}
					sc = append(sc, a*scale)
					mx = math.Max(mx, a*scale)
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				o := out[(i*nH+h)*hd : (i*nH+h+1)*hd]
				for j, w := range sc {
					for dd, v := range vf[(from+j)*kvDim+kvh*hd : (from+j)*kvDim+kvh*hd+hd] {
						o[dd] += w * v
					}
				}
				for dd := range o {
					o[dd] /= sum
				}
			}
		}(h)
	}
	wg.Wait()
	return out
}

func halvesToF64(h []uint16) []float64 {
	f := make([]float64, len(h))
	for i, x := range h {
		f[i] = float64(f16ToF32(x))
	}
	return f
}

// steelCompare gives the worst cosine over (row, head) vectors, the max absolute difference and the NaN count of the
// kernel's f16 output against a float64 reference laid out [M][nH][128].
func steelCompare(got []uint16, ref []float64, nH, M int) (worstCos, maxAbs float64, nans int) {
	return steelCompareF64(halvesToF64(got), ref, nH, M)
}

func steelCompareF64(got, ref []float64, nH, M int) (worstCos, maxAbs float64, nans int) {
	const hd = 128
	worstCos = 1
	for v := range M * nH {
		var dot, na, nb float64
		for dd := range hd {
			a, b := got[v*hd+dd], ref[v*hd+dd]
			if math.IsNaN(a) {
				nans++
				continue
			}
			dot, na, nb = dot+a*b, na+a*a, nb+b*b
			maxAbs = math.Max(maxAbs, math.Abs(a-b))
		}
		worstCos = math.Min(worstCos, dot/(math.Sqrt(na)*math.Sqrt(nb)+1e-300))
	}
	return worstCos, maxAbs, nans
}
