//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The attention-score softcap at kernel level (OP1's gate G2-K, docs/tasks/task-forward-architecture-2026-10.md §4.6):
// each kernel a capped layer can take, compiled with the cap, against a float64 reference that applies
// cap·tanh(score/cap) to every scaled score. The inputs carry planted keys aligned with a query, in two sets: one whose
// scores compete inside ±50 (Gemma 2's cap) and cross ±5, and one whose scores cross ±50. Competing keys matter: with a
// single dominant key the softmax is one-hot with or without a cap, and a dropped cap would not move the output.
//
// Each case also runs the same inputs through the kernel compiled without the cap and requires that to MISS the bar, so
// a build that dropped the cap cannot pass here (G2-D).

const (
	softcapCosBar    = 0.9999
	softcapMaxAbsBar = 1e-2
)

// softcapPlants are the multiples of a query planted as keys: two close positive scores, a weaker one and a negative.
var softcapPlants = map[string][]float32{
	"scores inside ±50": {4, 3.8, 2, -4},
	"scores past ±50":   {12, 11.5, -12},
}

func softcapRef(x, cap float64) float64 {
	if cap == 0 {
		return x
	}
	return cap * math.Tanh(x/cap)
}

// attnRef64Cap is attnRef64 with the softcap and no sink.
func attnRef64Cap(q, k, v []float32, nH, nKV, hd, from, to int, scale, cap float64) []float32 {
	out := make([]float32, nH*hd)
	kvDim := nKV * hd
	for h := range nH {
		kvh := h / (nH / nKV)
		scores := make([]float64, to-from)
		mx := math.Inf(-1)
		for s := from; s < to; s++ {
			var a float64
			for i := range hd {
				a += float64(q[h*hd+i]) * float64(k[s*kvDim+kvh*hd+i])
			}
			scores[s-from] = softcapRef(a*scale, cap)
			mx = math.Max(mx, scores[s-from])
		}
		var den float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - mx)
			den += scores[i]
		}
		for i := range hd {
			var a float64
			for s := from; s < to; s++ {
				a += scores[s-from] * float64(v[s*kvDim+kvh*hd+i])
			}
			out[h*hd+i] = float32(a / den)
		}
	}
	return out
}

func softcapCosMaxAbs(got, want []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		dot, na, nb = dot+g*w, na+g*g, nb+w*w
		maxAbs = math.Max(maxAbs, math.Abs(g-w))
	}
	return dot / math.Sqrt(na*nb), maxAbs
}

// TestAttnSoftcap_decodeKernels covers attention and attention_i8, on both sides of their 4096-key tile bound and with
// a window.
func TestAttnSoftcap_decodeKernels(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	plain, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cq := d.NewCommandQueue()
	const nH, nKV = 4, 2
	checked := 0
	for _, cap := range []float32{50, 5} {
		src, err := withAttnSoftcap(allKernels, cap)
		if err != nil {
			t.Fatal(err)
		}
		capped, err := d.CompileLibrary(src, MSL3_1)
		if err != nil {
			t.Fatalf("compile with cap %g: %v", cap, err)
		}
		for _, kernel := range []string{"attention", "attention_i8"} {
			pCap, err := d.NewComputePipeline(capped, kernel)
			if err != nil {
				t.Fatalf("pipeline %s: %v", kernel, err)
			}
			pPlain, err := d.NewComputePipeline(plain, kernel)
			if err != nil {
				t.Fatalf("pipeline %s: %v", kernel, err)
			}
			for _, hd := range []int{128, 256} {
				for _, c := range []struct{ nKeys, window int }{{24, 0}, {600, 0}, {600, 512}, {4096, 0}, {4097, 0}, {8193, 0}, {9000, 6000}} {
					for plantName, plants := range softcapPlants {
						name := fmt.Sprintf("cap=%g %s hd=%d nKeys=%d window=%d, %s", cap, kernel, hd, c.nKeys, c.window, plantName)
						rng := rand.New(rand.NewSource(int64(hd*100000 + c.nKeys*10 + len(plants))))
						kvDim := nKV * hd
						qv := randUnit(rng, nH*hd)
						kf, vf := randUnit(rng, c.nKeys*kvDim), randUnit(rng, c.nKeys*kvDim)
						winStart := 0
						if c.window > 0 && c.nKeys > c.window {
							winStart = c.nKeys - c.window
						}
						// Spread the planted keys over the keys the query sees, the last one included (so past the tile
						// bound the plants sit in different tiles).
						for i, a := range plants {
							s := winStart + (c.nKeys-winStart-1)*(i+1)/len(plants)
							for kvh := range nKV {
								h0 := kvh * (nH / nKV)
								for j := range hd {
									kf[s*kvDim+kvh*hd+j] = a * qv[h0*hd+j]
								}
							}
						}
						var kBuf, vBuf, ksBuf, vsBuf Buffer
						if kernel == "attention" {
							kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
							for i := range kf {
								kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
								kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
							}
							kBuf, vBuf = NewBufferU16s(d, kh), NewBufferU16s(d, vh)
						} else {
							kq, ks := quantKVRowsI8(kf, c.nKeys, nKV, hd)
							vq, vs := quantKVRowsI8(vf, c.nKeys, nKV, hd)
							for i := range kf {
								p, h := i/kvDim, (i%kvDim)/hd
								kf[i], vf[i] = float32(kq[i])*ks[p*nKV+h], float32(vq[i])*vs[p*nKV+h]
							}
							kBuf, vBuf, ksBuf, vsBuf = NewBufferInt8(d, kq), NewBufferInt8(d, vq), NewBufferFloats(d, ks), NewBufferFloats(d, vs)
						}
						scale := float32(1 / math.Sqrt(float64(hd)))
						u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
						run := func(pipe Pipeline) []float32 {
							out := d.NewBufferLen(nH * hd)
							bufs := []Buffer{NewBufferFloats(d, qv), kBuf, vBuf}
							if kernel == "attention_i8" {
								bufs = append(bufs, ksBuf, vsBuf)
							}
							bufs = append(bufs, out, u(nH), u(nKV), u(hd), u(c.nKeys), NewBufferFloats(d, []float32{scale}), u(c.window),
								NewBufferFloats(d, make([]float32, nH)), u(0))
							e := cq.Begin()
							e.Dispatch(pipe, nH*tgReduceAttn, tgReduceAttn, bufs...)
							e.End()
							if err := e.Err(); err != nil {
								t.Fatalf("%s: dispatch: %v", name, err)
							}
							return append([]float32(nil), out.Floats()[:nH*hd]...)
						}
						got, dropped := run(pCap), run(pPlain)
						d.ReleaseAll()
						want := attnRef64Cap(qv, kf, vf, nH, nKV, hd, winStart, c.nKeys, float64(scale), float64(cap))
						if cos, ma := softcapCosMaxAbs(got, want); !(cos >= softcapCosBar && ma <= softcapMaxAbsBar) {
							t.Errorf("%s: cosine %.7f, max abs error %.3g against the capped float64 reference (want >= %g and <= %g)", name, cos, ma, softcapCosBar, softcapMaxAbsBar)
						}
						if cos, ma := softcapCosMaxAbs(dropped, want); cos >= softcapCosBar && ma <= softcapMaxAbsBar {
							t.Errorf("%s: the kernel built WITHOUT the cap also meets the bar (cosine %.7f, max abs %.3g): this case cannot see a dropped cap", name, cos, ma)
						}
						checked++
					}
				}
			}
		}
	}
	t.Logf("%d cap × kernel × head-dim × key-count × plant cases within the bar, and outside it with the cap dropped", checked)
}

// TestAttnSoftcap_prefillKernel covers attention_prefill, the exact prefill kernel: M query rows over a cached prefix,
// with and without a window. The planted keys are aligned with the last row's query.
func TestAttnSoftcap_prefillKernel(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	plain, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cq := d.NewCommandQueue()
	const nH, nKV = 4, 2
	checked := 0
	for _, cap := range []float32{50, 5} {
		src, err := withAttnSoftcap(prefillKernels, cap)
		if err != nil {
			t.Fatal(err)
		}
		capped, err := d.CompileLibrary(src, MSL3_1)
		if err != nil {
			t.Fatalf("compile with cap %g: %v", cap, err)
		}
		pCap, err := d.NewComputePipeline(capped, "attention_prefill")
		if err != nil {
			t.Fatal(err)
		}
		pPlain, err := d.NewComputePipeline(plain, "attention_prefill")
		if err != nil {
			t.Fatal(err)
		}
		for _, hd := range []int{128, 256} {
			for _, c := range []struct{ startPos, M, window int }{{0, 24, 0}, {5, 40, 0}, {0, 64, 16}, {700, 32, 0}, {700, 32, 512}} {
				for plantName, plants := range softcapPlants {
					name := fmt.Sprintf("cap=%g attention_prefill hd=%d startPos=%d M=%d window=%d, %s", cap, hd, c.startPos, c.M, c.window, plantName)
					rng := rand.New(rand.NewSource(int64(hd*100000 + c.startPos*100 + c.M + len(plants))))
					total := c.startPos + c.M
					qStride, kvDim := (nH+2*nKV)*hd, nKV*hd
					qf := randUnit(rng, c.M*qStride)
					kf, vf := randUnit(rng, total*kvDim), randUnit(rng, total*kvDim)
					last := c.M - 1
					lo := 0
					if c.window > 0 && total > c.window {
						lo = total - c.window
					}
					for i, a := range plants { // inside the last row's window, the last key included
						s := lo + (total-lo-1)*(i+1)/len(plants)
						for kvh := range nKV {
							h0 := kvh * (nH / nKV)
							for j := range hd {
								kf[s*kvDim+kvh*hd+j] = a * qf[last*qStride+h0*hd+j]
							}
						}
					}
					half := func(x []float32) []uint16 {
						h := make([]uint16, len(x))
						for i := range x {
							h[i] = f32ToF16(x[i])
							x[i] = f16ToF32(h[i])
						}
						return h
					}
					qh, kh, vh := half(qf), half(kf), half(vf)
					scale := float32(1 / math.Sqrt(float64(hd)))
					run := func(pipe Pipeline) []float32 {
						out := NewBufferU16s(d, make([]uint16, c.M*nH*hd))
						e := cq.Begin()
						e.Dispatch(pipe, c.M*nH*tgReduceAttn, tgReduceAttn, NewBufferU16s(d, qh), NewBufferU16s(d, kh), NewBufferU16s(d, vh), out,
							NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(c.startPos)),
							NewBufferFloats(d, []float32{scale}), NewBufferU32(d, uint32(qStride)), NewBufferU32(d, uint32(c.window)))
						e.End()
						if err := e.Err(); err != nil {
							t.Fatalf("%s: dispatch: %v", name, err)
						}
						o := make([]float32, c.M*nH*hd)
						for i, h := range out.U16s()[:len(o)] {
							o[i] = f16ToF32(h)
						}
						return o
					}
					got, dropped := run(pCap), run(pPlain)
					d.ReleaseAll()
					want := make([]float32, 0, c.M*nH*hd)
					for m := range c.M {
						pos := c.startPos + m
						from := 0
						if c.window > 0 && pos+1 > c.window {
							from = pos + 1 - c.window
						}
						qrow := make([]float32, nH*hd)
						copy(qrow, qf[m*qStride:m*qStride+nH*hd])
						want = append(want, attnRef64Cap(qrow, kf, vf, nH, nKV, hd, from, pos+1, float64(scale), float64(cap))...)
					}
					// The output is f16, and the last row is the one whose query the planted keys answer.
					lastRow := func(x []float32) []float32 { return x[last*nH*hd:] }
					if cos, ma := softcapCosMaxAbs(got, want); !(cos >= softcapCosBar && ma <= softcapMaxAbsBar) {
						t.Errorf("%s: cosine %.7f, max abs error %.3g against the capped float64 reference (want >= %g and <= %g)", name, cos, ma, softcapCosBar, softcapMaxAbsBar)
					}
					if cos, ma := softcapCosMaxAbs(lastRow(dropped), lastRow(want)); cos >= softcapCosBar && ma <= softcapMaxAbsBar {
						t.Errorf("%s: the kernel built WITHOUT the cap also meets the bar on the planted row (cosine %.7f, max abs %.3g): this case cannot see a dropped cap", name, cos, ma)
					}
					checked++
				}
			}
		}
	}
	t.Logf("%d cap × head-dim × shape × plant cases within the bar, and outside it with the cap dropped", checked)
}

// TestWithAttnSoftcap_noCapIsTheSameSource: a model without a cap compiles the source it always did, and a cap replaces
// the score function and nothing else.
func TestWithAttnSoftcap_noCapIsTheSameSource(t *testing.T) {
	for _, src := range []string{allKernels + mc3RowsKernels, prefillKernels} {
		same, err := withAttnSoftcap(src, 0)
		if err != nil || same != src {
			t.Fatalf("cap 0: err %v, source changed %v", err, same != src)
		}
		capped, err := withAttnSoftcap(src, 50)
		if err != nil {
			t.Fatal(err)
		}
		if capped == src {
			t.Fatal("cap 50 left the source unchanged")
		}
		// Outside the replaced function the two sources are the same text.
		i := 0
		for i < len(src) && i < len(capped) && src[i] == capped[i] {
			i++
		}
		j := 0
		for j < len(src)-i && j < len(capped)-i && src[len(src)-1-j] == capped[len(capped)-1-j] {
			j++
		}
		if changed := src[i : len(src)-j]; len(changed) > 200 {
			t.Errorf("a cap changed %d bytes of the source, more than the score function: %q...", len(changed), changed[:80])
		}
	}
	if _, err := withAttnSoftcap(allKernels, -1); err == nil {
		t.Error("a negative cap was accepted")
	}
}
