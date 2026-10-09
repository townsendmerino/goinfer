//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// G-PR1 (docs/tasks/task-metal-pairwise-rope-2026-10.md): the pairwise rope twins against a float64 reference written
// from decoder/rope.go's semantics (applyRoPEInterleaved, applyMRoPEPairwise): pair dd is dims (2dd, 2dd+1) at invf[dd],
// partial rotary rotates [0, 2*rhalf) and leaves the tail alone, mscale scales cos and sin, the Q temperature multiplies
// rotated Q only. Tolerances as registered: 2e-5 on the f32 kernels, 4e-3 on the f16 ones (inputs in [-2, 2]). Every
// case also runs the NeoX kernel on the same input, which must miss the tolerance by at least 10x, so no case is
// vacuous. Positions stay small (<= 64) so the kernels' float32 angle stays inside the f32 tolerance.

const (
	pwTolF32 = 2e-5
	pwTolF16 = 4e-3
)

// pwRef rotates heads of x (head dim hd) pairwise in float64: pair dd turns by posOf(dd)*invf[dd], times scale, and the
// result is multiplied by mult.
func pwRef(x []float64, heads, hd, rhalf int, posOf func(dd int) float64, invf []float32, scale, mult float64) {
	for h := range heads {
		for dd := range rhalf {
			th := posOf(dd) * float64(invf[dd])
			c, s := math.Cos(th)*scale, math.Sin(th)*scale
			i := h*hd + 2*dd
			x0, x1 := x[i], x[i+1]
			x[i], x[i+1] = (x0*c-x1*s)*mult, (x0*s+x1*c)*mult
		}
	}
}

func pwInvf(rhalf, hd int) []float32 {
	f := make([]float32, rhalf)
	for i := range f {
		f[i] = float32(1 / math.Pow(10000, float64(2*i)/float64(hd)))
	}
	return f
}

func pwMaxDiff(a []float32, b []float64) float64 {
	m := 0.0
	for i := range a {
		m = math.Max(m, math.Abs(float64(a[i])-b[i]))
	}
	return m
}

// checkPW grades one case: the twin within tol, the NeoX control at least 10x outside it.
func checkPW(t *testing.T, name string, pw, neox float64, tol float64) {
	t.Helper()
	t.Logf("%-48s pairwise max|diff| %.3g (tol %.0e) | NeoX control %.3g", name, pw, tol, neox)
	if pw > tol {
		t.Errorf("%s: pairwise kernel max|diff| %.3g > %.0e", name, pw, tol)
	}
	if neox < 10*tol {
		t.Errorf("%s: the NeoX control is within 10x of the tolerance (%.3g): the case cannot tell the rotations apart", name, neox)
	}
}

func TestRopePairwise_f32Kernels(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return p
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(11))
	const hd = 16

	// rope / rope_pw: one region of heads, full and partial rotary, with and without mscale.
	for _, tc := range []struct {
		name         string
		heads, rhalf int
		pos          uint32
		scale        float32
	}{
		{"rope_pw full rotary", 3, 8, 37, 1},
		{"rope_pw partial rotary 8 of 16, mscale 0.85", 3, 4, 53, 0.85},
	} {
		src := make([]float32, tc.heads*hd)
		for i := range src {
			src[i] = rng.Float32()*4 - 2
		}
		invf := pwInvf(tc.rhalf, hd)
		run := func(kernel string) []float32 {
			xb := NewBufferFloats(d, src)
			q.Run1D(pipe(kernel), tc.heads*tc.rhalf, 64, xb, NewBufferFloats(d, invf), NewBufferU32(d, hd), NewBufferU32(d, tc.pos),
				NewBufferU32(d, uint32(tc.heads*tc.rhalf)), NewBufferU32(d, uint32(tc.rhalf)), NewBufferFloats(d, []float32{tc.scale}))
			return xb.Floats()
		}
		ref := make([]float64, len(src))
		for i, v := range src {
			ref[i] = float64(v)
		}
		pwRef(ref, tc.heads, hd, tc.rhalf, func(int) float64 { return float64(tc.pos) }, invf, float64(tc.scale), 1)
		checkPW(t, tc.name, pwMaxDiff(run("rope_pw"), ref), pwMaxDiff(run("rope"), ref), pwTolF32)
	}

	// rope2 / rope2_pw: Q ‖ K ‖ V in one buffer, GQA, the Q temperature on Q only, V untouched.
	for _, tc := range []struct {
		name  string
		rhalf int
		pos   uint32
		scale float32
		qTemp float32
	}{
		{"rope2_pw full rotary, qTemp 1.3", 8, 29, 1, 1.3},
		{"rope2_pw partial 6 of 16, mscale 0.85, qTemp 1.3", 3, 61, 0.85, 1.3},
	} {
		const nH, nKV = 4, 2
		qDim, kvDim := nH*hd, nKV*hd
		src := make([]float32, qDim+2*kvDim)
		for i := range src {
			src[i] = rng.Float32()*4 - 2
		}
		invf := pwInvf(tc.rhalf, hd)
		run := func(kernel string) []float32 {
			xb := NewBufferFloats(d, src)
			q.Run1D(pipe(kernel), (nH+nKV)*tc.rhalf, 64, xb, NewBufferFloats(d, invf), NewBufferU32(d, hd), NewBufferU32(d, tc.pos),
				NewBufferU32(d, uint32(nH*tc.rhalf)), NewBufferU32(d, uint32(nKV*tc.rhalf)), NewBufferU32(d, uint32(tc.rhalf)),
				NewBufferFloats(d, []float32{tc.scale}), NewBufferU32(d, uint32(qDim)), NewBufferFloats(d, []float32{tc.qTemp}))
			return xb.Floats()
		}
		ref := make([]float64, len(src))
		for i, v := range src {
			ref[i] = float64(v)
		}
		at := func(int) float64 { return float64(tc.pos) }
		pwRef(ref[:qDim], nH, hd, tc.rhalf, at, invf, float64(tc.scale), float64(tc.qTemp))
		pwRef(ref[qDim:qDim+kvDim], nKV, hd, tc.rhalf, at, invf, float64(tc.scale), 1)
		got := run("rope2_pw")
		checkPW(t, tc.name, pwMaxDiff(got, ref), pwMaxDiff(run("rope2"), ref), pwTolF32)
		for i := qDim + kvDim; i < len(src); i++ {
			if got[i] != src[i] {
				t.Fatalf("%s: V[%d] changed (%v -> %v): rope2_pw must not touch V", tc.name, i-qDim-kvDim, src[i], got[i])
			}
		}
	}
}

func TestRopePairwise_f16PrefillKernels(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefillKernels: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatalf("pipeline %s: %v", n, err)
		}
		return p
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(12))
	const hd, nH, nKV, M = 16, 4, 2, 37
	stride := (nH + 2*nKV) * hd // a fused qkv row
	for _, tc := range []struct {
		name  string
		rhalf int
		scale float32
	}{
		{"full rotary", 8, 1},
		{"partial 6 of 16, mscale 0.85", 3, 0.85},
	} {
		src := make([]uint16, M*stride)
		for i := range src {
			src[i] = f32ToF16(rng.Float32()*4 - 2)
		}
		invf := pwInvf(tc.rhalf, hd)
		pos := make([]uint32, M)
		pos3 := make([]uint32, 3*M)
		for m := range M {
			pos[m] = uint32(1 + (m*13)%64)
			// Distinct (t,h,w) triples, except every fifth row, where t = h = w (a text row: must equal rope_f16_pw).
			if m%5 == 0 {
				pos3[3*m], pos3[3*m+1], pos3[3*m+2] = pos[m], pos[m], pos[m]
			} else {
				pos3[3*m], pos3[3*m+1], pos3[3*m+2] = uint32(1+m%7), uint32(2+(m*3)%11), uint32(3+(m*5)%13)
			}
		}
		// Contiguous sections [2,3,3] over rhalf 8 (GLM-OCR's layout through MRopeAxisResident), scaled down for rhalf 3.
		axis := make([]uint32, tc.rhalf)
		for dd := range axis {
			switch {
			case dd < tc.rhalf*2/8:
				axis[dd] = 0
			case dd < tc.rhalf*5/8:
				axis[dd] = 1
			default:
				axis[dd] = 2
			}
		}
		// run rotates Q (base 0, nH heads) then K (base nH*hd, nKV heads) of every row, as the batched prefill does.
		run := func(kernel string, mrope bool) []uint16 {
			xb := NewBufferU16s(d, src)
			for _, reg := range [][2]int{{0, nH}, {nH * hd, nKV}} {
				args := []Buffer{xb, NewBufferFloats(d, invf), NewBufferU32(d, hd), NewBufferUint32s(d, pos),
					NewBufferU32(d, uint32(reg[1]*hd)), NewBufferU32(d, uint32(stride)), NewBufferU32(d, uint32(reg[0])),
					NewBufferU32(d, uint32(tc.rhalf)), NewBufferFloats(d, []float32{tc.scale})}
				if mrope {
					args[3] = NewBufferUint32s(d, pos3)
					args = append(args, NewBufferUint32s(d, axis))
				}
				q.Run1D(pipe(kernel), M*reg[1]*tc.rhalf, 128, args...)
			}
			return append([]uint16(nil), xb.U16s()[:M*stride]...)
		}
		ref := func(mrope bool) []float64 {
			r := make([]float64, len(src))
			for i, h := range src {
				r[i] = float64(f16ToF32(h))
			}
			for m := range M {
				posOf := func(int) float64 { return float64(pos[m]) }
				if mrope {
					posOf = func(dd int) float64 { return float64(pos3[3*m+int(axis[dd])]) }
				}
				row := r[m*stride:]
				pwRef(row[:nH*hd], nH, hd, tc.rhalf, posOf, invf, float64(tc.scale), 1)
				pwRef(row[nH*hd:(nH+nKV)*hd], nKV, hd, tc.rhalf, posOf, invf, float64(tc.scale), 1)
			}
			return r
		}
		f := func(h []uint16) []float32 {
			o := make([]float32, len(h))
			for i, v := range h {
				o[i] = f16ToF32(v)
			}
			return o
		}
		scalarPW := run("rope_f16_pw", false)
		checkPW(t, "rope_f16_pw "+tc.name, pwMaxDiff(f(scalarPW), ref(false)), pwMaxDiff(f(run("rope_f16", false)), ref(false)), pwTolF16)
		mropePW := run("rope_mrope_f16_pw", true)
		checkPW(t, "rope_mrope_f16_pw "+tc.name, pwMaxDiff(f(mropePW), ref(true)), pwMaxDiff(f(run("rope_mrope_f16", true)), ref(true)), pwTolF16)
		for m := range M {
			for j := (nH + nKV) * hd; j < stride; j++ { // V: neither kernel touches it
				if i := m*stride + j; scalarPW[i] != src[i] || mropePW[i] != src[i] {
					t.Fatalf("%s: row %d V[%d] changed", tc.name, m, j-(nH+nKV)*hd)
				}
			}
			if m%5 != 0 {
				continue
			}
			for j := range stride { // t = h = w: bit-identical to the scalar pairwise kernel
				if i := m*stride + j; mropePW[i] != scalarPW[i] {
					t.Fatalf("%s: row %d (t=h=w=%d) element %d: rope_mrope_f16_pw %#04x != rope_f16_pw %#04x", tc.name, m, pos[m], j, mropePW[i], scalarPW[i])
				}
			}
		}
	}
}
