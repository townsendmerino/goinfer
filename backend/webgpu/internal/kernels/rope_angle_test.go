//go:build gpu

package kernels

import (
	"math"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestRope_matchesCPUAngles holds the rotation the RoPE kernels apply to the CPU's float64 cos/sin of the same f32 angle (the shader's
// f32(pos)·invFreq), to about f32 accuracy on any adapter, at positions from 1 to 500000: the rope kernel (Q), ropeStore (K) and the
// fused qkvFinalize (Q and K). A WGSL builtin sin/cos may carry an absolute error far larger than f32's (2^-11 is conformant), and a
// floating-point range reduction loses its precision where a driver reassociates it; ropeCosSin does neither.
func TestRope_matchesCPUAngles(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatalf("ensureAttn: %v", err)
	}
	const nH, nKV, hd = 4, 2, 128
	const half, maxLen = hd / 2, 8
	kvDim := nKV * hd
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/hd))
	}
	rot := func(src []float32, heads, pos int) []float32 {
		out := make([]float32, len(src))
		for h := range heads {
			off := h * hd
			for d := range half {
				th := float64(float32(pos) * invFreq[d]) // the shader's own f32 angle
				cs, sn := math.Cos(th), math.Sin(th)
				x1, x2 := float64(src[off+d]), float64(src[off+half+d])
				out[off+d] = float32(x1*cs - x2*sn)
				out[off+half+d] = float32(x2*cs + x1*sn)
			}
		}
		return out
	}
	// bar: a few f32 roundings of the inputs' size, far under a 6e-5 sin/cos error.
	check := func(label string, pos int, got, want, src []float32) {
		t.Helper()
		for i := range want {
			if e, tol := math.Abs(float64(got[i]-want[i])), 2e-6*(1+math.Abs(float64(src[i]))); e > tol {
				t.Errorf("%s pos %d: element %d = %v, CPU %v (|diff| %.3g > %.3g)", label, pos, i, got[i], want[i], e, tol)
				return
			}
		}
	}
	rng := i8rng()
	for _, pos := range []int{1, 2, 3, 5, 7, 100, 1000, 4095, 32767, 65535, 500000} {
		slot := pos % maxLen // the cache row written; pos is the rotation's
		q, k, v := randF(rng, nH*hd), randF(rng, kvDim), randF(rng, kvDim)

		qb := c.sbuf(q)
		uni := c.ubuf([]uint32{nH, hd, half, uint32(pos), math.Float32bits(1), 0, 0, 0})
		if err := c.dispatchI8(c.ropePipeline, c.ropeLayout, (nH*half+63)/64, []*wgpu.Buffer{qb, c.sbuf(invFreq)}, uni); err != nil {
			t.Fatalf("rope: %v", err)
		}
		check("rope Q", pos, c.readF(qb, nH*hd), rot(q, nH, pos), q)

		kc := c.zbuf(maxLen * kvDim)
		suni := c.ubuf([]uint32{nKV, hd, half, uint32(pos), math.Float32bits(1), uint32(slot * kvDim), 0, 0})
		if err := c.dispatchI8(c.ropeStorePipeline, c.ropeStoreLayout, (nKV*half+63)/64, []*wgpu.Buffer{c.sbuf(k), c.sbuf(invFreq), kc}, suni); err != nil {
			t.Fatalf("ropeStore: %v", err)
		}
		check("ropeStore K", pos, c.readF(kc, maxLen*kvDim)[slot*kvDim:(slot+1)*kvDim], rot(k, nKV, pos), k)

		qb2, kc2, vc2 := c.sbuf(q), c.zbuf(maxLen*kvDim), c.zbuf(maxLen*kvDim)
		funi := c.ubuf([]uint32{nH, nKV, hd, half, uint32(pos), uint32(slot * kvDim), math.Float32bits(1), uint32(kvDim)})
		if err := c.dispatchI8(c.qkvFinPipeline, c.qkvFinLayout, (max(nH*half, kvDim)+63)/64,
			[]*wgpu.Buffer{qb2, c.sbuf(k), c.sbuf(v), c.sbuf(invFreq), kc2, vc2}, funi); err != nil {
			t.Fatalf("qkvFinalize: %v", err)
		}
		check("qkvFinalize Q", pos, c.readF(qb2, nH*hd), rot(q, nH, pos), q)
		check("qkvFinalize K", pos, c.readF(kc2, maxLen*kvDim)[slot*kvDim:(slot+1)*kvDim], rot(k, nKV, pos), k)
	}
}

// TestRope_mlaAndBatchedMatchCPUAngles is TestRope_matchesCPUAngles for the other kernels that write f32: MLA's query rope and latent
// store (NeoX and the interleaved GPT-J layout), and the batched prefill rope over several rows.
func TestRope_mlaAndBatchedMatchCPUAngles(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureMLAQRope(); err != nil {
		t.Fatalf("ensureMLAQRope: %v", err)
	}
	if err := c.ensureMLAStore(); err != nil {
		t.Fatalf("ensureMLAStore: %v", err)
	}
	if err := c.ensurePrefillBatched(); err != nil {
		t.Fatalf("ensurePrefillBatched: %v", err)
	}
	const half = 32
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/(2*half)))
	}
	cosSin := func(pos, d int) (float64, float64) {
		th := float64(float32(pos) * invFreq[d])
		return math.Cos(th), math.Sin(th)
	}
	// pair returns the (a, b) operands of pair i in a qkRope-wide span: [evens|odds] interleaved, or the two NeoX halves.
	pair := func(span []float32, i int, inter bool) (float64, float64) {
		if inter {
			return float64(span[2*i]), float64(span[2*i+1])
		}
		return float64(span[i]), float64(span[half+i])
	}
	near := func(label string, pos int, got, want float64, mag float64) bool {
		if e, tol := math.Abs(got-want), 2e-6*(1+mag); e > tol {
			t.Errorf("%s pos %d: %v, CPU %v (|diff| %.3g > %.3g)", label, pos, got, want, e, tol)
			return false
		}
		return true
	}
	rng := i8rng()
	for _, pos := range []int{3, 1000, 65535, 500000} {
		for _, inter := range []bool{false, true} {
			// mlaQRope: nH heads of [qkNope | qkRope]; the rope half lands at qAbs[h*latDim + rank ..].
			const nH, qkNope, rank = 2, 8, 16
			qkRope, latDim := 2*half, rank+2*half
			qkHead := qkNope + qkRope
			q := randF(rng, nH*qkHead)
			qAbs := c.zbuf(nH * latDim)
			u := c.ubuf([]uint32{nH, uint32(qkHead), qkNope, uint32(qkRope), rank, uint32(latDim), uint32(pos), BoolU32(inter), math.Float32bits(1), 0, 0, 0})
			if err := c.dispatchI8(c.mlaQRopePipeline, c.mlaQRopeLayout, (nH*half+63)/64, []*wgpu.Buffer{c.sbuf(q), c.sbuf(invFreq), qAbs}, u); err != nil {
				t.Fatalf("mlaQRope: %v", err)
			}
			got := c.readF(qAbs, nH*latDim)
		qrope:
			for h := range nH {
				span := q[h*qkHead+qkNope : (h+1)*qkHead]
				for i := range half {
					cs, sn := cosSin(pos, i)
					a, b := pair(span, i, inter)
					o := h*latDim + rank
					if !near("mlaQRope", pos, float64(got[o+i]), a*cs-b*sn, math.Abs(a)+math.Abs(b)) ||
						!near("mlaQRope", pos, float64(got[o+half+i]), b*cs+a*sn, math.Abs(a)+math.Abs(b)) {
						break qrope
					}
				}
			}

			// mlaLatentStore: the rope key krj after the rank block at base 0.
			kvDown := randF(rng, rank+qkRope)
			normW := make([]float32, rank)
			for i := range normW {
				normW[i] = 1
			}
			lat := c.zbuf(latDim)
			su := c.ubuf([]uint32{rank, uint32(qkRope), uint32(pos), math.Float32bits(1e-6), 0, math.Float32bits(1), BoolU32(inter), 0})
			if err := c.dispatchI8(c.mlaStorePipeline, c.mlaStoreLayout, 1, []*wgpu.Buffer{c.sbuf(kvDown), c.sbuf(normW), c.sbuf(invFreq), lat}, su); err != nil {
				t.Fatalf("mlaLatentStore: %v", err)
			}
			gl := c.readF(lat, latDim)
			span := kvDown[rank:]
			for i := range half {
				cs, sn := cosSin(pos, i)
				a, b := pair(span, i, inter)
				if !near("mlaLatentStore", pos, float64(gl[rank+i]), a*cs-b*sn, math.Abs(a)+math.Abs(b)) ||
					!near("mlaLatentStore", pos, float64(gl[rank+half+i]), b*cs+a*sn, math.Abs(a)+math.Abs(b)) {
					break
				}
			}
		}

		// ropeBatched: M rows at positions pos..pos+M-1.
		const heads, M = 2, 3
		hd := 2 * half
		vec := randF(rng, M*heads*hd)
		vb := c.sbuf(vec)
		bu := c.ubuf([]uint32{heads, uint32(hd), half, uint32(pos), math.Float32bits(1), 0, 0, 0})
		if err := c.dispatchI8Grid(c.ropeBatchedPipeline, c.ropeBatchedLayout, (heads*half+63)/64, M, []*wgpu.Buffer{vb, c.sbuf(invFreq)}, bu); err != nil {
			t.Fatalf("ropeBatched: %v", err)
		}
		gb := c.readF(vb, M*heads*hd)
	rows:
		for r := range M {
			for h := range heads {
				off := (r*heads + h) * hd
				for d := range half {
					cs, sn := cosSin(pos+r, d)
					x1, x2 := float64(vec[off+d]), float64(vec[off+half+d])
					if !near("ropeBatched", pos+r, float64(gb[off+d]), x1*cs-x2*sn, math.Abs(x1)+math.Abs(x2)) ||
						!near("ropeBatched", pos+r, float64(gb[off+half+d]), x2*cs+x1*sn, math.Abs(x1)+math.Abs(x2)) {
						break rows
					}
				}
			}
		}
	}
}

// dispatchI8Grid is dispatchI8 with a two-dimensional grid (x, y).
func (c *Context) dispatchI8Grid(pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, gx, gy int, storage []*wgpu.Buffer, unis ...*wgpu.Buffer) error {
	entries := make([]wgpu.BindGroupEntry, 0, len(storage)+len(unis))
	for i, b := range storage {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()})
	}
	for i, u := range unis {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(storage) + i), Buffer: u, Size: u.GetSize()})
	}
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: entries})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(gx), uint32(gy), 1)
	pass.TryEnd()
	cb, _ := enc.TryFinish(nil)
	c.queue.Submit(cb)
	c.device.Poll(true, nil)
	return nil
}
