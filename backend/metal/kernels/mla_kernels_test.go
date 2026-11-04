//go:build darwin && goinfer_testhooks

package kernels

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// kmlaGeom is K-MLA's synthetic geometry: wider than the tiny fixtures', with a rope slice large enough to matter.
const (
	kmlaRank, kmlaQKRope, kmlaQKNope, kmlaNH = 64, 16, 32, 4
	kmlaQKHead, kmlaLatDim                   = kmlaQKNope + kmlaQKRope, kmlaRank + kmlaQKRope
)

var kmlaPositions = []int{0, 1, 7, 255, 4095}

// kmlaWorst runs mla_store and mla_qrope, compiled from src, on non-flat synthetic inputs (values in ±4, a base-10,000 rope
// table) for one rope branch at every K-MLA position, and returns each kernel's worst max|error| / max|reference| against a
// float64 reference.
func kmlaWorst(t *testing.T, src string, interleave bool) (store, qrope float64) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pStore, err := d.NewComputePipeline(lib, "mla_store")
	if err != nil {
		t.Fatal(err)
	}
	pQRope, err := d.NewComputePipeline(lib, "mla_qrope")
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	vals := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.Float64()*8 - 4)
		}
		return v
	}
	hr := kmlaQKRope / 2
	invf := make([]float32, hr)
	for i := range invf {
		invf[i] = float32(math.Pow(10000, -2*float64(i)/kmlaQKRope))
	}
	il := uint32(0)
	if interleave {
		il = 1
	}
	const eps = float32(1e-6)
	// rot is the kernel's rotation of the pair (a, b) at position pos, in float64.
	rot := func(a, b float32, i, pos int) (float64, float64) {
		th := float64(pos) * float64(invf[i])
		c, s := math.Cos(th), math.Sin(th)
		return float64(a)*c - float64(b)*s, float64(b)*c + float64(a)*s
	}
	pair := func(v []float32, base, i int) (float32, float32) {
		if interleave {
			return v[base+2*i], v[base+2*i+1]
		}
		return v[base+i], v[base+hr+i]
	}
	ratio := func(got []float32, want []float64) float64 {
		var e, m float64
		for i := range want {
			e, m = max(e, math.Abs(float64(got[i])-want[i])), max(m, math.Abs(want[i]))
		}
		return e / m
	}
	for _, pos := range kmlaPositions {
		kv, normW := vals(kmlaLatDim), vals(kmlaRank)
		lat := d.NewBufferLen((pos + 1) * kmlaLatDim)
		q := d.NewCommandQueue()
		e := q.Begin()
		e.Dispatch(pStore, 64, 64, NewBufferFloats(d, kv), NewBufferFloats(d, normW), NewBufferFloats(d, invf), lat,
			NewBufferUint32s(d, []uint32{kmlaRank, kmlaQKRope, il}), NewBufferU32(d, uint32(pos)), NewBufferU32(d, uint32(pos)),
			NewBufferFloats(d, []float32{eps}), NewBufferFloats(d, []float32{1}))
		e.End()
		var ss float64
		for i := range kmlaRank {
			ss += float64(kv[i]) * float64(kv[i])
		}
		inv := 1 / math.Sqrt(ss/kmlaRank+float64(eps))
		want := make([]float64, kmlaLatDim)
		for i := range kmlaRank {
			want[i] = float64(kv[i]) * inv * float64(normW[i])
		}
		for i := range hr {
			a, b := pair(kv, kmlaRank, i)
			want[kmlaRank+i], want[kmlaRank+hr+i] = rot(a, b, i, pos)
		}
		store = max(store, ratio(lat.Floats()[pos*kmlaLatDim:(pos+1)*kmlaLatDim], want))

		qv := vals(kmlaNH * kmlaQKHead)
		qAbs := d.NewBufferLen(kmlaNH * kmlaLatDim)
		e = q.Begin()
		e.Dispatch(pQRope, kmlaNH*hr, 64, NewBufferFloats(d, qv), NewBufferFloats(d, invf), qAbs,
			NewBufferUint32s(d, []uint32{kmlaNH, kmlaQKHead, kmlaQKNope, kmlaQKRope, kmlaRank, kmlaLatDim, il}),
			NewBufferU32(d, uint32(pos)), NewBufferFloats(d, []float32{1}))
		e.End()
		got := qAbs.Floats()
		for h := range kmlaNH {
			w := make([]float64, kmlaQKRope)
			for i := range hr {
				a, b := pair(qv, h*kmlaQKHead+kmlaQKNope, i)
				w[i], w[hr+i] = rot(a, b, i, pos)
			}
			qrope = max(qrope, ratio(got[h*kmlaLatDim+kmlaRank:(h+1)*kmlaLatDim], w))
		}
	}
	return store, qrope
}

// TestMLAKernels_rope is OP2a's K-MLA gate (docs/tasks/task-forward-architecture-2026-10.md §4.7, registered before it ran):
// mla_store and mla_qrope against a float64 reference on non-flat inputs, both rope branches, at positions 0, 1, 7, 255 and
// 4,095, max|error| <= 2e-3 x max|reference|. With q's and the latent's rope and nope slices swapped in each branch (the
// plant), both kernels must exceed it. The tiny fixtures cannot see this defect (gate M's amendment 2).
func TestMLAKernels_rope(t *testing.T) {
	const bar = 2e-3
	plant := strings.NewReplacer(
		"uint qbase = h * qkHead + qkNope;", "uint qbase = h * qkHead;",
		"if (interleave == 1u) { a = kvDown[rank + 2u*i]; b = kvDown[rank + 2u*i + 1u]; }", "if (interleave == 1u) { a = kvDown[2u*i]; b = kvDown[2u*i + 1u]; }",
		"else { a = kvDown[rank + i]; b = kvDown[rank + hr + i]; }", "else { a = kvDown[i]; b = kvDown[hr + i]; }",
	).Replace(AllKernels)
	if plant == AllKernels {
		t.Fatal("the plant's targets are not in the MLA kernels")
	}
	for _, il := range []bool{false, true} {
		s, q := kmlaWorst(t, AllKernels, il)
		ps, pq := kmlaWorst(t, plant, il)
		t.Logf("interleave %v: mla_store %.2e, mla_qrope %.2e (bar %.0e); planted %.2e, %.2e", il, s, q, bar, ps, pq)
		if s > bar || q > bar {
			t.Errorf("interleave %v: a kernel exceeds the bar (store %.2e, qrope %.2e)", il, s, q)
		}
		if ps <= bar || pq <= bar {
			t.Errorf("interleave %v: the rope/nope swap does not turn both kernels red (store %.2e, qrope %.2e)", il, ps, pq)
		}
	}
}
