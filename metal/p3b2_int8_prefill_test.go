//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// TestP3b2_storeMatchesKvStoreI8 is P3B2-G1 (i) (docs/tasks/task-kv-precision-everywhere-2026-10.md, "P3b-2 gates"):
// kv_store_i8_prefill, storing M rows of a pass's half K/V at scattered positions, writes the codes and scales kv_store_i8
// writes for the same values widened to f32, one row at a time; a zero head (scale 1) included. kv_dequant_i8_f16's output
// is half(code × scale), each element rounded as the host rounds it.
func TestP3b2_storeMatchesKvStoreI8(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(allKernels+kvI8PrefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	pipe := func(n string) Pipeline {
		p, err := d.NewComputePipeline(lib, n)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	pRef, pPre, pDeq := pipe("kv_store_i8"), pipe("kv_store_i8_prefill"), pipe("kv_dequant_i8_f16")
	rng := rand.New(rand.NewSource(20261022))
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	for _, hd := range []int{64, 128} {
		const nH, nKV, M, ctx = 4, 2, 37, 96
		kvDim := nKV * hd
		stride, kOff, vOff := nH*hd+2*kvDim, nH*hd, nH*hd+kvDim
		qkv := make([]uint16, M*stride)
		for i := range qkv {
			qkv[i] = f32ToF16(float32(rng.NormFloat64()) * 3)
		}
		for i := range hd { // row 5's first K head is all zero: scale 1, codes 0
			qkv[5*stride+kOff+i] = 0
		}
		pos := rng.Perm(ctx)[:M]
		pos32 := make([]uint32, M)
		for i, p := range pos {
			pos32[i] = uint32(p)
		}
		kc, vc := d.NewBufferBytes(ctx*kvDim), d.NewBufferBytes(ctx*kvDim)
		ks, vs := d.NewBufferLen(ctx*nKV), d.NewBufferLen(ctx*nKV)
		e := q.Begin()
		e.Dispatch(pPre, M*nKV, 32, NewBufferU16s(d, qkv), NewBufferUint32s(d, pos32), kc, vc, ks, vs, u(nKV), u(hd), u(stride), u(kOff), u(vOff), u(M))
		e.End()
		rkc, rvc := d.NewBufferBytes(ctx*kvDim), d.NewBufferBytes(ctx*kvDim)
		rks, rvs := d.NewBufferLen(ctx*nKV), d.NewBufferLen(ctx*nKV)
		for m := range M {
			k, v := make([]float32, kvDim), make([]float32, kvDim)
			for i := range kvDim {
				k[i], v[i] = f16ToF32(qkv[m*stride+kOff+i]), f16ToF32(qkv[m*stride+vOff+i])
			}
			e := q.Begin()
			e.Dispatch(pRef, nKV, 1, NewBufferFloats(d, k), NewBufferFloats(d, v), rkc, rvc, rks, rvs, u(nKV), u(hd), u(pos[m]))
			e.End()
		}
		codes, scales := 0, 0
		for i := range ctx * kvDim {
			if kc.Int8s()[i] != rkc.Int8s()[i] || vc.Int8s()[i] != rvc.Int8s()[i] {
				codes++
			}
		}
		for i := range ctx * nKV {
			if ks.Floats()[i] != rks.Floats()[i] || vs.Floats()[i] != rvs.Floats()[i] {
				scales++
			}
		}
		// the dequantize, over every row the store wrote and the rows it did not (zero codes and scales)
		kOut, vOut := NewBufferU16s(d, make([]uint16, ctx*kvDim)), NewBufferU16s(d, make([]uint16, ctx*kvDim))
		e = q.Begin()
		e.Dispatch(pDeq, ctx*kvDim, 256, kc, vc, ks, vs, kOut, vOut, u(nKV), u(hd), u(ctx))
		e.End()
		deq := 0
		for i := range ctx * kvDim {
			p, h := i/kvDim, (i%kvDim)/hd
			wk := f32ToF16(float32(kc.Int8s()[i]) * ks.Floats()[p*nKV+h])
			wv := f32ToF16(float32(vc.Int8s()[i]) * vs.Floats()[p*nKV+h])
			if kOut.U16s()[i] != wk || vOut.U16s()[i] != wv {
				deq++
			}
		}
		fmt.Fprintf(os.Stderr, "[P3B2-G1] hd=%d, %d rows at scattered positions: %d codes and %d scales differ from kv_store_i8; %d dequantized elements differ from the host's\n", hd, M, codes, scales, deq)
		if codes != 0 || scales != 0 || deq != 0 {
			t.Errorf("hd=%d: %d codes, %d scales, %d dequantized elements differ", hd, codes, scales, deq)
		}
		if !slices.Contains(ks.Floats(), 1) {
			t.Errorf("hd=%d: no head stored the zero-row scale 1: the zero head did not reach the store", hd)
		}
	}
}
