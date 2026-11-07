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
	defer d.ReleaseAll() // runs first: the buffers are released before the device, as Close does
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
		e.Dispatch(pPre, M*nKV, 32, NewBufferU16s(d, qkv), NewBufferUint32s(d, pos32), kc, vc, ks, vs, u(nKV), u(hd), u(stride), u(kOff), u(vOff), u(M), kvIdentityTable(d))
		e.End()
		rkc, rvc := d.NewBufferBytes(ctx*kvDim), d.NewBufferBytes(ctx*kvDim)
		rks, rvs := d.NewBufferLen(ctx*nKV), d.NewBufferLen(ctx*nKV)
		for m := range M {
			k, v := make([]float32, kvDim), make([]float32, kvDim)
			for i := range kvDim {
				k[i], v[i] = f16ToF32(qkv[m*stride+kOff+i]), f16ToF32(qkv[m*stride+vOff+i])
			}
			e := q.Begin()
			e.Dispatch(pRef, nKV, 1, NewBufferFloats(d, k), NewBufferFloats(d, v), rkc, rvc, rks, rvs, u(nKV), u(hd), u(pos[m]), kvIdentityTable(d))
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
		e.Dispatch(pDeq, ctx*kvDim, 256, kc, vc, ks, vs, kOut, vOut, u(nKV), u(hd), u(ctx), kvIdentityTable(d))
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

// kvPrefillLib compiles the prefill library, the int8 cache's two kernels with it, as ensurePrefill does, with no softcap, and returns its pipelines by name.
func kvPrefillLib(t *testing.T) (*Device, func(string) Pipeline, Queue) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	src, err := withAttnSoftcap(prefillKernels+kvI8PrefillKernels, 0)
	if err != nil {
		t.Fatal(err)
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	pipe := func(name string) Pipeline {
		t.Helper()
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	return d, pipe, d.NewCommandQueue()
}

// The pass these tests run: rows [startPos, startPos+M) of a sequence whose keys reach 300, so the pass starts inside a block and
// crosses the edge at 256, and the keys it reads cross four. inOrder is a sequence's blocks as one slot holds them; scrambled names
// the same five blocks out of order.
const kvpStart, kvpM, kvpBlocks = 236, 64, 5

var kvpInOrder, kvpScrambled = []uint32{1, 2, 3, 4, 5}, []uint32{5, 2, 4, 1, 3}

func u16Differ(a, b []uint16) int {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

// TestKVPages_prefillI8StoreWritesTheTable: kv_store_i8_prefill puts row m's codes and scales at its position's row through the
// table, so a scrambled pool read back through its table holds what an in-order one does.
func TestKVPages_prefillI8StoreWritesTheTable(t *testing.T) {
	d, pipeOf, cq := kvPrefillLib(t)
	const nKV, hd = 2, 64
	const kvDim, qStride = nKV * hd, 4*hd + 2*nKV*hd
	const kOff, vOff = 4 * hd, 4*hd + kvDim
	rng := rand.New(rand.NewSource(7))
	qkv := make([]uint16, kvpM*qStride)
	for i := range qkv {
		qkv[i] = f32ToF16(float32(rng.NormFloat64()))
	}
	pos := make([]uint32, kvpM)
	for m := range pos {
		pos[m] = uint32(kvpStart + m)
	}
	poolRows := (kvpBlocks + 1) * 64
	row := func(perm []uint32, p int) int { return int(perm[p>>6])*64 + p&63 }
	t.Run("kv_store_i8_prefill", func(t *testing.T) {
		pipe := pipeOf("kv_store_i8_prefill")
		run := func(perm []uint32) (codes []int8, scales []float32) {
			kc, vc := d.NewBufferBytes(poolRows*kvDim), d.NewBufferBytes(poolRows*kvDim)
			ks, vs := d.NewBufferLen(poolRows*nKV), d.NewBufferLen(poolRows*nKV)
			e := cq.Begin()
			e.Dispatch(pipe, kvpM*nKV, 32, NewBufferU16s(d, qkv), NewBufferUint32s(d, pos), kc, vc, ks, vs, NewBufferU32(d, nKV), NewBufferU32(d, hd),
				NewBufferU32(d, qStride), NewBufferU32(d, kOff), NewBufferU32(d, vOff), NewBufferU32(d, kvpM), NewBufferUint32s(d, perm))
			e.End()
			if err := e.Err(); err != nil {
				t.Fatal(err)
			}
			for p := kvpStart; p < kvpStart+kvpM; p++ {
				r := row(perm, p)
				codes = append(codes, kc.Int8s()[r*kvDim:(r+1)*kvDim]...)
				codes = append(codes, vc.Int8s()[r*kvDim:(r+1)*kvDim]...)
				scales = append(scales, ks.Floats()[r*nKV:(r+1)*nKV]...)
				scales = append(scales, vs.Floats()[r*nKV:(r+1)*nKV]...)
			}
			return codes, scales
		}
		c1, s1 := run(kvpInOrder)
		c2, s2 := run(kvpScrambled)
		if !slices.Equal(c1, c2) || !slices.Equal(s1, s2) {
			t.Error("the scrambled int8 pool, read through its table, holds other codes or scales than the in-order one")
		}
		for i, s := range s2 {
			if s == 0 {
				t.Fatalf("scale %d of the stored rows is 0: the row was not written where its table says", i)
			}
		}
	})
}

// TestKVPages_prefillDequantReadsTheTable: kv_dequant_i8_f16 reads a layer's int8 rows and scales through the table into a contiguous
// scratch, so a scrambled cache dequantizes to the in-order one's scratch bit for bit, and a swapped table does not.
func TestKVPages_prefillDequantReadsTheTable(t *testing.T) {
	d, pipeOf, cq := kvPrefillLib(t)
	pipe := pipeOf("kv_dequant_i8_f16")
	const nKV, hd = 2, 64
	const kvDim, n = nKV * hd, kvpStart + kvpM
	rng := rand.New(rand.NewSource(11))
	codes := make([]int8, kvpBlocks*64*kvDim)
	for i := range codes {
		codes[i] = int8(rng.Intn(255) - 127)
	}
	scales := make([]float32, kvpBlocks*64*nKV)
	for i := range scales {
		scales[i] = 0.01 + rng.Float32()
	}
	run := func(perm []uint32, swap bool) []uint16 {
		tab := slices.Clone(perm)
		if swap {
			tab[1], tab[2] = tab[2], tab[1]
		}
		kc, vc := d.NewBufferBytes((kvpBlocks+1)*64*kvDim), d.NewBufferBytes((kvpBlocks+1)*64*kvDim)
		copy(kc.Int8s(), kvPagedRows(codes, kvDim, perm))
		copy(vc.Int8s(), kvPagedRows(codes, kvDim, perm))
		ks := NewBufferFloats(d, kvPagedRows(scales, nKV, perm))
		vs := NewBufferFloats(d, kvPagedRows(scales, nKV, perm))
		kOut, vOut := d.NewBufferBytes(n*kvDim*2), d.NewBufferBytes(n*kvDim*2)
		e := cq.Begin()
		e.Dispatch(pipe, n*kvDim, 256, kc, vc, ks, vs, kOut, vOut, NewBufferU32(d, nKV), NewBufferU32(d, hd), NewBufferU32(d, n), NewBufferUint32s(d, tab))
		e.End()
		if err := e.Err(); err != nil {
			t.Fatal(err)
		}
		return append(slices.Clone(kOut.U16s()[:n*kvDim]), vOut.U16s()[:n*kvDim]...)
	}
	want := run(kvpInOrder, false)
	if c := u16Differ(want, run(kvpScrambled, false)); c != 0 {
		t.Errorf("%d of %d dequantized values differ between scrambled and in-order blocks", c, len(want))
	}
	if c := u16Differ(want, run(kvpInOrder, true)); c == 0 {
		t.Error("two table entries swapped moved no dequantized value: the kernel does not read the table")
	}
}
