//go:build darwin

package kernels

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// kvPrefillLib compiles the prefill library and returns its pipelines by name.
func kvPrefillLib(t *testing.T) (*Device, func(string) Pipeline, Queue) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
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
const kvpStart, kvpM, kvpBlocks, kvpBlockRows, kvpTG = 236, 64, 5, 64, 128 // kvpBlockRows is KVBlockRows in the decoder; kvpTG the prefill kernels' threadgroup

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

// kvPagedRows lays src, rows of width elements, out block by block at perm's physical blocks, past an empty null block.
func kvPagedRows[T any](src []T, width int, perm []uint32) []T {
	out := make([]T, (len(perm)+1)*kvpBlockRows*width)
	for i, at := range perm {
		from, n := i*kvpBlockRows*width, kvpBlockRows*width
		if from+n > len(src) {
			n = len(src) - from
		}
		copy(out[int(at)*kvpBlockRows*width:], src[from:from+n])
	}
	return out
}

// TestKVPages_prefillKernelsReadTheTable: every prefill attention kernel reads a key through its block table. Rows laid out block by
// block in a scrambled order, with the table that names them, give the in-order layout's output bit for bit; the same pool under a
// table with two entries swapped does not; and each of a set of key rows on either side of a block edge, and the value row at each,
// moves the output alone, which both arms would share a wrong offset inside a block for. The window case starts a row's keys inside
// a block.
func TestKVPages_prefillKernelsReadTheTable(t *testing.T) {
	d, pipeOf, cq := kvPrefillLib(t)
	const nKV, G = 2, 2
	const nH = nKV * G
	for _, c := range []struct {
		kernel string
		hd     int
		grid   func() (int, int)
		extra  func() []Buffer
	}{
		{"attention_prefill", 128, func() (int, int) { return kvpM * nH * kvpTG, kvpTG }, nil},
		{"attention_prefill_tiled", 128, func() (int, int) { return kvpM * nH * kvpTG, kvpTG },
			func() []Buffer { return []Buffer{NewBufferU32(d, 48)} }},
		{"attention_prefill_img", 128, func() (int, int) { return kvpM * nH * kvpTG, kvpTG },
			func() []Buffer { return []Buffer{NewBufferUint32s(d, []uint32{250, 270}), NewBufferU32(d, 1)} }},
		{"attention_prefill_fused", 128, func() (int, int) { return (nH*((kvpM+7)/8) + 3) / 4 * 4 * 32, 128 },
			func() []Buffer { return []Buffer{NewBufferU32(d, kvpM)} }},
		{"attention_prefill_steel", 128, func() (int, int) { return nH * ((kvpM + 31) / 32) * 128, 128 },
			func() []Buffer { return []Buffer{NewBufferU32(d, kvpM)} }},
		{"attention_prefill_steel_hd64", 64, func() (int, int) { return nH * ((kvpM + 31) / 32) * 128, 128 },
			func() []Buffer { return []Buffer{NewBufferU32(d, kvpM)} }},
		{"attention_prefill_steel_hd256", 256, func() (int, int) { return nH * ((kvpM + 31) / 32) * 256, 256 },
			func() []Buffer { return []Buffer{NewBufferU32(d, kvpM)} }},
		{"attention_prefill_steel_hd512", 512, func() (int, int) { return nH * ((kvpM + 15) / 16) * 256, 256 },
			func() []Buffer { return []Buffer{NewBufferU32(d, kvpM)} }},
	} {
		for _, window := range []uint32{0, 150} {
			t.Run(c.kernel+map[bool]string{true: "/window", false: ""}[window > 0], func(t *testing.T) {
				pipe := pipeOf(c.kernel)
				kvDim := nKV * c.hd
				qStride := nH*c.hd + 2*kvDim
				rng := rand.New(rand.NewSource(int64(c.hd) + int64(window)))
				rows := func(n int) []uint16 {
					out := make([]uint16, n)
					for i := range out {
						out[i] = f32ToF16(float32(rng.NormFloat64()))
					}
					return out
				}
				kl, vl := rows(kvpBlocks*64*kvDim), rows(kvpBlocks*64*kvDim)
				q := rows(kvpM * qStride)
				run := func(perm []uint32, swap bool) []uint16 {
					tab := slices.Clone(perm)
					if swap {
						tab[1], tab[2] = tab[2], tab[1]
					}
					out := d.NewBufferBytes(kvpM * nH * c.hd * 2)
					args := []Buffer{NewBufferU16s(d, q), NewBufferU16s(d, kvPagedRows(kl, kvDim, perm)), NewBufferU16s(d, kvPagedRows(vl, kvDim, perm)), out,
						NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, uint32(c.hd)), NewBufferU32(d, kvpStart),
						NewBufferFloats(d, []float32{float32(1 / math.Sqrt(float64(c.hd)))}), NewBufferU32(d, uint32(qStride)), NewBufferU32(d, window)}
					if c.extra != nil {
						args = append(args, c.extra()...)
					}
					args = append(args, NewBufferUint32s(d, tab))
					total, tg := c.grid()
					e := cq.Begin()
					e.Dispatch(pipe, total, tg, args...)
					e.End()
					if err := e.Err(); err != nil {
						t.Fatal(err)
					}
					return slices.Clone(out.U16s()[:kvpM*nH*c.hd]) // NewBufferBytes counts bytes, and U16s that many elements
				}
				want := run(kvpInOrder, false)
				if n := u16Differ(want, run(kvpScrambled, false)); n != 0 {
					t.Errorf("%d of %d outputs differ between scrambled and in-order blocks", n, len(want))
				}
				if n := u16Differ(want, run(kvpInOrder, true)); n == 0 {
					t.Error("two table entries swapped moved no output: the kernel does not read the table")
				}
				if window > 0 {
					return
				}
				for _, row := range []int{0, 63, 64, 127, 128, 191, 192, 236, 255, 256, 299} {
					k0 := kl[row*kvDim]
					kl[row*kvDim] = f32ToF16(f16ToF32(k0) + 8)
					n := u16Differ(want, run(kvpInOrder, false))
					kl[row*kvDim] = k0
					if n == 0 {
						t.Errorf("a change to key row %d alone moved no output: the kernel does not read that row", row)
					}
					v0 := vl[row*kvDim]
					vl[row*kvDim] = f32ToF16(f16ToF32(v0) + 8)
					n = u16Differ(want, run(kvpInOrder, false))
					vl[row*kvDim] = v0
					if n == 0 {
						t.Errorf("a change to value row %d alone moved no output: the kernel does not read that row", row)
					}
				}
			})
		}
	}
}

// TestKVPages_prefillStoreWritesTheTable: kv_store_f16 puts row m at its position's row through the table, so a scrambled pool read
// back through its table holds what an in-order one does, and the stored rows are where the table says.
func TestKVPages_prefillStoreWritesTheTable(t *testing.T) {
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
	poolRows := (kvpBlocks + 1) * kvpBlockRows
	row := func(perm []uint32, p int) int { return int(perm[p>>6])*64 + p&63 }
	t.Run("kv_store_f16", func(t *testing.T) {
		pipe := pipeOf("kv_store_f16")
		run := func(perm []uint32) (k, v []uint16) {
			kc, vc := d.NewBufferBytes(poolRows*kvDim*2), d.NewBufferBytes(poolRows*kvDim*2)
			e := cq.Begin()
			e.Dispatch(pipe, kvpM*kvDim, 128, NewBufferU16s(d, qkv), kc, vc, NewBufferUint32s(d, pos), NewBufferU32(d, kvDim),
				NewBufferU32(d, qStride), NewBufferU32(d, kOff), NewBufferU32(d, vOff), NewBufferUint32s(d, perm))
			e.End()
			if err := e.Err(); err != nil {
				t.Fatal(err)
			}
			for p := kvpStart; p < kvpStart+kvpM; p++ { // the sequence's rows, read through its table
				r := row(perm, p)
				k = append(k, kc.U16s()[r*kvDim:(r+1)*kvDim]...)
				v = append(v, vc.U16s()[r*kvDim:(r+1)*kvDim]...)
			}
			return k, v
		}
		k1, v1 := run(kvpInOrder)
		k2, v2 := run(kvpScrambled)
		if u16Differ(k1, k2)+u16Differ(v1, v2) != 0 {
			t.Error("the scrambled pool, read through its table, holds other rows than the in-order one")
		}
		for m := range kvpM {
			for i := range kvDim {
				if k2[m*kvDim+i] != qkv[m*qStride+kOff+i] || v2[m*kvDim+i] != qkv[m*qStride+vOff+i] {
					t.Fatalf("row %d (position %d) is not at its table's row", m, kvpStart+m)
				}
			}
		}
	})
}
