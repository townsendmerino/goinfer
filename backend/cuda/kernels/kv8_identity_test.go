//go:build cuda

package kernels

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
)

// G1 of CUDA int8 KV (docs/tasks/task-kv-precision-everywhere-2026-10.md §4; docs/measurements/cuda-kv-i8-2026-11/scope.md): on a cache that holds int8 values times
// power-of-two scales, the i8 reader gives the f32 original's context BIT FOR BIT, because the product is exact and the kernel folds in the original's order. The f32
// reference reads the same values written out as floats. A reader that applied the wrong (position, head) scale, or the scale of a different row under a scrambled block
// table, cannot match.

// kv8Block is the pool's block size in rows (kvpage.cuh).
const kv8Block = 64

// kv8Case is a cache of nKeys positions, nKV heads of hd: int8 values, one power-of-two scale per (position, head) for K and for V, and the f32 values they stand for.
type kv8Case struct {
	nKeys, nKV, hd int
	kq, vq         []int8
	ksc, vsc       []float32
	kf, vf         []float32
}

func newKV8Case(nKeys, nKV, hd int, seed uint32) kv8Case {
	c := kv8Case{nKeys: nKeys, nKV: nKV, hd: hd}
	n := nKeys * nKV * hd
	c.kq, c.vq, c.kf, c.vf = make([]int8, n), make([]int8, n), make([]float32, n), make([]float32, n)
	c.ksc, c.vsc = make([]float32, nKeys*nKV), make([]float32, nKeys*nKV)
	s := seed
	next := func() uint32 { s = s*1664525 + 1013904223; return s >> 8 }
	for r := 0; r < nKeys*nKV; r++ {
		c.ksc[r] = float32(math.Ldexp(1, -9+int(next()%5))) // 2^-9 .. 2^-5: distinct per (position, head)
		c.vsc[r] = float32(math.Ldexp(1, -8+int(next()%5)))
		for d := 0; d < hd; d++ {
			i := r*hd + d
			c.kq[i], c.vq[i] = int8(int(next()%255)-127), int8(int(next()%255)-127)
			if r%37 == 0 && d == 3 {
				c.kq[i] = 127 // an outlier channel at full range
			}
			c.kf[i], c.vf[i] = float32(c.kq[i])*c.ksc[r], float32(c.vq[i])*c.vsc[r]
		}
	}
	return c
}

// scrambled returns a block table that is a fixed permutation of the first blocks, and the case's rows laid out under it: logical row k lives at physical row
// KV_ROW(table, k), in the cache and in the scale pools alike.
func (c kv8Case) scrambled() (table []int32, kq, vq []int8, ksc, vsc, kf, vf []float32) {
	blocks := (c.nKeys + kv8Block - 1) / kv8Block
	table = make([]int32, blocks)
	for b := range table {
		table[b] = int32((b*5 + 3) % (blocks + 4)) // distinct for the block counts used here; physical pool is blocks+4 blocks
	}
	phys := (blocks + 4) * kv8Block
	kq, vq = make([]int8, phys*c.nKV*c.hd), make([]int8, phys*c.nKV*c.hd)
	kf, vf = make([]float32, phys*c.nKV*c.hd), make([]float32, phys*c.nKV*c.hd)
	ksc, vsc = make([]float32, phys*c.nKV), make([]float32, phys*c.nKV)
	for k := 0; k < c.nKeys; k++ {
		pr := int(table[k/kv8Block])*kv8Block + k%kv8Block
		copy(kq[pr*c.nKV*c.hd:], c.kq[k*c.nKV*c.hd:(k+1)*c.nKV*c.hd])
		copy(vq[pr*c.nKV*c.hd:], c.vq[k*c.nKV*c.hd:(k+1)*c.nKV*c.hd])
		copy(kf[pr*c.nKV*c.hd:], c.kf[k*c.nKV*c.hd:(k+1)*c.nKV*c.hd])
		copy(vf[pr*c.nKV*c.hd:], c.vf[k*c.nKV*c.hd:(k+1)*c.nKV*c.hd])
		copy(ksc[pr*c.nKV:], c.ksc[k*c.nKV:(k+1)*c.nKV])
		copy(vsc[pr*c.nKV:], c.vsc[k*c.nKV:(k+1)*c.nKV])
	}
	return
}

// glueAttentionBothWays runs glue.cu's `attention` on the f32 module over the f32 values and on the i8 module over the int8 values and scales, under the same block table,
// and returns both contexts. plant, when set, swaps the K and V scale pools in the i8 launch: a planted defect the comparison must catch.
func glueAttentionBothWays(t *testing.T, cx *gc.Context, c kv8Case, nH, window int, plant bool) (f32Out, i8Out []float32) {
	t.Helper()
	table, kq, vq, ksc, vsc, kf, vf := c.scrambled()
	q := make([]float32, nH*c.hd)
	for i := range q {
		q[i] = float32(int(i*7%13)-6) / 16
	}
	bg := context.Background()
	if e := func() error {
		stream, e := cx.NewStream()
		if e != nil {
			return e
		}
		defer stream.Close()
		run := func(ptx []byte, i8 bool) ([]float32, error) {
			mod, e := cx.LoadModule(ptx)
			if e != nil {
				return nil, e
			}
			fn, e := mod.Function("attention")
			if e != nil {
				return nil, e
			}
			dq, e1 := allocBuffer[float32](cx, len(q))
			dt, e2 := allocBuffer[int32](cx, len(table))
			dc, e3 := allocBuffer[float32](cx, nH*c.hd)
			if e1 != nil || e2 != nil || e3 != nil {
				return nil, errors.Join(e1, e2, e3)
			}
			defer func() { dq.Close(); dt.Close(); dc.Close() }()
			if e := errors.Join(copyHtoD(bg, dq, q), copyHtoD(bg, dt, table)); e != nil {
				return nil, e
			}
			cfg := gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((c.nKeys + 128) * 4)}
			scale := float32(0.125)
			base := []gc.KernelArg{gc.Arg(dq)}
			var closers []func()
			defer func() {
				for _, f := range closers {
					f()
				}
			}()
			var tail []gc.KernelArg
			if i8 {
				dk, e1 := allocBuffer[int8](cx, len(kq))
				dv, e2 := allocBuffer[int8](cx, len(vq))
				dks, e3 := allocBuffer[float32](cx, len(ksc))
				dvs, e4 := allocBuffer[float32](cx, len(vsc))
				if e := errors.Join(e1, e2, e3, e4); e != nil {
					return nil, e
				}
				closers = append(closers, func() { dk.Close(); dv.Close(); dks.Close(); dvs.Close() })
				if e := errors.Join(copyHtoD(bg, dk, kq), copyHtoD(bg, dv, vq), copyHtoD(bg, dks, ksc), copyHtoD(bg, dvs, vsc)); e != nil {
					return nil, e
				}
				base = append(base, gc.Arg(dk), gc.Arg(dv))
				if plant {
					tail = []gc.KernelArg{gc.Arg(dvs), gc.Arg(dks)}
				} else {
					tail = []gc.KernelArg{gc.Arg(dks), gc.Arg(dvs)}
				}
			} else {
				dk, e1 := allocBuffer[float32](cx, len(kf))
				dv, e2 := allocBuffer[float32](cx, len(vf))
				if e := errors.Join(e1, e2); e != nil {
					return nil, e
				}
				closers = append(closers, func() { dk.Close(); dv.Close() })
				if e := errors.Join(copyHtoD(bg, dk, kf), copyHtoD(bg, dv, vf)); e != nil {
					return nil, e
				}
				base = append(base, gc.Arg(dk), gc.Arg(dv))
			}
			args := append(base, gc.Arg(dt), gc.ArgValue(int32(nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)), gc.ArgValue(int32(c.nKeys)),
				gc.ArgValue(scale), gc.ArgValue(int32(window)), gc.Arg(dc))
			args = append(args, tail...)
			if e := fn.LaunchOn(bg, stream, cfg, args...); e != nil {
				return nil, e
			}
			if e := stream.Synchronize(bg); e != nil {
				return nil, e
			}
			out := make([]float32, nH*c.hd)
			return out, gc.CopyDtoH(bg, out, dc)
		}
		var e1, e2 error
		f32Out, e1 = run(gluePTX, false)
		i8Out, e2 = run(glueKV8PTX, true)
		return errors.Join(e1, e2)
	}(); e != nil {
		t.Fatalf("launch: %v", e)
	}
	return f32Out, i8Out
}

func kv8CountDiff(a, b []float32) (diff, nonzero int) {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			diff++
		}
		if a[i] != 0 {
			nonzero++
		}
	}
	return
}

// TestKV8_exactInputIdentity_glueAttention is G1 for the decode attention in glue.cu: GQA at head dimensions 64 and 128, key counts around a block edge and a long one, a
// sliding window, and a scrambled block table.
func TestKV8_exactInputIdentity_glueAttention(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	for _, g := range []struct{ nH, nKV, hd int }{{12, 2, 64}, {8, 2, 128}, {4, 4, 64}} {
		for _, nKeys := range []int{1, 63, 64, 65, 300} {
			for _, window := range []int{0, 70} {
				c := newKV8Case(nKeys, g.nKV, g.hd, uint32(31+nKeys+g.hd))
				a, b := glueAttentionBothWays(t, cx, c, g.nH, window, false)
				diff, nz := kv8CountDiff(a, b)
				if nz == 0 {
					t.Errorf("nH=%d nKV=%d hd=%d nKeys=%d window=%d: the f32 context is all zero, the comparison is vacuous", g.nH, g.nKV, g.hd, nKeys, window)
				}
				if diff != 0 {
					t.Errorf("nH=%d nKV=%d hd=%d nKeys=%d window=%d: %d/%d context elements differ between the i8 reader and the f32 original on int8 x 2^e values", g.nH, g.nKV, g.hd, nKeys, window, diff, len(a))
				}
			}
		}
	}
}

// TestKV8_exactInputIdentity_plant: the comparison goes red when the K and V scale pools are swapped in the i8 launch, so the identity is not satisfied by any scale at all.
func TestKV8_exactInputIdentity_plant(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	c := newKV8Case(300, 2, 64, 77)
	a, b := glueAttentionBothWays(t, cx, c, 12, 0, true)
	if diff, _ := kv8CountDiff(a, b); diff == 0 {
		t.Error("with the K and V scale pools swapped the i8 reader still equals the original: the identity test does not see the scales")
	}
}

// kv8Context is the device's primary context, or a skip when there is no device.
func kv8Context(t *testing.T) *gc.Context {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	return cx
}

// attnBatchedBothWays runs prefill_batched.cu's attn_batched (M query rows over startPos+M keys) on the f32 module over the f32 values and on the i8 module over the int8
// values and scales, under the same scrambled block table.
func attnBatchedBothWays(t *testing.T, cx *gc.Context, c kv8Case, nH, startPos, M, window int, sinks bool) (f32Out, i8Out []float32) {
	t.Helper()
	table, kq, vq, ksc, vsc, kf, vf := c.scrambled()
	qDim := nH * c.hd
	q := make([]float32, M*qDim)
	for i := range q {
		q[i] = float32(int(i*7%13)-6) / 16
	}
	bg := context.Background()
	stream := mustStream(t, cx)
	defer stream.Close()
	var sinkVals []float32
	if sinks {
		sinkVals = make([]float32, nH)
		for h := range sinkVals {
			sinkVals[h] = float32(h%3) - 1
		}
	}
	run := func(ptx []byte, i8 bool) []float32 {
		mod, err := cx.LoadModule(ptx)
		if err != nil {
			t.Fatalf("LoadModule: %v", err)
		}
		fn, err := mod.Function("attn_batched")
		if err != nil {
			t.Fatalf("Function(attn_batched): %v", err)
		}
		dq := mustAlloc[float32](t, cx, len(q))
		dt := mustAlloc[int32](t, cx, len(table))
		dc := mustAlloc[float32](t, cx, M*qDim)
		for _, e := range []error{copyHtoD(bg, dq, q), copyHtoD(bg, dt, table)} {
			if e != nil {
				t.Fatal(e)
			}
		}
		defer func() { dq.Close(); dt.Close(); dc.Close() }()
		sinkArg := gc.ArgDevicePtr(0)
		if sinks {
			ds := mustAlloc[float32](t, cx, nH)
			defer ds.Close()
			if e := copyHtoD(bg, ds, sinkVals); e != nil {
				t.Fatal(e)
			}
			sinkArg = gc.Arg(ds)
		}
		args := []gc.KernelArg{gc.Arg(dq)}
		var tail []gc.KernelArg
		if i8 {
			dk, dv := mustAlloc[int8](t, cx, len(kq)), mustAlloc[int8](t, cx, len(vq))
			dks, dvs := mustAlloc[float32](t, cx, len(ksc)), mustAlloc[float32](t, cx, len(vsc))
			defer func() { dk.Close(); dv.Close(); dks.Close(); dvs.Close() }()
			for _, e := range []error{copyHtoD(bg, dk, kq), copyHtoD(bg, dv, vq), copyHtoD(bg, dks, ksc), copyHtoD(bg, dvs, vsc)} {
				if e != nil {
					t.Fatal(e)
				}
			}
			args = append(args, gc.Arg(dk), gc.Arg(dv))
			tail = []gc.KernelArg{gc.Arg(dks), gc.Arg(dvs)}
		} else {
			dk, dv := mustAlloc[float32](t, cx, len(kf)), mustAlloc[float32](t, cx, len(vf))
			defer func() { dk.Close(); dv.Close() }()
			for _, e := range []error{copyHtoD(bg, dk, kf), copyHtoD(bg, dv, vf)} {
				if e != nil {
					t.Fatal(e)
				}
			}
			args = append(args, gc.Arg(dk), gc.Arg(dv))
		}
		maxNWin := startPos + M
		if window > 0 && window < maxNWin {
			maxNWin = window
		}
		cfg := gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((maxNWin + 128) * 4)}
		args = append(args, gc.Arg(dt), gc.ArgValue(int32(nH)), gc.ArgValue(int32(c.nKV)), gc.ArgValue(int32(c.hd)), gc.ArgValue(int32(startPos)),
			gc.ArgValue(float32(0.125)), gc.ArgValue(int32(window)), gc.ArgValue(int32(M)), gc.Arg(dc), sinkArg)
		args = append(args, tail...)
		if e := fn.LaunchOn(bg, stream, cfg, args...); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatal(e)
		}
		out := make([]float32, M*qDim)
		if e := gc.CopyDtoH(bg, out, dc); e != nil {
			t.Fatal(e)
		}
		return out
	}
	return run(prefillBatchedPTX, false), run(prefillBatchedKV8PTX, true)
}

// TestKV8_exactInputIdentity_attnBatched is G1 for the batched-prefill attention: the same shapes, a fresh prompt and a warm cache, windows that reach back past
// the cache seam, with and without attention sinks, and a scrambled block table.
func TestKV8_exactInputIdentity_attnBatched(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	for _, g := range []struct{ nH, nKV, hd int }{{12, 2, 64}, {8, 2, 128}, {4, 4, 64}} {
		for _, tc := range []struct{ startPos, M, window int }{{0, 16, 0}, {7, 40, 0}, {0, 70, 33}, {65, 70, 33}, {3, 130, 0}} {
			for _, sinks := range []bool{false, true} {
				c := newKV8Case(tc.startPos+tc.M, g.nKV, g.hd, uint32(41+tc.startPos+tc.M+g.hd))
				a, b := attnBatchedBothWays(t, cx, c, g.nH, tc.startPos, tc.M, tc.window, sinks)
				diff, nz := kv8CountDiff(a, b)
				if nz == 0 {
					t.Errorf("%+v %+v sinks=%v: the f32 context is all zero, the comparison is vacuous", g, tc, sinks)
				}
				if diff != 0 {
					t.Errorf("%+v %+v sinks=%v: %d/%d context elements differ between the i8 reader and the f32 original", g, tc, sinks, diff, len(a))
				}
			}
		}
	}
}

// TestKV8_store_matchesTheCPUQuantizer holds kv8_store to the scheme every other backend's int8 cache uses (aikit linalg.QuantizeRowInt8): the same int8 bytes and the same
// f32 scale for every (token, head), over data with zero heads, an outlier, values on the rounding tie, and sign changes, stored under a scrambled block table at
// positions that cross a block edge. A store that rounded differently, scaled a head by another head's absmax, or addressed the wrong row or scale cannot match.
func TestKV8_store_matchesTheCPUQuantizer(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	bg := context.Background()
	stream := mustStream(t, cx)
	defer stream.Close()
	mod, err := cx.LoadModule(kv8StorePTX)
	if err != nil {
		t.Fatalf("LoadModule: %v", err)
	}
	fn, err := mod.Function("kv8_store")
	if err != nil {
		t.Fatalf("Function(kv8_store): %v", err)
	}
	for _, g := range []struct{ nKV, hd, startPos, M int }{{2, 64, 0, 5}, {4, 128, 60, 9}, {2, 80, 127, 3}, {1, 256, 1, 2}} {
		kvDim := g.nKV * g.hd
		k, v := make([]float32, g.M*kvDim), make([]float32, g.M*kvDim)
		s := uint32(g.hd*7 + g.startPos)
		next := func() float32 { s = s*1664525 + 1013904223; return float32(int32(s>>8)%2001-1000) / 137 }
		for i := range k {
			k[i], v[i] = next(), next()
		}
		for m := 0; m < g.M; m++ {
			for h := 0; h < g.nKV; h++ {
				o := m*kvDim + h*g.hd
				switch (m + h) % 4 {
				case 0: // a zero head
					clear(k[o : o+g.hd])
					clear(v[o : o+g.hd])
				case 1: // absmax 127 makes the scale 1 and 1/scale 1, so 2.5 and -3.5 are exact rounding ties: half away from zero gives 3 and -4, half to even 2 and -4
					k[o+1], k[o+5] = 127, 2.5
					v[o+2], v[o+5], v[o+6] = -127, -3.5, 0.5
				}
			}
		}
		pos := g.startPos + g.M
		blocks := (pos + kv8Block - 1) / kv8Block
		table := make([]int32, blocks)
		for b := range table {
			table[b] = int32((b*3 + 1) % (blocks + 3))
		}
		phys := (blocks + 3) * kv8Block
		dk, dv := mustAlloc[float32](t, cx, len(k)), mustAlloc[float32](t, cx, len(v))
		dkc, dvc := mustAlloc[int8](t, cx, phys*kvDim), mustAlloc[int8](t, cx, phys*kvDim)
		dks, dvs := mustAlloc[float32](t, cx, phys*g.nKV), mustAlloc[float32](t, cx, phys*g.nKV)
		dt := mustAlloc[int32](t, cx, len(table))
		for _, e := range []error{copyHtoD(bg, dk, k), copyHtoD(bg, dv, v), copyHtoD(bg, dt, table), copyHtoD(bg, dkc, make([]int8, phys*kvDim)), copyHtoD(bg, dvc, make([]int8, phys*kvDim)),
			copyHtoD(bg, dks, make([]float32, phys*g.nKV)), copyHtoD(bg, dvs, make([]float32, phys*g.nKV))} {
			if e != nil {
				t.Fatal(e)
			}
		}
		if e := fn.LaunchOn(bg, stream, gc.LaunchConfig{GridX: uint32(g.nKV), GridY: uint32(g.M), GridZ: 1, BlockX: 32, BlockY: 1, BlockZ: 1},
			gc.Arg(dk), gc.Arg(dv), gc.Arg(dkc), gc.Arg(dvc), gc.Arg(dks), gc.Arg(dvs), gc.Arg(dt),
			gc.ArgValue(int32(g.nKV)), gc.ArgValue(int32(g.hd)), gc.ArgValue(int32(g.startPos)), gc.ArgValue(int32(g.M))); e != nil {
			t.Fatalf("launch: %v", e)
		}
		if e := stream.Synchronize(bg); e != nil {
			t.Fatal(e)
		}
		gotKq, gotVq := make([]int8, phys*kvDim), make([]int8, phys*kvDim)
		gotKs, gotVs := make([]float32, phys*g.nKV), make([]float32, phys*g.nKV)
		for _, e := range []error{gc.CopyDtoH(bg, gotKq, dkc), gc.CopyDtoH(bg, gotVq, dvc), gc.CopyDtoH(bg, gotKs, dks), gc.CopyDtoH(bg, gotVs, dvs)} {
			if e != nil {
				t.Fatal(e)
			}
		}
		bad, checked := 0, 0
		for m := 0; m < g.M; m++ {
			p := g.startPos + m
			row := int(table[p/kv8Block])*kv8Block + p%kv8Block
			for h := 0; h < g.nKV; h++ {
				for which, src := range [][]float32{k, v} {
					wq := make([]int8, g.hd)
					ws := linalg.QuantizeRowInt8(src[m*kvDim+h*g.hd:m*kvDim+(h+1)*g.hd], wq)
					gq, gs := gotKq, gotKs
					if which == 1 {
						gq, gs = gotVq, gotVs
					}
					checked++
					if math.Float32bits(gs[row*g.nKV+h]) != math.Float32bits(ws) || !slices.Equal(gq[row*kvDim+h*g.hd:row*kvDim+(h+1)*g.hd], wq) {
						bad++
						if bad <= 3 {
							t.Errorf("%+v token %d head %d %c: scale %v want %v", g, m, h, "KV"[which], gs[row*g.nKV+h], ws)
						}
					}
				}
			}
		}
		if bad != 0 {
			t.Errorf("%+v: %d of %d (token, head, K/V) differ from the CPU quantizer", g, bad, checked)
		}
		// nothing is written outside the stored rows
		for r := 0; r < phys; r++ {
			stored := false
			for m := 0; m < g.M; m++ {
				p := g.startPos + m
				if r == int(table[p/kv8Block])*kv8Block+p%kv8Block {
					stored = true
				}
			}
			if stored {
				continue
			}
			for _, b := range gotKq[r*kvDim : (r+1)*kvDim] {
				if b != 0 {
					t.Fatalf("%+v: the store wrote row %d, which no token maps to", g, r)
				}
			}
		}
		for _, b := range []interface{ Close() error }{dk, dv, dkc, dvc, dks, dvs, dt} {
			_ = b.Close()
		}
	}
}

// kv8Args are the device buffers a both-ways launch shares: the query, the scrambled block table, the K and V caches (f32 values, or int8) and, for the i8 arm, the two
// scale pools as the trailing kernel arguments.
type kv8Args struct {
	q     *gc.Buffer[float32]
	table *gc.Buffer[int32]
	kc    gc.KernelArg
	vc    gc.KernelArg
	tail  []gc.KernelArg // the K and V scale pools on the i8 arm; nil on the f32 arm
}

// kv8Both runs launch once on the f32 module over the f32 values and once on the i8 module over the int8 values and scales, under the case's scrambled block table, and
// returns what each returns. launch gets the function, the shared buffers and the stream; it allocates and reads back its own outputs.
func kv8Both(t *testing.T, cx *gc.Context, c kv8Case, q []float32, f32PTX, i8PTX []byte, name string, launch func(fn *gc.Function, a kv8Args, st *gc.Stream) []float32) (f32Out, i8Out []float32) {
	t.Helper()
	table, kq, vq, ksc, vsc, kf, vf := c.scrambled()
	bg := context.Background()
	st := mustStream(t, cx)
	defer st.Close()
	run := func(ptx []byte, i8 bool) []float32 {
		mod, err := cx.LoadModule(ptx)
		if err != nil {
			t.Fatalf("LoadModule: %v", err)
		}
		fn, err := mod.Function(name)
		if err != nil {
			t.Fatalf("Function(%s): %v", name, err)
		}
		a := kv8Args{q: mustAlloc[float32](t, cx, len(q)), table: mustAlloc[int32](t, cx, len(table))}
		defer func() { a.q.Close(); a.table.Close() }()
		for _, e := range []error{copyHtoD(bg, a.q, q), copyHtoD(bg, a.table, table)} {
			if e != nil {
				t.Fatal(e)
			}
		}
		if i8 {
			dk, dv := mustAlloc[int8](t, cx, len(kq)), mustAlloc[int8](t, cx, len(vq))
			dks, dvs := mustAlloc[float32](t, cx, len(ksc)), mustAlloc[float32](t, cx, len(vsc))
			defer func() { dk.Close(); dv.Close(); dks.Close(); dvs.Close() }()
			for _, e := range []error{copyHtoD(bg, dk, kq), copyHtoD(bg, dv, vq), copyHtoD(bg, dks, ksc), copyHtoD(bg, dvs, vsc)} {
				if e != nil {
					t.Fatal(e)
				}
			}
			a.kc, a.vc, a.tail = gc.Arg(dk), gc.Arg(dv), []gc.KernelArg{gc.Arg(dks), gc.Arg(dvs)}
			return launch(fn, a, st)
		}
		dk, dv := mustAlloc[float32](t, cx, len(kf)), mustAlloc[float32](t, cx, len(vf))
		defer func() { dk.Close(); dv.Close() }()
		for _, e := range []error{copyHtoD(bg, dk, kf), copyHtoD(bg, dv, vf)} {
			if e != nil {
				t.Fatal(e)
			}
		}
		a.kc, a.vc = gc.Arg(dk), gc.Arg(dv)
		return launch(fn, a, st)
	}
	return run(f32PTX, false), run(i8PTX, true)
}

func kv8Query(n int) []float32 {
	q := make([]float32, n)
	for i := range q {
		q[i] = float32(int(i*7%13)-6) / 16
	}
	return q
}

// TestKV8_exactInputIdentity_flashDecode is G1 for decode_fa.cu's single-row lane (fa_partial_{64,128,256} then fa_combine): head dimensions 64, 128 and 256, GQA group
// sizes, key counts around block edges, a window, S key splits.
func TestKV8_exactInputIdentity_flashDecode(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	bg := context.Background()
	const split = 4
	for _, g := range []struct{ nH, nKV, hd int }{{8, 2, 64}, {12, 2, 128}, {4, 1, 256}, {6, 6, 64}} {
		for _, tc := range []struct{ nKeys, winStart int }{{1, 0}, {63, 0}, {64, 0}, {65, 0}, {300, 0}, {300, 140}} {
			c := newKV8Case(tc.nKeys, g.nKV, g.hd, uint32(51+tc.nKeys+g.hd))
			G := g.nH / g.nKV
			which := map[int]string{64: "fa_partial_64", 128: "fa_partial_128", 256: "fa_partial_256"}[g.hd]
			launch := func(ptxCombine []byte) func(fn *gc.Function, a kv8Args, st *gc.Stream) []float32 {
				return func(fn *gc.Function, a kv8Args, st *gc.Stream) []float32 {
					part := mustAlloc[float32](t, cx, g.nH*split*8*(g.hd+4)*2)
					ctx := mustAlloc[float32](t, cx, g.nH*g.hd)
					defer func() { part.Close(); ctx.Close() }()
					cfg := gc.LaunchConfig{GridX: uint32(g.nKV), GridY: split, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((G*g.hd + 8*64) * 4)}
					args := []gc.KernelArg{gc.Arg(a.q), a.kc, a.vc, gc.Arg(a.table), gc.ArgValue(int32(g.nH)), gc.ArgValue(int32(g.nKV)), gc.ArgValue(int32(tc.winStart)),
						gc.ArgValue(int32(tc.nKeys)), gc.ArgValue(float32(0.125)), gc.ArgValue(int32(split)), gc.Arg(part)}
					args = append(args, a.tail...)
					if e := fn.LaunchOn(bg, st, cfg, args...); e != nil {
						t.Fatalf("fa_partial: %v", e)
					}
					mod, err := cx.LoadModule(ptxCombine)
					if err != nil {
						t.Fatal(err)
					}
					comb, err := mod.Function("fa_combine")
					if err != nil {
						t.Fatal(err)
					}
					if e := comb.LaunchOn(bg, st, gc.LaunchConfig{GridX: uint32(g.nH), GridY: 1, GridZ: 1, BlockX: uint32(g.hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * split * 8 * 4)},
						gc.Arg(part), gc.ArgValue(int32(g.hd)), gc.ArgValue(int32(split*8)), gc.Arg(ctx)); e != nil {
						t.Fatalf("fa_combine: %v", e)
					}
					if e := st.Synchronize(bg); e != nil {
						t.Fatal(e)
					}
					out := make([]float32, g.nH*g.hd)
					if e := gc.CopyDtoH(bg, out, ctx); e != nil {
						t.Fatal(e)
					}
					return out
				}
			}(decodeFAPTX)
			a, b := kv8Both(t, cx, c, kv8Query(g.nH*g.hd), decodeFAPTX, decodeFAKV8PTX, which, launch)
			diff, nz := kv8CountDiff(a, b)
			if nz == 0 {
				t.Errorf("%+v %+v: the f32 context is all zero, the comparison is vacuous", g, tc)
			}
			if diff != 0 {
				t.Errorf("%+v %+v: %d/%d context elements differ between the i8 reader and the f32 original", g, tc, diff, len(a))
			}
		}
	}
}

// TestKV8_exactInputIdentity_flashDecodeRows is G1 for the multi-row verify lane (fa_partial_rows_<hd>_g<G> then fa_combine_rows): one run of consecutive rows, a window, and
// each head dimension at two group sizes.
func TestKV8_exactInputIdentity_flashDecodeRows(t *testing.T) {
	cx := kv8Context(t)
	defer cx.Close()
	bg := context.Background()
	const split = 4
	for _, g := range []struct{ nH, nKV, hd int }{{8, 2, 64}, {6, 2, 64}, {12, 2, 128}, {8, 2, 128}, {4, 1, 256}, {2, 1, 256}} {
		G := g.nH / g.nKV
		R := launchgeom.FaRowsPerCTA(g.hd, G)
		for _, tc := range []struct{ n0, m, window int }{{70, 5, 0}, {200, 7, 90}} {
			runs := launchgeom.FaRowRuns(tc.n0, tc.m, tc.window, split*launchgeom.FaWarps)
			total := tc.n0 + tc.m
			c := newKV8Case(total, g.nKV, g.hd, uint32(61+total+g.hd))
			name := fmt.Sprintf("fa_partial_rows_%d_g%d", g.hd, G)
			launch := func(fn *gc.Function, a kv8Args, st *gc.Stream) []float32 {
				part := mustAlloc[float32](t, cx, launchgeom.FaMaxRows*g.nH*split*8*(g.hd+4))
				ctx := mustAlloc[float32](t, cx, tc.m*g.nH*g.hd)
				defer func() { part.Close(); ctx.Close() }()
				mod, err := cx.LoadModule(decodeFAPTX)
				if err != nil {
					t.Fatal(err)
				}
				comb, err := mod.Function("fa_combine_rows")
				if err != nil {
					t.Fatal(err)
				}
				for _, run := range runs {
					grid := uint32((run.N + R - 1) / R)
					cfg := gc.LaunchConfig{GridX: uint32(g.nKV), GridY: split, GridZ: grid, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((R*G*g.hd + 8*R*64) * 4)}
					args := []gc.KernelArg{gc.Arg(a.q), a.kc, a.vc, gc.Arg(a.table), gc.ArgValue(int32(g.nH)), gc.ArgValue(int32(g.nKV)), gc.ArgValue(int32(run.WinStart)),
						gc.ArgValue(int32(run.NKeys0)), gc.ArgValue(int32(run.Per)), gc.ArgValue(float32(0.125)), gc.ArgValue(int32(split)),
						gc.ArgValue(int32(run.N)), gc.ArgValue(int32(run.Row0)), gc.Arg(part)}
					args = append(args, a.tail...)
					if e := fn.LaunchOn(bg, st, cfg, args...); e != nil {
						t.Fatalf("%s: %v", name, e)
					}
					if e := comb.LaunchOn(bg, st, gc.LaunchConfig{GridX: uint32(g.nH), GridY: uint32(run.N), GridZ: 1, BlockX: uint32(g.hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * split * 8 * 4)},
						gc.Arg(part), gc.ArgValue(int32(g.nH)), gc.ArgValue(int32(g.hd)), gc.ArgValue(int32(split*8)), gc.ArgValue(int32(run.Row0)), gc.Arg(ctx)); e != nil {
						t.Fatalf("fa_combine_rows: %v", e)
					}
					if e := st.Synchronize(bg); e != nil {
						t.Fatal(e)
					}
				}
				out := make([]float32, tc.m*g.nH*g.hd)
				if e := gc.CopyDtoH(bg, out, ctx); e != nil {
					t.Fatal(e)
				}
				return out
			}
			a, b := kv8Both(t, cx, c, kv8Query(tc.m*g.nH*g.hd), decodeFAPTX, decodeFAKV8PTX, name, launch)
			diff, nz := kv8CountDiff(a, b)
			if nz == 0 {
				t.Errorf("%+v %+v: the f32 context is all zero, the comparison is vacuous", g, tc)
			}
			if diff != 0 {
				t.Errorf("%+v %+v: %d/%d context elements differ between the i8 rows reader and the f32 original", g, tc, diff, len(a))
			}
		}
	}
}
