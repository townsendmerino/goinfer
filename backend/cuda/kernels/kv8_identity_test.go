//go:build cuda

package kernels

import (
	"context"
	"errors"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
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
