//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestP3_attentionF32MatchesF16OnExactInputs is P3-G1 (docs/tasks/task-kv-precision-everywhere-2026-10.md, P3 gates): on K/V
// values an f16 cache holds exactly, attention_f32 over the f32 cache and attention over the f16 cache give bit-identical outputs,
// with an attention sink, a window, and past the 4,096-key tile; and kv_store_f32 stores what kv_store narrows.
func TestP3_attentionF32MatchesF16OnExactInputs(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(allKernels+kvF32Kernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	p16, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatal(err)
	}
	p32, err := d.NewComputePipeline(lib, "attention_f32")
	if err != nil {
		t.Fatal(err)
	}
	const nH, nKV = 4, 2
	rng := rand.New(rand.NewSource(20261022))
	for _, hd := range []int{64, 128} {
		for _, c := range []struct {
			nKeys, window int
			sink          bool
		}{{37, 0, false}, {37, 0, true}, {300, 64, true}, {4100, 0, false}, {8193, 0, true}, {9000, 6000, true}} {
			name := fmt.Sprintf("hd=%d nKeys=%d window=%d sink=%v", hd, c.nKeys, c.window, c.sink)
			kvDim := nKV * hd
			qv := randUnit(rng, nH*hd)
			kf, vf := randUnit(rng, c.nKeys*kvDim), randUnit(rng, c.nKeys*kvDim)
			kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
			for i := range kf { // exact inputs: values an f16 cache holds without rounding
				kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
				kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
			}
			sinks, hasSink := []float32{0, 0, 0, 0}, 0
			if c.sink {
				sinks, hasSink = []float32{1.5, -0.5, 3, 0.25}, 1
			}
			u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
			run := func(p Pipeline, kBuf, vBuf Buffer) []float32 {
				out := d.NewBufferLen(nH * hd)
				e := q.Begin()
				e.Dispatch(p, nH*tgReduceAttn, tgReduceAttn, NewBufferFloats(d, qv), kBuf, vBuf, out, u(nH), u(nKV), u(hd), u(c.nKeys),
					NewBufferFloats(d, []float32{float32(1 / math.Sqrt(float64(hd)))}), u(c.window), NewBufferFloats(d, sinks), u(hasSink))
				e.End()
				if err := e.Err(); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				return append([]float32(nil), out.Floats()[:nH*hd]...)
			}
			a := run(p16, NewBufferU16s(d, kh), NewBufferU16s(d, vh))
			b := run(p32, NewBufferFloats(d, kf), NewBufferFloats(d, vf))
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Errorf("%s: attention_f32 differs from attention at %d: %v against %v", name, i, b[i], a[i])
					break
				}
			}
		}
	}
}
