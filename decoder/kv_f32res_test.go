package decoder

import (
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
)

// TestKVQuant_reservesNoF32 is F32R-G1 (docs/tasks/task-memory-campaign-2026-10.md, MEM12-f32res gates): an f16 or int8 CPU
// cache built with a capacity hint holds no f32 K/V capacity on any layer, its quantized arrays keep their reservation,
// and the bytes the hint costs are the quantized reservation alone (within 1%). An f32 cache keeps its f32 reservation.
func TestKVQuant_reservesNoF32(t *testing.T) {
	const hint = 512
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	allocated := func(fn func()) uint64 {
		var a, b runtime.MemStats
		runtime.ReadMemStats(&a)
		fn()
		runtime.ReadMemStats(&b)
		return b.TotalAlloc - a.TotalAlloc
	}
	for _, dir := range []string{"llama-tiny", "gemma2-tiny"} {
		base := f1Load(t, filepath.Join("..", "testdata", dir))
		a := base.w.arch
		for _, arm := range []struct {
			name string
			q    kvQuant
			ok   bool
		}{{"f32", kvqF32, true}, {"f16", kvqF16, a.kvF16OK()}, {"int8", kvI8, a.kvInt8OK()}} {
			if !arm.ok {
				continue
			}
			m := *base
			m.kvH, m.kvI8 = arm.q == kvqF16, arm.q == kvI8
			var c *KVCache
			grown := allocated(func() { c = m.NewCache(hint) })
			empty := allocated(func() { m.NewCache(0) })
			var quantBytes uint64
			for l := range c.numLayers {
				want := hint * c.kvDim
				f32k, f32v := cap(c.keys[l]), cap(c.vals[l])
				switch arm.q {
				case kvqF32:
					if f32k != want || f32v != want {
						t.Errorf("%s %s layer %d: f32 capacity %d/%d, want %d", dir, arm.name, l, f32k, f32v, want)
					}
					continue
				case kvqF16:
					if cap(c.keysH[l]) != want || cap(c.valsH[l]) != want {
						t.Errorf("%s f16 layer %d: f16 capacity %d/%d, want %d", dir, l, cap(c.keysH[l]), cap(c.valsH[l]), want)
					}
					quantBytes += 2 * uint64(cap(c.keysH[l])+cap(c.valsH[l]))
				case kvI8:
					if cap(c.keysQ[l]) != want || cap(c.valsQ[l]) != want {
						t.Errorf("%s int8 layer %d: int8 capacity %d/%d, want %d", dir, l, cap(c.keysQ[l]), cap(c.valsQ[l]), want)
					}
					quantBytes += uint64(cap(c.keysQ[l])+cap(c.valsQ[l])) + 4*uint64(cap(c.keyScale[l])+cap(c.valScale[l]))
				}
				if f32k != 0 || f32v != 0 {
					t.Errorf("%s %s layer %d: f32 capacity %d/%d beside the %s arrays; want 0", dir, arm.name, l, f32k, f32v, arm.name)
				}
			}
			extra := int64(grown) - int64(empty)
			t.Logf("%s %s: NewCache(%d) allocates %d bytes over NewCache(0); quantized reservation %d", dir, arm.name, hint, extra, quantBytes)
			if arm.q != kvqF32 {
				if d := extra - int64(quantBytes); d > int64(quantBytes)/100 || d < -int64(quantBytes)/100 {
					t.Errorf("%s %s: the hint costs %d bytes, the quantized reservation is %d (%+d)", dir, arm.name, extra, quantBytes, d)
				}
			}
		}
	}
}
