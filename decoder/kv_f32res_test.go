package decoder

import (
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
)

// TestKVQuant_reservesNoF32 is F32R-G1 and RR-G1 (docs/tasks/task-memory-campaign-2026-10.md, MEM12-f32res and MEM12-ringres
// gates): an f16 or int8 CPU cache built with a capacity hint holds no f32 K/V capacity on any layer, and a ring layer holds
// none in any form; a global layer keeps its reservation in its own form; and the bytes the hint costs are the global layers'
// reservation alone (within 1%), computed from the capacity each layer should hold, not read back from the cache.
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
			rings := 0
			for l := range c.numLayers {
				want := hint * c.kvDim
				if c.rings[l] != nil {
					rings++
					want = 0
				}
				f32k, f32v := cap(c.keys[l]), cap(c.vals[l])
				switch arm.q {
				case kvqF32:
					if f32k != want || f32v != want {
						t.Errorf("%s %s layer %d (ring %v): f32 capacity %d/%d, want %d", dir, arm.name, l, c.rings[l] != nil, f32k, f32v, want)
					}
					quantBytes += 4 * 2 * uint64(want)
					continue
				case kvqF16:
					if cap(c.keysH[l]) != want || cap(c.valsH[l]) != want {
						t.Errorf("%s f16 layer %d (ring %v): f16 capacity %d/%d, want %d", dir, l, c.rings[l] != nil, cap(c.keysH[l]), cap(c.valsH[l]), want)
					}
					quantBytes += 2 * 2 * uint64(want)
				case kvI8:
					if cap(c.keysQ[l]) != want || cap(c.valsQ[l]) != want {
						t.Errorf("%s int8 layer %d (ring %v): int8 capacity %d/%d, want %d", dir, l, c.rings[l] != nil, cap(c.keysQ[l]), cap(c.valsQ[l]), want)
					}
					wantSc := want / c.headDim // one scale per head per position
					if cap(c.keyScale[l]) != wantSc || cap(c.valScale[l]) != wantSc {
						t.Errorf("%s int8 layer %d (ring %v): scale capacity %d/%d, want %d", dir, l, c.rings[l] != nil, cap(c.keyScale[l]), cap(c.valScale[l]), wantSc)
					}
					quantBytes += 2*uint64(want) + 4*2*uint64(wantSc)
				}
				if f32k != 0 || f32v != 0 {
					t.Errorf("%s %s layer %d: f32 capacity %d/%d beside the %s arrays; want 0", dir, arm.name, l, f32k, f32v, arm.name)
				}
			}
			// RR-G1 needs rings to test: gemma2-tiny alternates local and global layers, llama-tiny has no window.
			if wantRings := dir == "gemma2-tiny"; wantRings != (rings > 0 && rings < c.numLayers) || (!wantRings && rings != 0) {
				t.Errorf("%s %s: %d of %d layers ring", dir, arm.name, rings, c.numLayers)
			}
			extra := int64(grown) - int64(empty)
			t.Logf("%s %s: NewCache(%d) allocates %d bytes over NewCache(0); the global layers' reservation %d (%d of %d layers ring)",
				dir, arm.name, hint, extra, quantBytes, rings, c.numLayers)
			if d := extra - int64(quantBytes); d > int64(quantBytes)/100 || d < -int64(quantBytes)/100 {
				t.Errorf("%s %s: the hint costs %d bytes, the global layers' reservation is %d (%+d)", dir, arm.name, extra, quantBytes, d)
			}
		}
	}
}

// TestKVQuant_setQuantHonoursSkip: setQuant gives no quantized reservation to a layer NewKVCache's skip predicate names (a
// recurrent mixer layer, as hasNoAttentionKVAt reports, or a ring layer), in every quantized form, and the full hint to the
// rest. No hybrid fixture admits quantized KV today (kvF16OK and kvInt8OK are false for qwen35-tiny, nemotron-tiny,
// lfm2-tiny and nemotron3nano-tiny), so NewCache never reaches this for a mixer layer; the test pins setQuant's own rule for
// the day one does, by building the cache the way NewCache does.
func TestKVQuant_setQuantHonoursSkip(t *testing.T) {
	const layers, nKV, headDim, hint = 4, 2, 8, 64
	skip := func(l int) bool { return l%2 == 1 } // layers 1 and 3 hold no attention KV
	for _, q := range []kvQuant{kvqF16, kvI8} {
		c := NewKVCache(layers, nKV, headDim, 0, 0, skip)
		c.setQuant(q, hint)
		for l := range layers {
			want := hint * nKV * headDim
			if skip(l) {
				want = 0
			}
			var k, v, ks, vs, wantSc int
			if q == kvqF16 {
				k, v = cap(c.keysH[l]), cap(c.valsH[l])
			} else {
				k, v, ks, vs, wantSc = cap(c.keysQ[l]), cap(c.valsQ[l]), cap(c.keyScale[l]), cap(c.valScale[l]), want/headDim
			}
			if k != want || v != want || ks != wantSc || vs != wantSc {
				t.Errorf("quant %v layer %d (skipped %v): capacity %d/%d, scales %d/%d; want %d, scales %d", q, l, skip(l), k, v, ks, vs, want, wantSc)
			}
		}
	}
}
