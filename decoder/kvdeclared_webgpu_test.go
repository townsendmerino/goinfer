package decoder

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// webgpuKVRateOracle is WebGPU's K/V bytes a position as it was written from the architecture alone, before the price read the rows
// each layer declares (MM5's WebGPU half). It is what kvBytesPerPositionAllLayers must keep equaling.
func webgpuKVRateOracle(m *Model, f16, i8 bool) int64 {
	perElem := 4.0
	if f16 && !i8 {
		perElem = 2
	}
	bv := m.BackendView()
	arch := m.w.arch
	_, nLayers, _, _, _, _, _ := bv.Dims()
	var perPos float64
	for l := range nLayers {
		if arch.gemma4KVSrcAt(l) != l {
			continue
		}
		kvDim := arch.kvDimAt(l)
		if arch.mla != nil {
			perPos += 4 * float64(kvDim)
			continue
		}
		if i8 && kvDim > 0 {
			perPos += 2 * (float64(kvDim) + 4*float64(bv.Layer(l).KVHeads()))
			continue
		}
		perPos += 2 * float64(kvDim) * perElem
	}
	return int64(perPos)
}

// webgpuKVBuffersOracle is layer i's cache buffers as webgpuLayerCaches listed them from the architecture alone: K and V (then, at
// int8, their scale buffers) on an attention layer that owns its cache, the one f32 latent on an MLA layer, nothing else.
func webgpuKVBuffersOracle(m *Model, i, ctx int, kvF16, kvI8 bool) []int64 {
	arch := m.w.arch
	mixer, _ := m.BackendView().Layer(i).Ops()
	perElem := 4.0
	switch {
	case kvI8:
		perElem = 1
	case kvF16:
		perElem = 2
	}
	switch mixer {
	case MixerAttention:
		if arch.gemma4KVSrcAt(i) != i {
			return nil
		}
		kvDim := arch.kvDimAt(i)
		if kvDim == 0 || ctx <= 0 {
			return nil
		}
		c := int64(float64(kvDim) * perElem * float64(ctx))
		if kvI8 {
			sc := 4 * int64(m.BackendView().Layer(i).KVHeads()) * int64(ctx)
			return []int64{c, c, sc, sc}
		}
		return []int64{c, c}
	case MixerMLA:
		if ctx <= 0 {
			return nil
		}
		if d := arch.kvDimAt(i); d > 0 {
			return []int64{int64(ctx) * int64(d) * 4}
		}
	}
	return nil
}

// TestWebGPUKV_isTheDeclaredRows is G-MP2w: over every census fixture that loads, three contexts and each KV precision, WebGPU's K/V bytes a position and each
// attention or MLA layer's buffer list, read from the rows the layer declares, equal what the architecture-only arithmetic they
// replaced gave, to the byte.
func TestWebGPUKV_isTheDeclaredRows(t *testing.T) {
	var rates, lists int
	for _, fx := range hostKVFixtures(t) {
		m := fx.m
		for _, kv := range []string{"", "f16", "i8"} {
			f16, i8 := kv == "f16", kv == "i8"
			if got, want := m.kvBytesPerPositionAllLayers(f16, i8), webgpuKVRateOracle(m, f16, i8); got != want {
				t.Errorf("%s kv=%q: bytes a position from the declared rows %d, from the architecture %d", fx.name, kv, got, want)
			}
			rates++
			_, nLayers, _, _, _, _, _ := m.BackendView().Dims()
			for _, ctx := range []int{1, 256, 4096} {
				for l := range nLayers {
					mixer, _ := m.BackendView().Layer(l).Ops()
					if mixer != MixerAttention && mixer != MixerMLA {
						continue
					}
					got, _ := m.webgpuLayerCaches(l, ctx, f16, i8)
					if want := webgpuKVBuffersOracle(m, l, ctx, f16, i8); !slices.Equal(got, want) {
						t.Errorf("%s layer %d ctx=%d kv=%q: buffers from the declared rows %v, from the architecture %v", fx.name, l, ctx, kv, got, want)
					}
					lists++
				}
			}
		}
	}
	if rates == 0 || lists == 0 {
		t.Fatal("VOID: no fixture was checked")
	}
	t.Logf("%d rates and %d layer lists checked", rates, lists)
}

// TestWebGPUKV_handTable is G-MP3w: WebGPU's K/V bytes a position on three shapes, written as arithmetic on the config's numbers (not
// read from the declaration or the oracle): llama-tiny (dense GQA), deepseek-tiny (MLA's f32 latent at any precision) and
// qwen35-tiny (DeltaNet layers hold none). The KV-shared tail of gemma4-moe-kv-tiny is held by the sweep above and by
// TestResidentKV_webgpuSharedLayersOwnNone, not by a hand table: its per-layer widths differ.
func TestWebGPUKV_handTable(t *testing.T) {
	for _, c := range []struct {
		dir  string
		want func(m *Model, f16, i8 bool) int64
	}{
		{"../testdata/llama-tiny", func(m *Model, f16, i8 bool) int64 {
			cfg := m.Config()
			kvDim := int64(cfg.NumKVHeads * cfg.HeadDim)
			row := 4 * kvDim
			switch {
			case i8:
				row = kvDim + 4*int64(cfg.NumKVHeads)
			case f16:
				row = 2 * kvDim
			}
			return int64(cfg.NumLayers) * 2 * row
		}},
		{"../testdata/deepseek-tiny", func(m *Model, f16, i8 bool) int64 {
			// the latent row (kv_lora_rank + the rope key) is f32 at every precision, one row a layer
			_, kvLoRA, _, qkRope, _, _, _, _, ok := m.BackendView().MLAParams()
			if !ok {
				t.Fatal("deepseek-tiny has no MLA parameters")
			}
			return int64(m.Config().NumLayers) * int64(kvLoRA+qkRope) * 4
		}},
		{"selftestdata/qwen35-tiny", func(m *Model, f16, i8 bool) int64 {
			// only the attention layers hold a position-indexed cache; the DeltaNet layers hold a state, not rows
			cfg := m.Config()
			kvDim := int64(cfg.NumKVHeads * cfg.HeadDim)
			row := 4 * kvDim
			switch {
			case i8:
				row = kvDim + 4*int64(cfg.NumKVHeads)
			case f16:
				row = 2 * kvDim
			}
			var attn int64
			for l := range cfg.NumLayers {
				if mixer, _ := m.BackendView().Layer(l).Ops(); mixer == MixerAttention {
					attn++
				}
			}
			if attn == 0 || attn == int64(cfg.NumLayers) {
				t.Fatalf("qwen35-tiny has %d attention layers of %d: the fixture no longer mixes DeltaNet and attention", attn, cfg.NumLayers)
			}
			return attn * 2 * row
		}},
	} {
		m, err := Load(c.dir, Options{Backend: "cpu", Quant: "f32"})
		if err != nil {
			t.Fatalf("%s: %v", c.dir, err)
		}
		for _, kv := range []string{"", "f16", "i8"} {
			f16, i8 := kv == "f16", kv == "i8"
			if c.dir == "../testdata/deepseek-tiny" {
				f16, i8 = false, false // the f16 and int8 caches are refused on an MLA MoE family; the rate asked of the plan is the latent's
			}
			if got, want := m.kvBytesPerPositionAllLayers(f16, i8), c.want(m, f16, i8); got != want {
				t.Errorf("%s kv=%q: bytes a position %d, hand-computed %d", c.dir, kv, got, want)
			}
		}
		m.Close()
	}
}

// TestWebGPUKVBuffersOf_plantedShortRowGoesRed is the plant for G-MP2w: a K row declared one element short moves a layer's buffers off
// the oracle, so the sweep above would go red on it.
func TestWebGPUKVBuffersOf_plantedShortRowGoesRed(t *testing.T) {
	planted := 0
	for _, fx := range hostKVFixtures(t) {
		m := fx.m
		_, nLayers, _, _, _, _, _ := m.BackendView().Dims()
		for l := range nLayers {
			oracle := webgpuKVBuffersOracle(m, l, 256, false, false)
			if len(oracle) == 0 {
				continue
			}
			short := append([]program.Buf(nil), m.LayerBuffers(l)...)
			for i := range short {
				if short[i].Unit == program.PerPosition {
					short[i].Elems--
					break
				}
			}
			if slices.Equal(webgpuKVBuffersOf(m.LayerBuffers(l), 256, false, false), oracle) && slices.Equal(webgpuKVBuffersOf(short, 256, false, false), oracle) {
				t.Errorf("%s layer %d: a row declared one element short left the buffers at %v: the plant did not go red", fx.name, l, oracle)
			}
			planted++
			break
		}
	}
	if planted == 0 {
		t.Fatal("VOID: no fixture holds a K/V row to plant on")
	}
}
