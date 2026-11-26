package decoder

import (
	"slices"
	"testing"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// kvBytesForCtxOracle is the CPU cache's K/V price as it was written from the architecture alone, before the price read the
// per-layer declarations (MM5's CPU half): it is what kvBytesForCtx must keep equaling, so a change to either side is seen.
func kvBytesForCtxOracle(a *darch.Architecture, ctx int, f16, i8 bool) int64 {
	if a == nil || ctx <= 0 || a.NumLayers <= 0 {
		return 0
	}
	quant := kvqF32
	if i8 && a.KVInt8OK() {
		quant = kvI8
	} else if f16 && a.KVF16OK() {
		quant = kvqF16
	}
	var total int64
	for l := range a.NumLayers {
		if a.HasNoAttentionKVAt(l) || a.Gemma4KVSrcAt(l) != l {
			continue
		}
		width := int64(a.KVDimAt(l))
		if a.MLA != nil {
			total += 4 * width * int64(ctx)
			continue
		}
		rows := int64(ctx)
		if a.RingLayerAt(l) {
			rows = int64(a.SlidingWindow)
			if quant == kvqF32 && ctx > a.SlidingWindow {
				rows *= 2
			}
		}
		row := 4 * width
		switch quant {
		case kvI8:
			row = width + 4*int64(a.NumKVHeads)
		case kvqF16:
			row = 2 * width
		}
		total += 2 * rows * row
	}
	return total
}

// TestKVBytesForCtx_isTheDeclaredRows is G-MP2c: over every census fixture that loads, each context around its window and each precision a caller can ask, the CPU's
// K/V price read from the rows each layer declares equals the architecture-only arithmetic it replaced, to the byte.
func TestKVBytesForCtx_isTheDeclaredRows(t *testing.T) {
	var checked int
	for _, fx := range hostKVFixtures(t) {
		a := fx.m.w.Arch
		for _, ctx := range hostKVContexts(a) {
			for _, kv := range []string{"", "f16", "i8"} {
				got, want := kvBytesForCtx(a, ctx, kv == "f16", kv == "i8"), kvBytesForCtxOracle(a, ctx, kv == "f16", kv == "i8")
				if got != want {
					t.Errorf("%s ctx=%d kv=%q: the price from the declared rows is %d, the architecture's is %d", fx.name, ctx, kv, got, want)
				}
				checked++
			}
		}
	}
	if checked == 0 {
		t.Fatal("VOID: no fixture was checked")
	}
	t.Logf("%d prices checked", checked)
}

// TestKVBytesForRows_plantedShortRowGoesRed is the plant for G-MP2c: one declared K row one element short moves the price off the
// oracle, so the sweep above would go red on it.
func TestKVBytesForRows_plantedShortRowGoesRed(t *testing.T) {
	planted := 0
	for _, fx := range hostKVFixtures(t) {
		a := fx.m.w.Arch
		short := func(l int) []program.Buf {
			bufs := slices.Clone(layerBuffers(a, l))
			for i := range bufs {
				if bufs[i].Unit == program.PerPosition && bufs[i].Elems > 0 {
					bufs[i].Elems--
					break
				}
			}
			return bufs
		}
		ctx := hostKVContexts(a)[1]
		if kvBytesForRows(a, ctx, kvqF32, layerBuffersOf(a)) != kvBytesForCtxOracle(a, ctx, false, false) {
			t.Errorf("%s: the unplanted rows already differ from the oracle", fx.name)
			continue
		}
		if oracle := kvBytesForCtxOracle(a, ctx, false, false); oracle > 0 {
			planted++
			if kvBytesForRows(a, ctx, kvqF32, short) == oracle {
				t.Errorf("%s: a K row declared one element short left the price at %d: the plant did not go red", fx.name, oracle)
			}
		}
	}
	if planted == 0 {
		t.Fatal("VOID: no fixture holds a K/V row to plant on")
	}
}
