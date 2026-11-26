package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// kvBytesForCtx is the K/V a CPU cache (NewCache) holds for ctx positions of arch, at the precision NewCache takes for the request
// (int8 or f16 only where the family accepts it, else f32). An append-forever layer holds ctx rows. A ring layer (ringLayerAt) holds
// its window's W rows from its first write, and at f32 twice that once a read wraps, which any context past W reaches. An int8 row is
// one byte an element plus one f32 scale a KV head; an MLA layer holds one f32 latent row a position; a layer that reads another's
// cache, or has no attention, holds none. Both host-RAM guards price the CPU's KV with it (estimateKVBytes), and Model.cpuKVBytes
// is it, so the guard and the plan cannot price one cache two ways. Do not replace it with a flat NumLayers*NumKVHeads*headDim
// formula: that overprices hybrid, sliding-window and MLA models, which cache far less.
func kvBytesForCtx(a *darch.Architecture, ctx int, f16, i8 bool) int64 {
	if a == nil || ctx <= 0 || a.NumLayers <= 0 {
		return 0
	}
	quant := kvqF32
	if i8 && a.KVInt8OK() {
		quant = kvI8
	} else if f16 && a.KVF16OK() {
		quant = kvqF16
	}
	return kvBytesForRows(a, ctx, quant, layerBuffersOf(a))
}

// layerBuffersOf is layerBuffers over a, as the per-layer source kvBytesForRows takes.
func layerBuffersOf(a *darch.Architecture) func(l int) []program.Buf {
	return func(l int) []program.Buf { return layerBuffers(a, l) }
}

// kvBytesForRows prices a CPU cache from the per-position rows each layer declares (bufsOf) at the precision quant: a K, V or latent
// row of b.Elems elements. A window is a ring, which holds its window's rows from the first write and at f32 twice that once a read
// wraps (ctx past the window). An MLA latent is declared f32 and stays f32 at any KV precision; an int8 K or V row is one byte an
// element plus one f32 scale a KV head. A layer that declares no row (it reads another's KV, or keeps none) adds nothing.
func kvBytesForRows(a *darch.Architecture, ctx int, quant kvQuant, bufsOf func(l int) []program.Buf) int64 {
	var total int64
	for l := range a.NumLayers {
		if a.HasNoAttentionKVAt(l) { // an architecture that marks a layer recurrent holds no row for it, whatever its operation list says
			continue
		}
		for _, b := range bufsOf(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			rows := int64(ctx)
			if b.Window > 0 && a.RingLayerAt(l) { // a window is a ring only where the CPU keeps one (Gemma 4's local layers hold the whole context)
				rows = int64(b.Window)
				if quant == kvqF32 && ctx > b.Window {
					rows *= 2
				}
			}
			row := 4 * int64(b.Elems)
			switch {
			case b.Prec != program.PrecKV:
			case quant == kvI8:
				row = int64(b.Elems) + 4*int64(b.Heads)
			case quant == kvqF16:
				row = 2*int64(b.Elems) + 2*int64(kvRowPadPriced(a, b.Elems, 2))
			default:
				row += 4 * int64(kvRowPadPriced(a, b.Elems, 4))
			}
			total += rows * row
		}
	}
	return total
}

// MoEConfig is a mixture-of-experts layer's routing configuration (program.MoEConfig).
type MoEConfig = program.MoEConfig

// ActKind is the FFN's activation (program.ActKind).
type ActKind = program.ActKind

// kvRowPadPriced is the row pad, in elements, on a row of elems elements of elemBytes bytes where the plan prices it: kvRowPad's
// answer for this architecture, and only for its uniform KV width.
func kvRowPadPriced(a *darch.Architecture, elems, elemBytes int) int {
	if elems != a.NumKVHeads*a.HeadDim {
		return 0
	}
	return a.KVRowPad(elemBytes)
}
