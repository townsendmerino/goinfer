package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// One memory-accounting path per backend (docs/tasks/task-memory-accounting-2026-09.md).
//
// ResidentNeedBytes is the single definition that Plan("metal") (what `fit` reports) and Metal's own pre-build guard
// both use, so they cannot price the same load differently. Its KV half is ResidentKVBytes, which prices a backend's KV
// the way that backend allocates it. The layout differs per backend, so "one formula" has to be one function with a
// per-backend layout, not one flat formula:
//
//   - Metal allocates the precision ResolveKV delivers there (f16, f32 or int8 with per-head scales; an MLA layer's latent
//     f32) and the full context for every attention layer, sliding-window ones included, padded to 8 positions; kvBytesForCtx's sliding-window cap
//     models the CPU ring buffer and would under-count Metal. Metal also holds a second copy of the dense weights in
//     unified memory (the host WeightMat and the re-packed device buffer, ResidentHostCopyBytes), which Plan must count.
//   - CUDA allocates f32 K/V, or f16 where kv16_cuda.go clears the model (the default for a cleared family, or -kv f16 for one the kernels
//     cover); an MLA layer holds one latent buffer, at the same precision, rather than K and V (docs/measurements/memory-accounting-cuda-2026-09-25.md).
//   - Other backends (webgpu, cpu) keep Plan's per-position formula.

// WeightsMemFraction is the share of a memory figure the weights (and the rest of a resident build)
// may occupy before a load is refused — the load-time fit guard applies it to available host memory,
// Metal's resident guard to physical RAM. One constant, so the two cannot drift.
const WeightsMemFraction = mem.WeightsFraction

// metalKVPad is the position granularity Metal's KV takes: its pools hold 64-row blocks (metal/model.go).
const metalKVPad = KVBlockRows

// ResidentKVBytes is the KV cache a resident build on backend allocates for ctx positions. On "metal"
// it is exact to the allocation at the precision ResolveKV delivers there (f16; f32; int8 with per-head f32 scales), the full
// padded context on every attention layer, nothing for a layer whose mixer is not
// attention (ResidentLayerOps: DeltaNet, Mamba, or none). On "cuda" it is exact to the resident's K/V buffers (cuda/resident.go's kvBytesForCap:
// f32, f16 or int8 with per-head f32 scales where the KV policy delivers it, the full context on every attention layer, one latent buffer on an MLA layer, none on a
// layer with no attention KV) — before the driver's per-buffer 2 MiB rounding, which the CUDA resident's
// own fit check does not add either. On "cpu" it is exact to what NewCache holds (cpuKVBytes). On any other backend it is Plan's
// long-standing per-position formula at the precision requested (kvF16 / kvI8), multiplied by ctx.
func (m *Model) ResidentKVBytes(backend string, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	if backend == "cuda" {
		return m.cudaKVBytes(ctx, kvF16, kvI8)
	}
	if backend == "cpu" {
		return m.cpuKVBytes(ctx, kvF16, kvI8)
	}
	if backend != "metal" {
		return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
	}
	slot, null := m.metalKVParts(ctx)
	return slot + null
}

// cpuKVBytes is the K/V a CPU cache (NewCache) holds for ctx positions, at the precision NewCache takes for the request (int8 or f16
// only where the family accepts it, else f32). An append-forever layer holds ctx rows. A ring layer (ringLayerAt) holds its
// window's W rows from its first write, and at f32 twice that once a read wraps, which any context past W reaches. An int8 row is
// one byte an element plus one f32 scale a KV head; an MLA layer holds one f32 latent row a position; a layer that reads another's
// cache, or has no attention, holds none.
func (m *Model) cpuKVBytes(ctx int, f16, i8 bool) int64 {
	a := m.w.arch
	if a == nil {
		return m.kvBytesPerPositionAllLayers(f16, i8) * int64(ctx)
	}
	quant := kvqF32
	if i8 && a.kvInt8OK() {
		quant = kvI8
	} else if f16 && a.kvF16OK() {
		quant = kvqF16
	}
	var total int64
	for l := range a.NumLayers {
		if a.hasNoAttentionKVAt(l) || a.gemma4KVSrcAt(l) != l {
			continue
		}
		width := int64(a.kvDimAt(l))
		if a.mla != nil {
			total += 4 * width * int64(ctx)
			continue
		}
		rows := int64(ctx)
		if a.ringLayerAt(l) {
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

// MetalKVSlotBytes is what one more resident KV slot costs a metal load at ctx positions: ResidentKVBytes less the pools' null blocks.
func (m *Model) MetalKVSlotBytes(ctx int) int64 {
	slot, _ := m.metalKVParts(ctx)
	return slot
}

// MetalKVBytes is the KV a metal load of slots resident slots holds at ctx positions: the pools' null blocks once, every slot's blocks,
// and every slot's block table at one conversation's cap (MetalKVTableRaiseBytes).
func (m *Model) MetalKVBytes(slots, ctx int) int64 {
	return m.ResidentKVBytes("metal", ctx, false, false) + int64(slots-1)*m.MetalKVSlotBytes(ctx) + m.MetalKVTableRaiseBytes(slots, ctx)
}

// MetalKVCap is one conversation's cap on a metal load of slots KV slots at a planned per-slot context of share positions: the pool's
// positions (every slot's share in whole blocks), within the model's window and MetalCtxCeiling, so one conversation may borrow from
// the others. It is share for one slot, a context the caller or the fit guard pinned, a load that prices KV per position beside the pool
// (a drafter's), and an int8 cache, whose prefill dequantizes the whole context into an f16 scratch the fit does not price.
func (m *Model) MetalKVCap(slots, share int) int {
	if slots < 2 || m.ResidentContextRequest() > 0 || m.ExtraResidentKVPerPosition() != 0 || m.KVDeliveredFor("metal") == kvPrecI8 {
		return share
	}
	c := min(slots*((share+KVBlockRows-1)/KVBlockRows)*KVBlockRows, MetalCtxCeiling)
	if w := m.Config().MaxPositions; w > 0 {
		c = min(c, w)
	}
	return max(c, share)
}

// MetalKVTableRaiseBytes is what slots slots' block tables hold past the per-slot price at ctx when one conversation's cap is raised
// (MetalKVCap): each table addresses the cap, 4 B a block. 0 for a model with no KV.
func (m *Model) MetalKVTableRaiseBytes(slots, ctx int) int64 {
	if ctx <= 0 || m.MetalKVSlotBytes(ctx) == 0 {
		return 0
	}
	blocks := func(n int) int64 { return int64((n + KVBlockRows - 1) / KVBlockRows) }
	return int64(slots) * (blocks(m.MetalKVCap(slots, ctx)) - blocks(ctx)) * 4
}

// metalKVParts is one slot's KV on Metal, at the precision ResolveKV delivers, and its pools' null blocks, once per load: per 64-row
// block, every row its layers declare (LayerBuffers) at Metal's size for it (MetalKVRowBytes), and the slot's table, 4 B a block.
func (m *Model) metalKVParts(ctx int) (slot, null int64) {
	blocks := int64((ctx + KVBlockRows - 1) / KVBlockRows)
	delivered := m.KVDeliveredFor("metal")
	var perBlock int64
	for l := range m.w.arch.NumLayers {
		for _, b := range m.LayerBuffers(l) {
			if b.Unit == program.PerPosition {
				perBlock += KVBlockRows * MetalKVRowBytes(b, delivered)
			}
		}
	}
	if perBlock == 0 {
		return 0, 0
	}
	return blocks*perBlock + blocks*4, perBlock
}

// MetalKVRowBytes is one position's row of b in Metal's KV pools when the cache is delivered at precision delivered (ResolveKV's
// "f16", "f32" or "i8"): an f32 buffer at 4 bytes an element, else int8 codes and one f32 scale per head, f32, or f16.
func MetalKVRowBytes(b program.Buf, delivered string) int64 {
	switch {
	case b.Prec == program.PrecF32 || delivered == kvPrecF32:
		return int64(b.Elems) * 4
	case delivered == kvPrecI8:
		return int64(b.Elems) + int64(b.Heads)*4
	}
	return int64(b.Elems) * 2
}

// ResidentNeedBytes is what a resident build on backend asks the device (or, on Metal, unified memory)
// to hold with `slots` expert slots per layer (0 = every expert resident) and a ctx-position KV cache:
// the weights, on Metal the host copy of them (ResidentHostCopyBytes — 0 for weights aliased from a
// .giw mapping), the buffers its build derives (metalDerivedDenseBytes) and decode's scratch (MetalDecodeScratchBytes), the KV
// cache, and one slot's recurrent state. Plan's NeedBytes and
// Metal's resident guard are this number.
func (m *Model) ResidentNeedBytes(backend string, slots, ctx int, kvF16, kvI8 bool) int64 {
	need := m.ResidentWeightBytesPaged(slots) - m.kquantSavings(backend) - m.mambaProjSavings(backend) + m.ResidentKVBytes(backend, ctx, kvF16, kvI8) + m.stateBytes(backend)
	switch backend {
	case "metal":
		need += m.ResidentHostCopyBytes(slots) + m.MetalDecodeScratchBytes(slots) + m.metalDerivedDenseBytes()
	case "cpu":
		dense, experts := m.cpuStructBytes()
		need += dense + experts
	}
	return need
}

// residentHostCopyFor is ResidentNeedBytes' host-copy term alone, for Plan's piecewise accounting.
func (m *Model) residentHostCopyFor(backend string, slots int) int64 {
	if backend != "metal" {
		return 0
	}
	return m.ResidentHostCopyBytes(slots)
}

// cudaKVBytes is ResidentKVBytes("cuda", ctx, …): what a one-slot cuda resident allocates for its paged K/V (cuda/kvpool.go). One
// slot's blocks: ceil(ctx/KVBlockRows) blocks of K and V on a full-attention layer and of one latent row on an MLA layer
// (the V head is reconstructed from the latent), and on a windowed layer the blocks of window+slack positions plus one; its two
// tables, 4 B a block each. Then the pools' null blocks, one block of every layer that holds K/V, once per load. Nothing on a layer that
// holds no attention K/V (DeltaNet and the other recurrent mixers). A row is priced at the precision the CUDA backend's KV policy
// delivers for the request (cudaKVRowBytes): kvI8 or kvF16 asked of an architecture the policy does not cover is f32 on the device. Each
// further slot costs CUDAKVSlotBytes.
func (m *Model) cudaKVBytes(ctx int, kvF16, kvI8 bool) int64 {
	slot, null := m.cudaKVParts(ctx, kvF16, kvI8)
	return slot + null
}

// CUDAKVSlotBytes is what one more resident KV slot costs a cuda load at ctx positions: cudaKVBytes less the null blocks.
func (m *Model) CUDAKVSlotBytes(ctx int, kvF16, kvI8 bool) int64 {
	slot, _ := m.cudaKVParts(ctx, kvF16, kvI8)
	return slot
}

// cudaKVPrec is the precision a CUDA resident's KV pools would hold for this architecture when kvF16 or kvI8 is asked: int8 where the CUDA
// int8 kernels cover the architecture, else f16 where the f16 kernels do, else f32. It reads the architecture alone, never the options this
// model was loaded with: ResidentKVBytes answers what a precision would cost (fit and Plan ask it of loads that hold another), so a load's
// slot count or drafter must not change the price of the cache asked about.
func (m *Model) cudaKVPrec(kvF16, kvI8 bool) string {
	a := m.w.arch
	bare := Options{Backend: "cuda"}
	if p, ok := kvPolicies["cuda"]; kvI8 && ok && p.DeclineI8 != nil && p.DeclineI8(a.kvFacts(), bare) == "" {
		return kvPrecI8
	}
	if kvF16 && kvF16Decline("cuda", a, bare) == "" {
		return kvPrecF16
	}
	return kvPrecF32
}

// cudaKVRowBytes is one position's row of b in a CUDA pool at prec: a row at the KV precision in int8 codes and one f32 scale per head
// (the scale pools, laid out like the cache), else 2 B an element at f16 and 4 at f32. A row declared f32 whatever the KV precision (MLA's
// latent) follows an f16 cache on CUDA, whose MLA kernels have f16 twins, and has no int8 form. The cuda package's kvRowBytes sizes the
// same row from the layer its build checks against the declaration (kvCheckDeclared).
func cudaKVRowBytes(b program.Buf, prec string) int64 {
	switch {
	case prec == kvPrecI8 && b.Prec == program.PrecKV:
		return int64(b.Elems) + 4*int64(b.Heads)
	case prec == kvPrecF16:
		return 2 * int64(b.Elems)
	}
	return 4 * int64(b.Elems)
}

// cudaKVParts is one slot's KV on CUDA at the precision its policy delivers (cudaKVPrec), and its pools' null blocks, once per load:
// every per-position row its layers declare (LayerBuffers) at CUDA's size for it (cudaKVRowBytes), per 64-row block. A row with a
// window, when the windowed cache is on and the context is past the window plus WindowedKVSlack, takes the blocks of window+slack
// positions plus one, for a range that starts mid-block (cuda/kvpool.go's kvWindowBlocks); at a shorter context the windowed cache does
// not engage. Then the slot's two tables, 4 B a block each.
func (m *Model) cudaKVParts(ctx int, kvF16, kvI8 bool) (slot, null int64) {
	prec := m.cudaKVPrec(kvF16, kvI8)
	blocks := int64((ctx + KVBlockRows - 1) / KVBlockRows)
	var perBlock, windowed int64
	for l := range m.w.arch.NumLayers {
		for _, b := range m.LayerBuffers(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			block := KVBlockRows * cudaKVRowBytes(b, prec)
			null += block
			if held := b.Window + WindowedKVSlack; b.Window > 0 && m.windowedKV && ctx > held {
				windowed += block * int64((held+KVBlockRows-1)/KVBlockRows+1)
				continue
			}
			perBlock += block
		}
	}
	return perBlock*blocks + windowed + 2*4*blocks, null
}
