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
// own fit check does not add either. On any other backend it is Plan's long-standing per-position formula
// at the precision requested (kvF16 / kvI8), multiplied by ctx.
func (m *Model) ResidentKVBytes(backend string, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	if backend == "cuda" {
		return m.cudaKVBytes(ctx, kvF16, kvI8)
	}
	if backend != "metal" {
		return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
	}
	slot, null := m.metalKVParts(ctx)
	return slot + null
}

// MetalKVSlotBytes is what one more resident KV slot costs a metal load at ctx positions: ResidentKVBytes less the pools' null blocks.
func (m *Model) MetalKVSlotBytes(ctx int) int64 {
	slot, _ := m.metalKVParts(ctx)
	return slot
}

// MetalKVBytes is the KV a metal load of slots resident slots holds at ctx positions: the pools' null blocks once, every slot's blocks.
func (m *Model) MetalKVBytes(slots, ctx int) int64 {
	return m.ResidentKVBytes("metal", ctx, false, false) + int64(slots-1)*m.MetalKVSlotBytes(ctx)
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
// .giw mapping), the KV cache, and one slot's recurrent state. Plan's NeedBytes and Metal's resident guard are this number.
func (m *Model) ResidentNeedBytes(backend string, slots, ctx int, kvF16, kvI8 bool) int64 {
	need := m.ResidentWeightBytesPaged(slots) - m.kquantSavings(backend) - m.mambaProjSavings(backend) + m.ResidentKVBytes(backend, ctx, kvF16, kvI8) + m.stateBytes(backend)
	switch backend {
	case "metal":
		need += m.ResidentHostCopyBytes(slots)
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

// cudaKVPrec is the precision a CUDA resident's KV pools hold when kvF16 or kvI8 is asked of this model: int8 where ResolveKV delivers it
// under the load's other options, f16 where the CUDA policy covers the architecture, else f32.
func (m *Model) cudaKVPrec(kvF16, kvI8 bool) string {
	a := m.w.arch
	if kvI8 {
		o := m.kvOpts
		o.Backend, o.KVPrecision = "cuda", kvPrecI8
		if got, _ := ResolveKV("cuda", a, o); got == kvPrecI8 {
			return kvPrecI8
		}
		return kvPrecF32
	}
	if kvF16 && kvF16Decline("cuda", a, Options{Backend: "cuda"}) == "" {
		return kvPrecF16
	}
	return kvPrecF32
}

// cudaKVRowBytes is one position's K (or V) row of layer l in a CUDA pool at prec: kvDim elements of 4, 2 or 1 B, and at int8 one f32 scale per
// KV head (the scale pools, laid out like the cache).
func cudaKVRowBytes(a *Architecture, l int, prec string) int64 {
	kvDim := int64(a.kvDimAt(l))
	switch {
	case prec == kvPrecI8 && a.mla == nil:
		return kvDim + 4*int64(a.kvHeadsAt(l))
	case prec == kvPrecF16:
		return 2 * kvDim
	}
	return 4 * kvDim
}

func (m *Model) cudaKVParts(ctx int, kvF16, kvI8 bool) (slot, null int64) {
	a := m.w.arch
	prec := m.cudaKVPrec(kvF16, kvI8)
	blocks := int64((ctx + KVBlockRows - 1) / KVBlockRows)
	_, nLayers, _, _, _, _, _ := m.Dims()
	var perBlock, windowed int64
	for l := range nLayers {
		if m.KVSrcAtResident(l) != l {
			continue // a Gemma 4 E-model KV-shared layer aliases its source's cache (cuda/backend.go)
		}
		if a.kvDimAt(l) == 0 { // a no-attention-KV layer
			continue
		}
		row := cudaKVRowBytes(a, l, prec) // the latent width on MLA
		if a.mla != nil {
			perBlock += KVBlockRows * row // one latent row per position
			null += KVBlockRows * row
			continue
		}
		null += 2 * KVBlockRows * row
		if held := m.SlidingWindowResident() + WindowedKVSlack; m.WindowedKVLayer(l) && ctx > held {
			// A windowed layer maps window+slack positions however long the context, plus a block for a range that starts mid-block
			// (cuda/kvpool.go's kvWindowBlocks); at a context no longer than that, the windowed KV does not engage.
			windowed += 2 * row * KVBlockRows * int64((held+KVBlockRows-1)/KVBlockRows+1)
			continue
		}
		perBlock += 2 * KVBlockRows * row // K and V
	}
	return perBlock*blocks + windowed + 2*4*blocks, null
}
