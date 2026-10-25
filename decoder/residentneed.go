package decoder

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
const WeightsMemFraction = 0.70

// metalKVPad is the position granularity Metal rounds each KV buffer up to (metal/model.go: attention_prefill_fused
// reads whole 8-row tiles).
const metalKVPad = 8

// ResidentKVBytes is the KV cache a resident build on backend allocates for ctx positions. On "metal"
// it is exact to the allocation at the precision ResolveKV delivers there (f16; f32; int8 with per-head f32 scales), the full
// padded context on every attention layer, nothing for a layer whose mixer is not
// attention (ResidentLayerOps: DeltaNet, Mamba, or none). On "cuda" it is exact to the resident's K/V buffers (cuda/resident.go's kvBytesForCap:
// f32, or f16 where the KV policy delivers it, the full context on every attention layer, one latent buffer on an MLA layer, none on a
// layer with no attention KV) — before the driver's per-buffer 2 MiB rounding, which the CUDA resident's
// own fit check does not add either. On any other backend it is Plan's long-standing per-position formula
// at the precision requested (kvF16 / kvI8), multiplied by ctx.
func (m *Model) ResidentKVBytes(backend string, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	if backend == "cuda" {
		return m.cudaKVBytes(ctx, kvF16)
	}
	if backend != "metal" {
		return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
	}
	padded := int64((ctx + metalKVPad - 1) / metalKVPad * metalKVPad)
	delivered := m.KVDeliveredFor("metal")
	i8 := delivered == kvPrecI8
	elem := int64(2) // f16
	if delivered == kvPrecF32 {
		elem = 4
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	var total int64
	for l := range nLayers {
		mixer, _ := m.ResidentLayerOps(l)
		if mixer == MixerMLA { // the latent rows, f32 (metal/mla.go)
			_, kvLoRA, _, qkRope, _, _, _, _, _ := m.MLAResidentParams()
			total += padded * int64(kvLoRA+qkRope) * 4
			continue
		}
		if mixer != MixerAttention {
			continue // no KV cache on a recurrent or mixer-less layer (metal/model.go leaves r.kc[l]/r.vc[l] zero)
		}
		if m.KVSrcAtResident(l) != l {
			continue // a Gemma 4 E-model KV-shared layer aliases its source's cache (metal/model.go)
		}
		nKV := int64(m.KVHeadsAtResident(l))
		kvDim := nKV * int64(m.HeadDimAtResident(l))
		if i8 {
			total += 2 * padded * kvDim   // K and V payload, 1 byte/elem
			total += 2 * padded * nKV * 4 // K and V per-head f32 scales
			continue
		}
		total += 2 * padded * kvDim * elem // K and V, f16 or f32
	}
	return total
}

// ResidentNeedBytes is what a resident build on backend asks the device (or, on Metal, unified memory)
// to hold with `slots` expert slots per layer (0 = every expert resident) and a ctx-position KV cache:
// the weights, on Metal the host copy of them (ResidentHostCopyBytes — 0 for weights aliased from a
// .giw mapping), and the KV cache. Plan's NeedBytes and Metal's resident guard are this number.
func (m *Model) ResidentNeedBytes(backend string, slots, ctx int, kvF16, kvI8 bool) int64 {
	need := m.ResidentWeightBytesPaged(slots) - m.kquantSavings(backend) + m.ResidentKVBytes(backend, ctx, kvF16, kvI8)
	if backend == "metal" {
		need += m.ResidentHostCopyBytes(slots)
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
// slot's blocks: ceil(ctx/KVBlockRows) blocks of K and V (f32, or f16) on a full-attention layer and of one latent row on an MLA layer
// (the V head is reconstructed from the latent), and on a windowed layer the blocks of window+slack positions plus one; its two
// tables, 4 B a block each. Then the pools' null blocks, one block of every layer that holds K/V, once per load. Nothing on a layer that
// holds no attention K/V (DeltaNet and the other recurrent mixers). kvF16 prices an f16 cache (half the bytes), but only for an
// architecture the CUDA backend's KV policy covers (kvpolicy.go): any other asks for f16 and gets f32 on the device, so it is priced as
// f32. int8 is not a CUDA cache. Each further slot costs CUDAKVSlotBytes.
func (m *Model) cudaKVBytes(ctx int, kvF16 bool) int64 {
	slot, null := m.cudaKVParts(ctx, kvF16)
	return slot + null
}

// CUDAKVSlotBytes is what one more resident KV slot costs a cuda load at ctx positions: cudaKVBytes less the null blocks.
func (m *Model) CUDAKVSlotBytes(ctx int, kvF16 bool) int64 {
	slot, _ := m.cudaKVParts(ctx, kvF16)
	return slot
}

func (m *Model) cudaKVParts(ctx int, kvF16 bool) (slot, null int64) {
	a := m.w.arch
	elem := int64(4) // f32; an f16 cache holds half
	if kvF16 && kvF16Decline("cuda", a, Options{Backend: "cuda"}) == "" {
		elem = 2
	}
	blocks := int64((ctx + KVBlockRows - 1) / KVBlockRows)
	_, nLayers, _, _, _, _, _ := m.Dims()
	var perBlock, windowed int64
	for l := range nLayers {
		if m.KVSrcAtResident(l) != l {
			continue // a Gemma 4 E-model KV-shared layer aliases its source's cache (cuda/backend.go)
		}
		kvDim := int64(a.kvDimAt(l)) // 0 for a no-attention-KV layer; the latent width on MLA
		if kvDim == 0 {
			continue
		}
		if a.mla != nil {
			perBlock += KVBlockRows * kvDim * elem // one latent row per position
			null += KVBlockRows * kvDim * elem
			continue
		}
		null += 2 * KVBlockRows * kvDim * elem
		if held := m.SlidingWindowResident() + WindowedKVSlack; m.WindowedKVLayer(l) && ctx > held {
			// A windowed layer maps window+slack positions however long the context, plus a block for a range that starts mid-block
			// (cuda/kvpool.go's kvWindowBlocks); at a context no longer than that, the windowed KV does not engage.
			windowed += 2 * kvDim * elem * KVBlockRows * int64((held+KVBlockRows-1)/KVBlockRows+1)
			continue
		}
		perBlock += 2 * KVBlockRows * kvDim * elem // K and V
	}
	return perBlock*blocks + windowed + 2*4*blocks, null
}
