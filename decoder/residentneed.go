package decoder

// One memory-accounting path per backend (docs/tasks/task-memory-accounting-2026-09.md).
//
// ResidentNeedBytes is the single definition that Plan("metal") (what `fit` reports) and Metal's own pre-build guard
// both use, so they cannot price the same load differently. Its KV half is ResidentKVBytes, which prices a backend's KV
// the way that backend allocates it. The layout differs per backend, so "one formula" has to be one function with a
// per-backend layout, not one flat formula:
//
//   - Metal allocates f16 only (the f32 KV kernels are compiled out, metal/model.go's r.kvF32) and the full context for
//     every attention layer, sliding-window ones included, padded to 8 positions; kvBytesForCtx's sliding-window cap
//     models the CPU ring buffer and would under-count Metal. Metal also holds a second copy of the dense weights in
//     unified memory (the host WeightMat and the re-packed device buffer, ResidentHostCopyBytes), which Plan must count.
//   - CUDA allocates f32 K/V whatever KV precision was requested (it reads no KV-precision option), and an MLA layer
//     holds one latent buffer rather than K and V (docs/measurements/memory-accounting-cuda-2026-09-25.md).
//   - Other backends (webgpu, cpu) keep Plan's per-position formula.

// WeightsMemFraction is the share of a memory figure the weights (and the rest of a resident build)
// may occupy before a load is refused — the load-time fit guard applies it to available host memory,
// Metal's resident guard to physical RAM. One constant, so the two cannot drift.
const WeightsMemFraction = 0.70

// metalKVPad is the position granularity Metal rounds each KV buffer up to (metal/model.go: attention_prefill_fused
// reads whole 8-row tiles).
const metalKVPad = 8

// ResidentKVBytes is the KV cache a resident build on backend allocates for ctx positions. On "metal"
// it is exact to the allocation: f16 (int8 with per-head f32 scales when the model was loaded with an
// int8 KV cache), the full padded context on every attention layer, nothing for a Gated-DeltaNet
// linear layer. On "cuda" it is exact to the resident's K/V buffers (cuda/resident.go's kvBytesForCap:
// f32 always, the full context on every attention layer, one latent buffer on an MLA layer, none on a
// layer with no attention KV) — before the driver's per-buffer 2 MiB rounding, which the CUDA resident's
// own fit check does not add either. On any other backend it is Plan's long-standing per-position formula
// at the precision requested (kvF16 / kvI8), multiplied by ctx.
func (m *Model) ResidentKVBytes(backend string, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	if backend == "cuda" {
		return m.cudaKVBytes(ctx)
	}
	if backend != "metal" {
		return m.kvBytesPerPositionAllLayers(kvF16, kvI8) * int64(ctx)
	}
	padded := int64((ctx + metalKVPad - 1) / metalKVPad * metalKVPad)
	i8 := m.KVCacheI8()
	_, nLayers, _, _, _, _, _ := m.Dims()
	_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
	var total int64
	for l := range nLayers {
		if dnetOK && m.Qwen35LinearLayer(l) {
			continue // no KV cache on a linear-attention layer (metal/model.go leaves r.kc[l]/r.vc[l] zero)
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
		total += 2 * padded * kvDim * 2 // K and V, f16
	}
	return total
}

// ResidentNeedBytes is what a resident build on backend asks the device (or, on Metal, unified memory)
// to hold with `slots` expert slots per layer (0 = every expert resident) and a ctx-position KV cache:
// the weights, on Metal the host copy of them (ResidentHostCopyBytes — 0 for weights aliased from a
// .giw mapping), and the KV cache. Plan's NeedBytes and Metal's resident guard are this number.
func (m *Model) ResidentNeedBytes(backend string, slots, ctx int, kvF16, kvI8 bool) int64 {
	need := m.ResidentWeightBytesPaged(slots) + m.ResidentKVBytes(backend, ctx, kvF16, kvI8)
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

// cudaKVBytes is ResidentKVBytes("cuda", ctx, …): what cuda/backend.go allocates for the K/V caches —
// r.af(ctx*kvDim) (f32) for K and for V on every attention layer, a single latent buffer of
// ctx*latDim on an MLA layer (the V head is reconstructed from the latent), nothing on a layer that
// holds no attention K/V (DeltaNet and the other recurrent mixers). The requested KV precision is
// ignored because CUDA ignores it.
func (m *Model) cudaKVBytes(ctx int) int64 {
	a := m.w.arch
	_, nLayers, _, _, _, _, _ := m.Dims()
	var perPos int64
	for l := range nLayers {
		if m.KVSrcAtResident(l) != l {
			continue // a Gemma 4 E-model KV-shared layer aliases its source's cache (cuda/backend.go)
		}
		kvDim := int64(a.kvDimAt(l)) // 0 for a no-attention-KV layer; the latent width on MLA
		if a.mla != nil {
			perPos += kvDim * 4 // one latent row per position
			continue
		}
		perPos += 2 * kvDim * 4 // K and V, f32
	}
	return perPos * int64(ctx)
}
