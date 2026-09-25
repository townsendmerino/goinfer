package decoder

// One memory-accounting path per backend (docs/tasks/task-memory-accounting-2026-09.md).
//
// Before this, Plan("metal") — what `fit` reports — and Metal's own pre-build guard priced the same
// load differently, measured on an M1 Pro (2026-09-25, qwen2.5-coder-1.5b at ctx 4096):
//
//   - a direct .gguf: the guard 4.33 GB, Plan 2.34 GB. Plan had no term for Metal's second copy of
//     the dense weights (unified memory holds the host WeightMat AND the re-packed device buffer —
//     ResidentHostCopyBytes), 2.1 GB here, so `fit` could say "resident" for a load the guard refuses;
//   - a v14 .giw (weights aliased, host copy 0): the guard 1.40 GB, Plan 1.52 GB. Plan priced Metal's
//     KV at f32; Metal only allocates f16 (the f32 KV kernels are compiled out, metal/model.go's
//     r.kvF32), so Plan's KV was twice the real one.
//
// ResidentNeedBytes is now the single definition both use. The KV half is ResidentKVBytes, which
// prices a backend's KV the way that backend ALLOCATES it — the layout differs per backend, so "one
// formula" has to be one function with a per-backend layout, not one flat formula (Metal allocates
// the full context for every attention layer, sliding-window ones included, padded to 8 positions;
// kvBytesForCtx's sliding-window cap models the CPU ring buffer and would under-count Metal).
//
// CUDA got its own layout on 2026-09-25 (docs/measurements/memory-accounting-cuda-2026-09-25.md), after
// measuring real CUDA residents against Plan's per-position formula: CUDA allocates f32 K/V whatever
// KV precision was asked for (it reads no KV-precision option), so a requested f16 / i8 halved or
// quartered Plan's figure below what CUDA builds; and an MLA layer holds ONE latent buffer, not K and
// V, so Plan doubled a DeepSeek-class model's KV. Dense, sliding-window (full context, no ring buffer),
// per-layer-geometry and DeltaNet-hybrid models already agreed. Other backends (webgpu, cpu) keep Plan's
// existing per-position formula unchanged.

// WeightsMemFraction is the share of a memory figure the weights (and the rest of a resident build)
// may occupy before a load is refused — the load-time fit guard applies it to available host memory,
// Metal's resident guard to physical RAM. One constant, so the two cannot drift.
const WeightsMemFraction = 0.70

// metalKVPad is the position granularity Metal rounds each KV buffer up to (metal/model.go, C-01:
// attention_prefill_fused reads whole 8-row tiles).
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
		kvDim := int64(a.kvDimAt(l)) // 0 for a no-attention-KV layer; the latent width on MLA
		if a.mla != nil {
			perPos += kvDim * 4 // one latent row per position
			continue
		}
		perPos += 2 * kvDim * 4 // K and V, f32
	}
	return perPos * int64(ctx)
}
