package decoder

import (
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// wmBytes is one weight matrix's in-memory footprint, read from the backing slices, not computed from Rows()*Cols(): a
// quantized matrix carries per-group scales beside its packed payload and int4 packs two values per byte, so a
// dimension-derived figure is wrong both ways. Asking the matrix what it holds cannot drift when a new kind is added.
func wmBytes(w *linalg.WeightMat) int64 {
	var n int64
	if q8, sc, _, ok := w.Int8(); ok {
		n += int64(len(q8)) + 4*int64(len(sc))
	}
	if q4, sc, _, ok := w.Int4F16(); ok {
		n += int64(len(q4)) + 2*int64(len(sc)) // int4 scales are binary16
	}
	if p4, sc, ok := w.Int4Row4F16(); ok {
		n += int64(len(p4)) + 2*int64(len(sc))
	}
	if f, ok := w.F32(); ok {
		n += 4 * int64(len(f))
	}
	if raw, ok := w.Q4K(); ok {
		n += int64(len(raw))
	}
	n += int64(w.SplitHalfBytes())
	return n
}

// mmapAliasedBytes is wmBytes' quantized-kind sum counting only bytes that alias m's .giw mmap region (MmapByteOffset): a
// zero-copy load, not a heap allocation. f32 and SplitHalfBytes are never mmap-aliased (f32 is copied at .giw read time;
// SplitHalfBytes is a derived repack). Unlike aikit's WeightMat.MappedSpan (the expert paging registry), this does not
// page-align: it answers "did this allocation happen at all", not "can MADV_DONTNEED release whole pages", and rounding
// would undercount a small mmap-backed tensor.
func mmapAliasedBytes(m *Model, w *linalg.WeightMat) int64 {
	if m == nil || len(m.mmap) == 0 {
		return 0
	}
	var n int64
	if q8, sc, _, ok := w.Int8(); ok && len(q8) > 0 {
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&q8[0])), len(q8))
		if _, aliased := m.MmapByteOffset(raw); aliased {
			n += int64(len(q8)) + 4*int64(len(sc))
		}
	}
	if q4, sc, _, ok := w.Int4F16(); ok && len(q4) > 0 {
		if _, aliased := m.MmapByteOffset(q4); aliased {
			n += int64(len(q4)) + 2*int64(len(sc)) // int4 scales are binary16
		}
	}
	if p4, sc, ok := w.Int4Row4F16(); ok && len(p4) > 0 {
		if _, aliased := m.MmapByteOffset(p4); aliased {
			n += int64(len(p4)) + 2*int64(len(sc))
		}
	}
	return n
}

// f32SliceBytes is wmBytes' counterpart for the families that keep a projection as a plain []float32 (MLA, Mamba-2, LFM2's
// short conv). These are real matrices, not norms/biases, so they belong in the same sum as wmBytes' callers build.
func f32SliceBytes(s []float32) int64 { return 4 * int64(len(s)) }

// ResidentWeightBytes is the total byte footprint of this model's weight matrices: what a resident backend must hold to
// run the whole model on-device, assuming every routed MoE expert is resident. See ResidentWeightBytesPaged for the
// synchronous-paging case.
//
// It exists so a backend can ask "will this model fit?" before allocating. Dims() does not expose the expert count, so a
// shape-derived estimate under-reports a sparse MoE by the factor that matters most.
//
// It sums the matrices, where the bytes are; elementwise norms/biases are [hidden]-sized and round to nothing. That makes
// this a lower bound, the safe direction for a guard: it can fail to refuse a marginal model but cannot refuse one that
// would have fit.
//
// This is a quantity we compute, deliberately, not the OS's account of free memory: Darwin's UBC reclaims under
// pressure, so "available" reports what survived rather than what can be asked for. Do not key a memory guard on RSS or
// free memory. The incident: docs/code-notes/decoder.md#Model.ResidentWeightBytes.
func (m *Model) ResidentWeightBytes() int64 { return m.ResidentWeightBytesPaged(0) }

// ResidentWeightBytesPaged is ResidentWeightBytes under Metal's synchronous MoE paging (GOINFER_METAL_MOE_SLOTS=N keeps
// only N of each layer's routed experts resident, staging the rest per token). slots<=0 means unpaged. SharedExpert is
// never paged (it is always active, not top-k routed) and counts in full either way, as does every dense matrix. The
// memory-fit guard must call this with the slot count the caller asked for, not the unpaged sum, or it declines a model
// that fits paged. A layer's experts are uniform in shape, so per-expert bytes is the per-layer sum divided by the expert
// count, exact across layers.
func (m *Model) ResidentWeightBytesPaged(slots int) int64 {
	dense, _, expertBytesAt, _ := m.residentWeightBytesSplit()
	return dense + expertBytesAt(slots)
}

// ResidentDenseWeightBytes is ResidentWeightBytesPaged's non-expert term alone: every matrix a routed-MoE cap cannot
// shrink (projections, the mixer/MLA/Mamba/shortConv fields, PLE, embeddings and head; everything except l.Experts and
// gemma4moe's fused experts). A guard that must check whether the fixed part of the model fits before an elastic expert
// cache sizes itself against live free memory (e.g. CUDA's capSlots) wants this, not ResidentWeightBytesPaged(0), which
// would decline a model whose experts are about to be sized down to fit.
func (m *Model) ResidentDenseWeightBytes() int64 {
	dense, _, _, _ := m.residentWeightBytesSplit()
	return dense
}

// ResidentDenseWeightBytesFor is ResidentDenseWeightBytes less the model-level tables backend keeps on the host rather
// than its device: the dense figure that backend's fit checks must price against device memory.
//
// CUDA uploads every per-layer matrix and the LM head (the embedding table when the embeddings are tied). It gathers
// token embeddings on the host and never uploads the learned position table or Gemma 4's per-layer embedding tables, so
// an untied model's token-embedding table must not be priced as device memory; pricing it makes the slot-aware context
// plan land too low. Every other backend gets ResidentDenseWeightBytes unchanged.
// Measurement: docs/code-notes/decoder.md#Model.ResidentDenseWeightBytesFor.
func (m *Model) ResidentDenseWeightBytesFor(backend string) int64 {
	return m.ResidentDenseWeightBytes() - m.residentHostSideBytes(backend)
}

// residentHostSideBytes is the dense-sum portion backend keeps host-side; see ResidentDenseWeightBytesFor.
func (m *Model) residentHostSideBytes(backend string) int64 {
	if backend != "cuda" || m == nil || m.w == nil {
		return 0
	}
	w := m.w
	n := wmBytes(&w.PosEmbed) + wmBytes(&w.PerLayerTokenEmbed) + wmBytes(&w.PerLayerModelProj)
	if w.LMHead.Rows() != 0 {
		n += wmBytes(&w.Embed) // untied: the table stays on the host (tied, it IS the uploaded LM head)
	}
	return n
}

// ResidentHostCopyBytes is the portion of ResidentWeightBytesPaged's footprint that a unified-memory backend (Metal:
// "device" memory is host RAM) holds in two places at once: the quantized host WeightMat the loader materializes, and a
// second, freshly re-packed device buffer built from it, with nothing released in between.
//
// Dense weights (every non-expert matrix, and an unpaged model's experts) double this way for a GGUF or safetensors
// load. Genuinely paged routed experts (0 < slots < nExperts) do not: they stream from the .giw file into their device
// slot, or via an mmap the OS can reclaim, so an unstaged expert leaves no committed host allocation to double.
//
// A .giw-loaded model's int8/int4 payloads are themselves aliased into the mmap (LoadSerializedWeights: big int8/int4
// arrays are zero-copy). That is reclaimable page cache, not a second committed host allocation, so it is subtracted
// here and must not count as a host copy. Counting it priced a .giw int4 dense model at about twice its real anonymous
// footprint; the history is in docs/code-notes/decoder.md#Model.ResidentHostCopyBytes.
func (m *Model) ResidentHostCopyBytes(slots int) int64 {
	dense, mmapDense, expertBytesAt, mmapExpertBytesAt := m.residentWeightBytesSplit()
	if slots > 0 {
		return dense - mmapDense // paged experts stream; only dense doubles, minus any mmap-aliased portion
	}
	// Unpaged: every expert is a materialized host copy too, minus any mmap-aliased portion of dense or expert bytes.
	return dense + expertBytesAt(0) - mmapDense - mmapExpertBytesAt(0)
}

// residentWeightBytesSplit is the one enumeration pass ResidentWeightBytesPaged and ResidentHostCopyBytes share, so the
// two cannot disagree about what "dense" means. It returns the dense (non-expert) sum, plus a closure capping the
// routed-expert sum at `slots` experts. mmapDense and mmapExpertBytesAt are the parallel sums of the portion that aliases
// this model's .giw mmap rather than a heap allocation, computed in the same pass for the same reason.
func (m *Model) residentWeightBytesSplit() (dense, mmapDense int64, expertBytesAt, mmapExpertBytesAt func(slots int) int64) {
	noop := func(int) int64 { return 0 }
	if m == nil || m.w == nil {
		return 0, 0, noop, noop
	}
	w := m.w
	count := func(mat *linalg.WeightMat) {
		dense += wmBytes(mat)
		mmapDense += mmapAliasedBytes(m, mat)
	}
	count(&w.Embed)
	count(&w.LMHead)
	count(&w.PosEmbed)
	// Gemma 4's model-level PLE tables; empty WeightMats (a no-op sum) on every other family.
	count(&w.PerLayerTokenEmbed)
	count(&w.PerLayerModelProj)
	var fullExpertBytes, mmapExpertBytes []int64 // one entry per (layer, expert-kind) — generic l.Experts and gemma4moe's fused set are both "kinds"
	var expertCounts []int
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, mat := range []*linalg.WeightMat{
			&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj,
			&l.GateProj, &l.UpProj, &l.DownProj,
			&l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj,
		} {
			count(mat)
		}
		// The routed experts are most of a sparse MoE: omitting them is the under-report this accessor exists to prevent.
		var expertBytes, mmapEBytes int64
		for j := range l.Experts {
			e := &l.Experts[j]
			expertBytes += wmBytes(&e.Gate) + wmBytes(&e.Up) + wmBytes(&e.Down)
			mmapEBytes += mmapAliasedBytes(m, &e.Gate) + mmapAliasedBytes(m, &e.Up) + mmapAliasedBytes(m, &e.Down)
		}
		fullExpertBytes = append(fullExpertBytes, expertBytes)
		mmapExpertBytes = append(mmapExpertBytes, mmapEBytes)
		expertCounts = append(expertCounts, len(l.Experts))
		count(&l.SharedExpert.Gate)
		count(&l.SharedExpert.Up)
		count(&l.SharedExpert.Down)

		// qwen3_5_moe's per-layer mixer (DeltaNet or gated-softmax attention): the three dominant projections are WeightMats; the
		// rest are f32 vectors small enough to fall under the norms/biases exemption. At most one of these is non-nil.
		if d := l.delta; d != nil {
			count(&d.inProjQKV)
			count(&d.inProjZ)
			count(&d.outProj)
		}
		if q := l.qattn; q != nil {
			count(&q.qProj)
			count(&q.kProj)
			count(&q.vProj)
			count(&q.oProj)
		}
		// MLA (DeepSeek, Kimi) projections are kept as f32 slices. f32 slices are never mmap-aliased (LoadSerializedWeights
		// copies float arrays), so there is no mmap counterpart.
		if mla := l.mla; mla != nil {
			dense += f32SliceBytes(mla.qAProj) + f32SliceBytes(mla.qBProj) + f32SliceBytes(mla.qProj) +
				f32SliceBytes(mla.kvAProj) + f32SliceBytes(mla.kvBProj) + f32SliceBytes(mla.oProj)
		}
		// Mamba-2 (Granite, Nemotron) is kept as f32. The recurrent state is per-sequence and never counted here.
		if mb := l.mamba; mb != nil {
			dense += f32SliceBytes(mb.inProj) + f32SliceBytes(mb.convW) + f32SliceBytes(mb.outProj)
		}
		// LFM2's gated short-convolution mixer, kept as f32.
		if sc := l.shortConv; sc != nil {
			dense += f32SliceBytes(sc.inProj) + f32SliceBytes(sc.convW) + f32SliceBytes(sc.outProj)
		}
		// Gemma 4's MoE sub-block. mlpGate/mlpUp/mlpDown alias l.GateProj/UpProj/DownProj (the gemma4Layer comment in
		// serialize.go) and are already counted above; only routerProj and the fused experts are new tensors. The fused experts
		// page the same way as l.Experts.
		if mo := l.gemma4moe; mo != nil {
			count(&mo.routerProj)
			var fusedBytes, mmapFusedBytes int64
			for e := range mo.expertsGateUp {
				fusedBytes += wmBytes(&mo.expertsGateUp[e]) + wmBytes(&mo.expertsDown[e])
				mmapFusedBytes += mmapAliasedBytes(m, &mo.expertsGateUp[e]) + mmapAliasedBytes(m, &mo.expertsDown[e])
			}
			fullExpertBytes = append(fullExpertBytes, fusedBytes)
			mmapExpertBytes = append(mmapExpertBytes, mmapFusedBytes)
			expertCounts = append(expertCounts, len(mo.expertsGateUp))
		}
	}
	// capAt caps a per-expert-kind full byte sum at `slots` experts when paging applies. It serves both expertBytesAt and
	// mmapExpertBytesAt, so the two cannot disagree about which experts are in at a given slot count.
	capAt := func(full []int64) func(slots int) int64 {
		return func(slots int) int64 {
			// Cap one expert-kind's full byte sum at `slots` experts when paging applies. nExperts is the expert count, not the
			// matrix count (each expert holds several matrices). A layer's experts are uniform in shape, so per-expert bytes =
			// full/nExperts exactly.
			var total int64
			for k, fullK := range full {
				nExperts := expertCounts[k]
				if nExperts == 0 || slots <= 0 || slots >= nExperts {
					total += fullK
					continue
				}
				total += fullK / int64(nExperts) * int64(slots)
			}
			return total
		}
	}
	return dense, mmapDense, capAt(fullExpertBytes), capAt(mmapExpertBytes)
}
