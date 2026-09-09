package decoder

import "github.com/townsendmerino/aikit/linalg"

// wmBytes is one weight matrix's in-memory footprint, read from the BACKING SLICES rather than
// computed from Rows()*Cols(). The two disagree: a quantized matrix carries per-group scales
// alongside its packed payload, and int4 packs two values per byte, so a dimension-derived
// figure is wrong in both directions at once. Asking the matrix what it is holding cannot drift
// from what it actually holds when a new kind is added.
func wmBytes(w *linalg.WeightMat) int64 {
	var n int64
	if q8, sc, _, ok := w.Int8(); ok {
		n += int64(len(q8)) + 4*int64(len(sc))
	}
	if q4, sc, _, ok := w.Int4(); ok {
		n += int64(len(q4)) + 4*int64(len(sc))
	}
	if p4, sc, ok := w.Int4Row4(); ok {
		n += int64(len(p4)) + 4*int64(len(sc))
	}
	if f, ok := w.F32(); ok {
		n += 4 * int64(len(f))
	}
	n += int64(w.SplitHalfBytes())
	return n
}

// f32SliceBytes is wmBytes' counterpart for the families that keep a projection matrix as a plain
// []float32 rather than a linalg.WeightMat (MLA, Mamba-2, LFM2's short conv — all parity-first,
// per their own load-time comments). These are real matrices, not [hidden]-sized norms/biases, so
// they belong in the same sum wmBytes' callers build, not in the "elementwise, rounds to nothing"
// category ResidentWeightBytes' doc comment excuses.
func f32SliceBytes(s []float32) int64 { return 4 * int64(len(s)) }

// ResidentWeightBytes is the total byte footprint of this model's weight matrices — what a
// resident backend must hold to run the whole model on-device, assuming every routed MoE expert
// is resident. See ResidentWeightBytesPaged for the synchronous-paging case
// (GOINFER_METAL_MOE_SLOTS), where only N experts per layer are.
//
// WHY IT EXISTS. A resident backend had no way to ask "will this model fit?" before allocating,
// and nothing else in the tree answers it: Dims() exposes hidden/layers/heads but NOT the expert
// count, so a shape-derived estimate under-reports a sparse MoE by the factor that matters most —
// gpt-oss-20b's experts ARE the model. Measured 2026-08-31: loading an 11.28 GB gpt-oss-20b on
// Metal's resident path on a 16 GB machine drove swap to 35.98 GB of 36 GB and never completed OR
// declined, because the only size guard in the tree caps the KV CONTEXT (metal/backend.go), not
// the weights.
//
// It sums the MATRICES, which is where the bytes are; the elementwise norms/biases are [hidden]-
// sized and round to nothing beside them. That makes this a LOWER BOUND on the real footprint,
// which is the safe direction for a guard: it can fail to refuse a marginal model, but it cannot
// refuse one that would have fit.
//
// This is a quantity we COMPUTE, deliberately — not the OS's account of free memory. Darwin's UBC
// reclaims under pressure, so "available" reports what survived rather than what can be asked
// for; an RSS-keyed ceiling once reported LESS memory at a known failure point than at baseline.
func (m *Model) ResidentWeightBytes() int64 { return m.ResidentWeightBytesPaged(0) }

// ResidentWeightBytesPaged is ResidentWeightBytes under Metal's synchronous MoE paging
// (metal/moe.go, metal/gemma4_moe.go): GOINFER_METAL_MOE_SLOTS=N keeps only N of each layer's
// ROUTED experts resident, staging the rest per token. slots<=0 means unpaged (identical to
// ResidentWeightBytes). SharedExpert is never paged — it is always active, not top-k routed — so
// it is counted in full either way, same as every dense matrix.
//
// M-02: this is what the memory-fit guard was missing. It always summed EVERY expert — the
// unpaged number — even when the caller had asked to page, so a model that would fit paged (e.g.
// Qwen3.5-35B-A3B's 22.1 GB unpaged vs. a few GB at N=64) was declined to CPU on a bound it never
// actually needed. A layer's experts are uniform in shape, so "per-expert bytes" is the full
// per-layer expert sum divided by the expert count — exact, not an approximation across layers.
func (m *Model) ResidentWeightBytesPaged(slots int) int64 {
	dense, expertBytesAt := m.residentWeightBytesSplit()
	return dense + expertBytesAt(slots)
}

// ResidentDenseWeightBytes is ResidentWeightBytesPaged's non-expert term alone: every matrix a
// routed-MoE cap cannot shrink (attention/FFN projections, the mixer/MLA/Mamba/shortConv fields,
// PLE, embeddings/head — everything except l.Experts and gemma4moe's fused experts). A guard that
// must pre-check "does the FIXED part of this model fit" before anything ELASTIC (an MoE expert
// cache) has had a chance to size itself against live free memory (e.g. CUDA's capSlots, which
// runs a bisection search rather than a bound) wants exactly this number, not
// ResidentWeightBytesPaged(0) — that includes every expert too, which would incorrectly decline a
// model whose experts are ABOUT to be sized down to fit.
func (m *Model) ResidentDenseWeightBytes() int64 {
	dense, _ := m.residentWeightBytesSplit()
	return dense
}

// ResidentHostCopyBytes is the portion of ResidentWeightBytesPaged's footprint that a UNIFIED-
// MEMORY backend (Metal — "device" memory IS host RAM) keeps resident in TWO PLACES at once: the
// quantized host WeightMat the loader materializes, and a second, freshly re-packed device buffer
// built from it (metal/model.go's int4Buf), with nothing released in between. See M-02
// (docs/audit-2026-09-02.md): a resident GGUF/safetensors model on Metal was measured landing at
// ~2x the guard's own estimate for exactly this reason.
//
// Dense weights (every non-expert matrix, including the mixer/MLA/Mamba/shortConv projections and
// an UNPAGED model's experts) always double this way — the loader never keeps them mmap-backed
// once quantized. Genuinely PAGED routed experts (0 < slots < nExperts) do NOT: they stream via
// pread straight from the .giw file into their device slot buffer, or via an mmap the OS can
// reclaim under pressure (metal/moe.go, metal/gemma4_moe.go) — an UNSTAGED expert leaves no
// committed host allocation behind to double. So the addend is the FULL weight sum when unpaged
// (slots<=0), or just the dense (non-expert) sum when paged.
func (m *Model) ResidentHostCopyBytes(slots int) int64 {
	dense, expertBytesAt := m.residentWeightBytesSplit()
	if slots > 0 {
		return dense // paged experts stream; only dense doubles
	}
	return dense + expertBytesAt(0) // unpaged: every expert is a materialized host copy too
}

// residentWeightBytesSplit does the one enumeration pass ResidentWeightBytesPaged and
// ResidentHostCopyBytes both need, returning the DENSE (non-expert) sum plus a closure that caps
// the routed-expert sum at `slots` experts (see pagedExperts' own doc) — so the two accessors
// cannot enumerate the model differently and disagree about what "dense" means.
func (m *Model) residentWeightBytesSplit() (dense int64, expertBytesAt func(slots int) int64) {
	if m == nil || m.w == nil {
		return 0, func(int) int64 { return 0 }
	}
	w := m.w
	n := wmBytes(&w.Embed) + wmBytes(&w.LMHead) + wmBytes(&w.PosEmbed)
	// Gemma 4's model-level PLE tables (per_layer_token_embd / per_layer_model_proj) — empty
	// WeightMats, so a no-op sum, on every other family.
	n += wmBytes(&w.PerLayerTokenEmbed) + wmBytes(&w.PerLayerModelProj)
	var fullExpertBytes []int64 // one entry per (layer, expert-kind) — generic l.Experts and gemma4moe's fused set are both "kinds"
	var expertCounts []int
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, mat := range []*linalg.WeightMat{
			&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj,
			&l.GateProj, &l.UpProj, &l.DownProj,
			&l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj,
		} {
			n += wmBytes(mat)
		}
		// The experts are the whole point of this accessor — a sparse MoE is mostly experts, and
		// omitting them is the specific under-report that would let gpt-oss through the guard.
		var expertBytes int64
		for j := range l.Experts {
			e := &l.Experts[j]
			expertBytes += wmBytes(&e.Gate) + wmBytes(&e.Up) + wmBytes(&e.Down)
		}
		fullExpertBytes = append(fullExpertBytes, expertBytes)
		expertCounts = append(expertCounts, len(l.Experts))
		n += wmBytes(&l.SharedExpert.Gate) + wmBytes(&l.SharedExpert.Up) + wmBytes(&l.SharedExpert.Down)

		// M-01: qwen3_5_moe's per-layer mixer (DeltaNet or gated-softmax attention) — the three
		// dominant projections quantize (WeightMat, 2026-08-19); the rest stay f32 vectors small
		// enough to fall under the doc's norms/biases exemption. At most one of these is non-nil.
		if d := l.delta; d != nil {
			n += wmBytes(&d.inProjQKV) + wmBytes(&d.inProjZ) + wmBytes(&d.outProj)
		}
		if q := l.qattn; q != nil {
			n += wmBytes(&q.qProj) + wmBytes(&q.kProj) + wmBytes(&q.vProj) + wmBytes(&q.oProj)
		}
		// M-01: MLA (DeepSeek/Kimi) — parity-first f32, real projection matrices, not norms.
		if mla := l.mla; mla != nil {
			n += f32SliceBytes(mla.qAProj) + f32SliceBytes(mla.qBProj) + f32SliceBytes(mla.qProj) +
				f32SliceBytes(mla.kvAProj) + f32SliceBytes(mla.kvBProj) + f32SliceBytes(mla.oProj)
		}
		// M-01: Mamba-2 (Granite/Nemotron) — parity-first f32; the recurrent STATE is per-sequence
		// and never resident here, only the weights below.
		if mb := l.mamba; mb != nil {
			n += f32SliceBytes(mb.inProj) + f32SliceBytes(mb.convW) + f32SliceBytes(mb.outProj)
		}
		// M-01: LFM2's gated short-convolution mixer — parity-first f32.
		if sc := l.shortConv; sc != nil {
			n += f32SliceBytes(sc.inProj) + f32SliceBytes(sc.convW) + f32SliceBytes(sc.outProj)
		}
		// M-01: Gemma 4's MoE sub-block. mlpGate/mlpUp/mlpDown ALIAS l.GateProj/UpProj/DownProj
		// (serialize.go's gemma4Layer comment) — already counted above; only routerProj and the
		// fused experts are new tensors here. The fused experts page the same way as l.Experts.
		if mo := l.gemma4moe; mo != nil {
			n += wmBytes(&mo.routerProj)
			var fusedBytes int64
			for e := range mo.expertsGateUp {
				fusedBytes += wmBytes(&mo.expertsGateUp[e]) + wmBytes(&mo.expertsDown[e])
			}
			fullExpertBytes = append(fullExpertBytes, fusedBytes)
			expertCounts = append(expertCounts, len(mo.expertsGateUp))
		}
	}
	dense = n
	expertBytesAt = func(slots int) int64 {
		// pagedExperts takes one expert-kind's FULL routed-expert byte sum and its expert COUNT
		// (not the matrix count — each expert contributes multiple matrices, e.g. Gate+Up+Down, so
		// the two must not be conflated) and caps it at `slots` experts when paging applies. A
		// layer's experts are uniform in shape, so per-expert bytes = full/nExperts exactly.
		var total int64
		for k, full := range fullExpertBytes {
			nExperts := expertCounts[k]
			if nExperts == 0 || slots <= 0 || slots >= nExperts {
				total += full
				continue
			}
			total += full / int64(nExperts) * int64(slots)
		}
		return total
	}
	return dense, expertBytesAt
}
