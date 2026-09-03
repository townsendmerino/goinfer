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
func (m *Model) ResidentWeightBytes() int64 { return m.residentWeightBytes(0) }

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
func (m *Model) ResidentWeightBytesPaged(slots int) int64 { return m.residentWeightBytes(slots) }

func (m *Model) residentWeightBytes(slots int) int64 {
	if m == nil || m.w == nil {
		return 0
	}
	w := m.w
	n := wmBytes(&w.Embed) + wmBytes(&w.LMHead) + wmBytes(&w.PosEmbed)
	// Gemma 4's model-level PLE tables (per_layer_token_embd / per_layer_model_proj) — empty
	// WeightMats, so a no-op sum, on every other family.
	n += wmBytes(&w.PerLayerTokenEmbed) + wmBytes(&w.PerLayerModelProj)
	// pagedExperts takes the layer's FULL routed-expert byte sum and its expert COUNT (not the
	// matrix count — each expert contributes multiple matrices, e.g. Gate+Up+Down, so the two
	// must not be conflated) and caps it at `slots` experts when paging applies. A layer's
	// experts are uniform in shape, so per-expert bytes = full/nExperts exactly.
	pagedExperts := func(full int64, nExperts int) int64 {
		if nExperts == 0 || slots <= 0 || slots >= nExperts {
			return full
		}
		return full / int64(nExperts) * int64(slots)
	}
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
		n += pagedExperts(expertBytes, len(l.Experts))
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
			n += pagedExperts(fusedBytes, len(mo.expertsGateUp))
		}
	}
	return n
}
