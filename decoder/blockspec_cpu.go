package decoder

import "fmt"

// CPU block drafting — MEASURED NEGATIVE. Do not wire this into serving.
//
// Two targets, both LOSSES, and the second one is the decisive measurement:
//
//	Laguna-XS.2  (sparse MoE, 33B-A3B)  3.20 tok/round  ->  0.82x
//	Qwen3-4B     (DENSE)                6.67 tok/round  ->  0.75x
//
// The dense number kills it, because dense was the case that should have worked: the sparse
// result was explainable (each verified row routes to its own top-8 of 256 experts, so an
// 8-row verify touches ~8x the expert weight instead of amortizing), and dense has no such
// excuse — every row reads the same weights.
//
// THE ARITHMETIC IS TERMINAL, not merely discouraging. At 6.67 tok/round measured at 0.75x,
// break-even needs 6.67/0.75 = 8.89 tok/round. The CEILING at block_size 8 is 8.00
// (the anchor plus 7 drafts). Break-even EXCEEDS the maximum achievable acceptance, so no
// drafter, however good, can make this pay on this target — the gap is not an acceptance
// problem to tune away.
//
// WHY, GENERALLY: speculation pays when verifying N rows costs far less than N sequential
// decodes. That is a GPU property — decode there is memory-bandwidth bound (weights stream
// per token), so extra rows ride along nearly free. On CPU the same batched verify costs
// close to N times a decode step, so the drafter's work is pure addition. P10's kill-gate
// found "the DRAFT was the wall, not the verify" on GPU; on CPU the VERIFY is the wall too.
//
// It is kept, unwired, because it is the only non-CUDA implementation of these interfaces
// and because the measurement above is worth more than the code: the next person to propose
// CPU speculation should read this comment first. `NewCPUBlockSpec` has no production caller
// by design.
//
// WHY THIS EXISTS. BlockSpec's orchestration — draft a block, verify it in ONE batched
// pass, accept the agreeing prefix, roll back — is backend-agnostic, but until now the
// only implementation of the two interfaces it drives was CUDA's. That made block
// drafting unavailable to any CPU-only family, which includes every family a resident
// backend declines: Laguna (FeatAttnOutputGate), gpt-oss (FeatAttnSink), the Gemma-4
// E-models (FeatGemma4EModel).
//
// The pieces were all already here — runLayersFromEmbedN does the batched forward and
// already carries the batched capture seam, and blockTrunk does the drafting with its
// own K/V context. What was missing was the adapter between them, which is all this file
// is. It adds no new numerics: verification goes through the SAME runLayersFromEmbedN
// that plain batched prefill uses, which is bit-identical to sequential decode by
// construction (acc64 — see causalAttention's note on why that matters for speculation).
//
// THE ACCEPTANCE MEASUREMENT IS NOT THE SPEEDUP. A sequential verify (feed the anchor,
// then each drafted token) measures acceptance perfectly well and is what
// dflash_accept_test.go does, but it performs exactly as many target forwards as plain
// decoding — so it can never be faster. The saving comes only from verifying the whole
// block in one batched pass, which is what PrefillLastNArgmax below does.

// cpuBlockDrafter adapts a DFlash-family drafter to ResidentBlockDrafter on the CPU,
// holding the drafter's own per-layer K/V context across rounds.
type cpuBlockDrafter struct {
	d   *DFlashDrafter
	m   *Model
	ctx *DFlashContext
}

func (c *cpuBlockDrafter) FuseContext(rows [][]float32) ([][]float32, error) {
	return c.d.FuseContext(c.m.be, rows)
}

func (c *cpuBlockDrafter) ExtendContext(fused [][]float32) error {
	c.d.ExtendContext(c.m.be, c.ctx, fused)
	return nil
}

func (c *cpuBlockDrafter) ContextLen() int { return c.ctx.Len() }

func (c *cpuBlockDrafter) TruncateContext(n int) { c.ctx.TruncateTo(n) }

func (c *cpuBlockDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {
	return c.d.DraftBlockCtx(c.m.be, c.ctx, blockIn)
}

// DraftTokens routes through DraftIDs so a drafter with its OWN reduced-vocab head
// (poolside's Laguna speculators) maps through d2t, while one that borrows the target's
// (z-lab's) takes the LM-head path. The interface's own doc says a block drafter "ships
// none of its own"; that stopped being true with the Laguna pairing.
func (c *cpuBlockDrafter) DraftTokens(trunk [][]float32) ([]int, error) {
	return c.d.DraftIDs(c.m, trunk), nil
}

// cpuDrafterHost is the verify half: a CPU target that can batch-verify a drafted block
// and hand back the tap hidden states the drafter needs for the next round.
type cpuDrafterHost struct {
	m     *Model
	cache *KVCache
	taps  []int
	capt  [][]float32
}

// AttachBlockDrafter is present to satisfy ResidentDrafterHost. The CPU path builds its
// drafter directly (NewCPUBlockSpec) rather than uploading weights to a device, because
// there is no device — the drafter's WeightMats already alias its mmap and the trunk runs
// on them in place. Going through the interface here would also mean type-asserting the
// concrete drafter back out of BlockDrafterWeights, which is exactly the kind of dispatch
// the decoder census exists to keep out of this package.
func (h *cpuDrafterHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return nil, fmt.Errorf("decoder: CPU block drafting is constructed via NewCPUBlockSpec, not AttachBlockDrafter")
}

// SetBatchedCapture arms the tap seam for the next verify. On CPU this is bookkeeping
// only: runLayersFromEmbedN already copies each requested layer's residual for ALL rows
// when cache.captureLayers is set, at zero cost when it is nil.
func (h *cpuDrafterHost) SetBatchedCapture(taps []int) error {
	h.taps = append(h.taps[:0], taps...)
	return nil
}

// BatchedCapture returns the last verify's tap rows as [tap][M*hidden].
func (h *cpuDrafterHost) BatchedCapture() [][]float32 { return h.capt }

// PrefillLastNArgmax verifies M embedding rows in one batched forward starting at
// startPos, returning each row's argmax.
//
// The cache is truncated to startPos FIRST. That is the rollback after a partial accept:
// the previous round committed a whole block's worth of K/V, and only the accepted prefix
// of it is real. Verifying without the truncation would leave the rejected tail in the
// cache and every later position would attend to tokens the model never emitted — which
// does not crash and does not break losslessness (the target still picks each token), it
// just silently conditions on garbage.
func (h *cpuDrafterHost) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("decoder: PrefillLastNArgmax with no rows")
	}
	hidden := h.m.w.arch.HiddenDim
	if !h.cache.TruncateTo(startPos) {
		return nil, fmt.Errorf("decoder: cache could not truncate to %d (ring eviction lost it)", startPos)
	}
	flat := make([]float32, 0, len(embeddings)*hidden)
	for i, e := range embeddings {
		if len(e) != hidden {
			return nil, fmt.Errorf("decoder: embedding row %d has %d dims, want %d", i, len(e), hidden)
		}
		flat = append(flat, e...)
	}
	if len(h.taps) > 0 {
		h.cache.captureLayers = append(h.cache.captureLayers[:0], h.taps...)
		if cap(h.cache.captured) < len(h.taps) {
			h.cache.captured = make([][]float32, len(h.taps))
		}
		h.cache.captured = h.cache.captured[:len(h.taps)]
	} else {
		h.cache.captureLayers = nil
	}
	out, err := h.m.runLayersFromEmbedN(flat, h.cache)
	if err != nil {
		return nil, err
	}
	h.capt = h.cache.captured
	// lmHeadN, NOT logitsFromHidden. runLayersFromEmbedN has ALREADY applied the final
	// norm to every row, and logitsFromHidden applies it again — a second normalization
	// that does not crash and does not look wrong, it just shifts every logit. It
	// diverged from plain greedy at the very FIRST token, which is the only reason it
	// was caught quickly; a subtler version of the same mistake would have looked like
	// a rollback bug. (DrafterHeadLogits carries a comment about exactly this hazard on
	// the drafter side; this is the target side of it.)
	vocab := h.m.w.arch.VocabSize
	logits := h.m.lmHeadN(out, len(embeddings))
	ids := make([]int, len(embeddings))
	for i := range embeddings {
		ids[i] = argmax(logits[i*vocab : i*vocab+vocab])
	}
	return ids, nil
}

// NewCPUBlockSpec builds a block-drafting speculator that runs entirely on the CPU
// forward path, for targets no resident backend will host.
//
// It deliberately does NOT go through m.NewBlockSpec: that one requires m.resident to
// implement ResidentDrafterHost, and the whole point here is a target with no resident
// backend at all.
func (m *Model) NewCPUBlockSpec(d *DFlashDrafter, ctxLen int) (*BlockSpec, error) {
	if d == nil {
		return nil, fmt.Errorf("decoder: nil drafter")
	}
	g := d.DrafterGeometry()
	if g.Hidden != m.w.arch.HiddenDim {
		return nil, fmt.Errorf("decoder: drafter hidden %d != target hidden %d — wrong pairing; "+
			"a drafter reads the target's residual stream directly", g.Hidden, m.w.arch.HiddenDim)
	}
	for _, l := range d.TargetLayerIDs() {
		if l < 0 || l >= m.w.arch.NumLayers {
			return nil, fmt.Errorf("decoder: drafter taps layer %d, outside the target's %d layers", l, m.w.arch.NumLayers)
		}
	}
	if !m.canBatchN(2) {
		return nil, fmt.Errorf("decoder: target cannot batch-verify (canBatchN false) — block " +
			"drafting would do no fewer forwards than plain decoding")
	}
	host := &cpuDrafterHost{m: m, cache: m.NewCache(ctxLen)}
	return &BlockSpec{
		m:    m,
		host: host,
		rd:   &cpuBlockDrafter{d: d, m: m, ctx: d.NewContext()},
		dw:   d,
		taps: d.TargetLayerIDs(),
	}, nil
}

var (
	_ ResidentBlockDrafter = (*cpuBlockDrafter)(nil)
	_ ResidentDrafterHost  = (*cpuDrafterHost)(nil)
)
