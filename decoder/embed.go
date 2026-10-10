package decoder

import (
	"context"
	"fmt"
	"sync/atomic"
)

// Decoder-as-embedder seam (docs/completed/task-decoder-as-embedder.md). qwen3-embedding and embeddinggemma are causal
// decoders used as embedders: the sentence-transformers stack around them is Transformer -> Pooling -> Normalize, so the
// decoder must expose the final hidden state. forward() consumes it through the LM head (logitsFromHidden normalizes it
// in place), and ForwardCapture returns per-layer residuals, which are pre-final-norm and not what pooling wants.

// HiddenLast runs ids causally through the layer stack in one fresh KV cache and returns the final hidden state of the
// last token after the model's final norm: HF's `last_hidden_state[:, -1, :]`, which sentence-transformers' last-token
// pooling reads. It stops before the LM head, which an embedder never needs and which is the most expensive matmul in a
// forward.
//
// It takes one sequence, so the last token is always the last real token; there is no padded batch whose last slot
// could be a pad. Callers that want a batch call it per sequence. The returned slice is a fresh copy the caller owns.
//
// Generic decode path only: families with their own runLayers return an error rather than a silently wrong vector, the
// same contract and guard as ForwardCapture.
func (m *Model) HiddenLast(ids []int) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("decoder.HiddenLast: empty token sequence")
	}
	a := m.w.arch
	// Derived from the dispatch table (ownForward), not a hand-written list.
	if _, own := a.ownForward(); own {
		return nil, fmt.Errorf("decoder.HiddenLast: hidden-state seam not wired for arch %q (own runLayers)", a.Name)
	}
	if err := m.checkHiddenIDs("decoder.HiddenLast", ids); err != nil {
		return nil, err
	}
	// On a GPU box this arch may decode resident, so try the resident backend first, under the same resBusy claim Generate
	// uses: a loser (a generation already in flight on this Model) falls through to the CPU path below. resIDs is left
	// unknown either way (residentForgetIDs), since a HiddenLast prefill leaves nothing durable to reuse in the next
	// Generate call.
	if m.resident != nil {
		if rh, ok := m.resident.(ResidentHiddenLast); ok && m.tryClaimResident() {
			out, err := m.hiddenLastResident(context.Background(), rh, ids)
			atomic.StoreInt32(&m.resBusy, 0)
			if err == nil {
				return out, nil
			}
			// A resident decline (OOM, cap, backend-specific refusal) is not a request failure —
			// fall through to the CPU path exactly as if this arch had no resident backend at all.
		}
	}
	// canBatchN excludes K==1, the own-runLayers families (rejected above) and the NonGatedMLP/LearnedPosEmbed families
	// runLayersFromEmbedN does not implement; those keep the per-token loop (hiddenLastSequential). Everything else runs
	// the whole sequence through the same batched prefill path plain generation's prompt phase uses.
	if m.canBatchN(len(ids)) {
		return m.hiddenLastBatched(ids)
	}
	return m.hiddenLastSequential(ids)
}

// hiddenLastResident runs the resident twin of hiddenLastBatched: embed ids on the host (the
// same embedResident the plain resident forward uses) and hand the whole sequence to the
// backend's ResidentHiddenLast in one call, starting at position 0 — HiddenLast never reuses a
// previous resident KV (each call is a fresh sequence), unlike Generate's prefix reuse.
func (m *Model) hiddenLastResident(ctx context.Context, rh ResidentHiddenLast, ids []int) ([]float32, error) {
	m.residentForgetIDs()
	embs := make([][]float32, len(ids))
	for i, id := range ids {
		embs[i] = m.embedResident(id)
	}
	return rh.HiddenLast(ctx, embs, 0)
}

// hiddenLastBatched is HiddenLast's fast path: one batched forward over the whole
// sequence via runLayersFromEmbedN, which returns rows already past the final
// norm (its own doc: "post-final-norm hidden states") — this must NOT normalize
// again, unlike hiddenLastSequential.
func (m *Model) hiddenLastBatched(ids []int) ([]float32, error) {
	a := m.w.arch
	cache := m.NewCache(len(ids))
	hN, err := m.runLayersFromEmbedN(context.TODO(), m.embedN(ids), cache, false) // never fast: exact HF parity is the point
	if err != nil {
		return nil, err
	}
	last := hN[(len(ids)-1)*a.HiddenDim : len(ids)*a.HiddenDim]
	return append([]float32(nil), last...), nil
}

// hiddenLastSequential is HiddenLast's original path: one runLayers call per
// token. Kept as the fallback for K==1 and the families canBatchN excludes, and
// callable directly so a test can compare it against hiddenLastBatched.
func (m *Model) hiddenLastSequential(ids []int) ([]float32, error) {
	a := m.w.arch
	cache := m.NewCache(len(ids))
	var h []float32
	for _, id := range ids {
		var err error
		if h, err = m.runLayers(id, cache); err != nil {
			return nil, err
		}
	}
	// Copy before normalizing: h aliases the decode scratch and normalize mutates in place (the
	// same in-place norm logitsFromHidden does on its way to the head).
	out := append([]float32(nil), h[:a.HiddenDim]...)
	normalize(a, out, m.w.FinalNorm, m.w.FinalNormBias, a.HiddenDim)
	return out, nil
}

// checkHiddenIDs is HiddenLast's and PromptHidden's input check: a length bound and a vocab bound. who prefixes the error.
func (m *Model) checkHiddenIDs(who string, ids []int) error {
	a := m.w.arch
	// A length bound, not just a vocab one. This preallocates KV for len(ids) positions and runs one sequential forward per
	// token with no context to cancel it, so an over-long input is a huge allocation (hundreds of GB at 500k positions) plus
	// attention over up to len(ids) keys per token, holding the caller's mutex until the process is OOM-killed. Positions
	// past the window would also pool from out-of-range RoPE: plausible and wrong. The serving embedder truncates to
	// MaxPositions first; this states the bound where the cost is incurred so a different caller cannot reintroduce it.
	// The key is m.Config().MaxPositions (max_position_embeddings), not a.MaxPositions: the Architecture field of that name
	// is the GPT-2 learned-position table size and is 0 for every RoPE family, so keying on it would make this guard
	// silently inert.
	if mp := m.Config().MaxPositions; mp > 0 && len(ids) > mp {
		return fmt.Errorf("%s: %d tokens exceeds the model's context window of "+
			"%d (context_length_exceeded); truncate before pooling", who, len(ids), mp)
	}
	for i, id := range ids {
		if id < 0 || id >= a.VocabSize {
			return fmt.Errorf("%s: token %d at index %d out of vocab [0,%d)", who, id, i, a.VocabSize)
		}
	}
	return nil
}

// PromptHidden returns the final-norm hidden state at the last position of prompt: HF's output_hidden_states[-1][:, -1],
// equal to last_hidden_state[:, -1]. It is the input a Route B decision head reads
// (docs/tasks/task-constrained-confidence.md): a linear head over the last prompt token's final-norm hidden state.
//
// Unlike HiddenLast it serves every family, including those with their own layer loop: it runs the per-token forward
// Generate's CPU path runs (runLayers), in a fresh cache, then the final norm logitsFromHidden applies before the LM
// head, so the hidden state is the one the logits are computed from. Families on the generic batched path (canBatchN)
// take hiddenLastBatched, which ends at the same final norm, and Qwen3.5 takes runLayersQwen35N;
// TestPromptHidden_batchedMatchesSequential bounds both against the per-token forward.
//
// On a resident backend it runs on the device first, through ResidentHiddenLast, and falls back to the CPU on a decline.
// The resident's numerics are its kernels', not the CPU reference's; TestPromptHidden_residentMatchesCPU in the backend
// modules bounds the difference. ctx is checked between tokens (CPU) or at the backend's own granularity, so a long
// prompt can be abandoned.
func (m *Model) PromptHidden(ctx context.Context, prompt []int) ([]float32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("decoder.PromptHidden: empty prompt")
	}
	if err := m.checkHiddenIDs("decoder.PromptHidden", prompt); err != nil {
		return nil, err
	}
	// A resident backend answers first, through the seam HiddenLast uses, over the whole prompt up to the final norm. A
	// decline (an arch the headless forward does not cover, a paged MoE, a cap) falls through to the CPU as if there were no
	// resident; a cancellation returns. The resident's prefix-reuse record is forgotten either way, since the prompt
	// overwrote its KV.
	if rh, ok := m.resident.(ResidentHiddenLast); ok && m.tryClaimResident() {
		out, err := m.hiddenLastResident(ctx, rh, prompt)
		atomic.StoreInt32(&m.resBusy, 0)
		if err == nil {
			return out, nil
		}
		if ctx.Err() != nil {
			return nil, err
		}
	}
	if m.canBatchN(len(prompt)) {
		return m.hiddenLastBatched(prompt)
	}
	a := m.w.arch
	cache := m.NewCache(len(prompt))
	if m.qwen35BatchN(len(prompt), cache) {
		hN, err := m.runLayersQwen35N(ctx, m.embedN(prompt), cache)
		if err != nil {
			return nil, err
		}
		return append([]float32(nil), hN[(len(prompt)-1)*a.HiddenDim:]...), nil
	}
	var h []float32
	for _, id := range prompt {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var err error
		if h, err = m.runLayers(id, cache); err != nil {
			return nil, err
		}
	}
	// Copy before normalizing: h aliases the decode scratch and normalize mutates in place.
	out := append([]float32(nil), h[:a.HiddenDim]...)
	normalize(a, out, m.w.FinalNorm, m.w.FinalNormBias, a.HiddenDim)
	return out, nil
}
