package decoder

import (
	"context"
	"fmt"
)

// PromptHiddenAll returns the final-norm hidden state at EVERY position of prompt: HF's last_hidden_state[0, :] (which is
// output_hidden_states[-1]), K rows of HiddenDim. It is what a head that reads all positions consumes (D11 of
// docs/tasks/task-constrained-confidence.md, Route C: Clef's joint schema head, docs/measurements/decisions-d10-clef-2026-10-02.md); D2's
// PromptHidden returns only the last row.
//
// CPU ONLY, and it says so rather than falling back silently: unlike PromptHidden it never asks a resident backend, because no resident
// executor exposes every position's hidden state yet (that is D11's own follow-up, a speed lever and not needed for correctness). It takes
// the same CPU paths PromptHidden takes, in the same order, so the last row is the row PromptHidden returns on a model with no resident:
//   - a family on the generic batched path (canBatchN): runLayersFromEmbedN, whose last step applies the final norm to each row;
//   - Qwen3.5 (qwen35BatchN): runLayersQwen35N, which returns the post-final-norm rows;
//   - everything else, and K == 1: one runLayers per token in a fresh cache, then the final norm on a copy of each row (h aliases the
//     decode scratch and normalize mutates in place).
//
// Memory is K x HiddenDim f32 (a 4K-token prompt on a 5120-wide model is about 84 MB), on top of the K x HiddenDim activations the forward
// itself holds; the length bound is checkHiddenIDs's. The rows are the caller's to keep or modify. ctx is checked once per layer (the
// Qwen3.5 batched path) or per token (the sequential path), so a long prompt can be abandoned.
//
// The batched paths are bounded against the sequential one by TestPromptHiddenAll_batchedMatchesSequential; they are not claimed bit-identical,
// since a batched matmul may reduce in another order than its matvec.
func (m *Model) PromptHiddenAll(ctx context.Context, prompt []int) ([][]float32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("decoder.PromptHiddenAll: empty prompt")
	}
	if err := m.checkHiddenIDs("decoder.PromptHiddenAll", prompt); err != nil {
		return nil, err
	}
	a := m.w.arch
	if m.canBatchN(len(prompt)) {
		cache := m.NewCache(len(prompt))
		hN, err := m.runLayersFromEmbedN(ctx, m.embedN(prompt), cache, false) // never fast: exact HF parity is the point
		if err != nil {
			return nil, err
		}
		return splitRows(hN, len(prompt), a.HiddenDim), nil
	}
	cache := m.NewCache(len(prompt))
	if m.qwen35BatchN(len(prompt), cache) {
		hN, err := m.runLayersQwen35N(ctx, m.embedN(prompt), cache)
		if err != nil {
			return nil, err
		}
		return splitRows(hN, len(prompt), a.HiddenDim), nil
	}
	return m.promptHiddenAllSequential(ctx, prompt)
}

// promptHiddenAllSequential is PromptHiddenAll's per-token path (the one PromptHidden's fallback is), callable directly so a test can
// bound the batched paths against it.
func (m *Model) promptHiddenAllSequential(ctx context.Context, prompt []int) ([][]float32, error) {
	a := m.w.arch
	cache := m.NewCache(len(prompt))
	out := make([][]float32, len(prompt))
	for i, id := range prompt {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		h, err := m.runLayers(id, cache)
		if err != nil {
			return nil, err
		}
		// Copy before normalizing: h aliases the decode scratch and normalize mutates in place.
		row := append([]float32(nil), h[:a.HiddenDim]...)
		normalize(a, row, m.w.FinalNorm, m.w.FinalNormBias, a.HiddenDim)
		out[i] = row
	}
	return out, nil
}

// splitRows cuts a flat [K*hidden] buffer into K rows that share it. The buffer is the forward's own fresh allocation (the embedded input
// the batched path consumed), not a scratch another call reuses, so the rows are safe to hand out.
func splitRows(flat []float32, k, hidden int) [][]float32 {
	rows := make([][]float32, k)
	for i := range rows {
		rows[i] = flat[i*hidden : (i+1)*hidden : (i+1)*hidden] // cap-limited: an append to one row cannot overwrite the next
	}
	return rows
}

// OutputEmbeddingRow writes the LM head's row for token id (HiddenDim wide, the raw output-embedding row, not normalised) into dst: the separate lm_head
// when the family has one, the input embedding when the head is tied. It is what a head that scores tokens by their output embedding consumes (Clef's
// joint head averages these over each option's tokens, docs/measurements/decisions-d10-clef-2026-10-02.md). The row is read from the weights as LOADED,
// so a quantized load returns the dequantized row of the quantized weight: the quantization is part of the arm being run, not hidden from it.
func (m *Model) OutputEmbeddingRow(id int, dst []float32) error {
	a := m.w.arch
	if id < 0 || id >= a.VocabSize {
		return fmt.Errorf("decoder.OutputEmbeddingRow: token id %d outside the vocabulary [0, %d)", id, a.VocabSize)
	}
	if len(dst) != a.HiddenDim {
		return fmt.Errorf("decoder.OutputEmbeddingRow: dst is %d wide, want %d", len(dst), a.HiddenDim)
	}
	w := &m.w.LMHead
	if a.TiedLMHead || w.Rows() == 0 {
		w = &m.w.Embed
	}
	w.Row(id, dst)
	return nil
}

// HiddenSize is the model's hidden width (the width PromptHiddenAll rows and OutputEmbeddingRow rows have).
func (m *Model) HiddenSize() int { return m.w.arch.HiddenDim }
