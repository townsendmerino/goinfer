package cpuengine

import "github.com/townsendmerino/goinfer/internal/runtime/inkling2"

// CacheRecurrent holds the in-place rolling mixer states (Gated DeltaNet, LFM2 short-conv, Mamba-2, KDA, Inkling 2)
// and DeepSeek MLA's compressed-KV latent arrays for one sequence.
type CacheRecurrent struct {
	// RecurrentResets counts ResetRecurrent calls: a diagnostic the test that proves a clean session does not zero its state twice
	// reads. Nothing branches on it.
	RecurrentResets int

	// Delta holds the Gated DeltaNet recurrent state for qwen3_5_moe's linear
	// layers (nil entry on softmax layers, nil slice on every other family).
	Delta []*DeltaState

	// Conv holds the LFM2 gated short-convolution rolling window for the conv layers
	// (nil entry on attention layers, nil slice on every other family).
	Conv []*ShortConvState

	// Mamba holds the Mamba-2 recurrent state ({conv window, SSM state}) for
	// Granite-4.0-H's and Nemotron-H's mamba layers (nil entry on attention layers, nil slice on
	// every other family).
	Mamba []*Mamba2State

	// MLALatent holds DeepSeek MLA's compressed-KV latent per layer, appended [pos*latentDim], latentDim = kv_lora_rank +
	// qk_rope_head_dim (the per-position stride, learned on the first append). The cache holds the low-rank latent, not a
	// reconstructed full K+V; forward_deepseek rebuilds per-head K/V from it each step. nil on every other family; the standard
	// keys/vals stay empty.
	MLALatent [][]float32
	LatentDim int

	// KDA holds Bailing Hybrid's (Ling 3.0) KDA recurrent state (three conv windows + the
	// [headDim,headDim]-per-head matrix state) for the linear layers (nil entry on the MLA
	// layers, which use MLALatent above instead; nil slice on every other family).
	KDA []*KDAState

	// Ink is each Inkling 2 visit's four short-convolution windows; nil for other families.
	Ink []*inkling2.ConvState
}

// AppendLatent stores one position's MLA latent (the compressed KV || rope-key vector)
// for a layer, growing the per-layer store. Append-forever (no ring/quant — the
// latent is already the compressed state). The caller advances pos once per token via
// Advance() (manualPos), since the standard Append path is bypassed.
func (r *CacheRecurrent) AppendLatent(layer int, latent []float32) {
	if r.LatentDim == 0 {
		r.LatentDim = len(latent)
	}
	r.MLALatent[layer] = append(r.MLALatent[layer], latent...)
}

// Latent returns the stored MLA latent history for a layer as [storedPos, latentDim].
func (r *CacheRecurrent) Latent(layer int) []float32 { return r.MLALatent[layer] }

// HasRecurrentState reports whether this cache carries state mutated in place per token with no per-position history: TruncateTo
// cannot rewind it, Snapshot cannot persist it, and a rolled-back session must go cold rather than warm-reuse it. It is one
// predicate over every kind (Mamba, Delta, Conv, KDA), read at every site: a kind missed at one site leaks one
// conversation's window into the next (TruncateTo(0) leaving it intact) and reports a partial rewind exact, so a rewound prefix
// is warm-reused over stale windows. recurrent_census_test.go asks the struct, so a new kind cannot be missed.
func (r *CacheRecurrent) HasRecurrentState() bool {
	return r.Mamba != nil || r.Delta != nil || r.Conv != nil || r.KDA != nil || r.Ink != nil
}

// ResetRecurrent re-zeroes the rolling state of every recurrent kind (Mamba-2, Gated DeltaNet, LFM2 conv, KDA: conv windows and
// SSM / linear-attention state) so a reused cache does not leak the prior sequence's recurrence into a fresh one. No-op on
// non-recurrent families (nil slices).
func (r *CacheRecurrent) ResetRecurrent() {
	r.RecurrentResets++
	// A conv window left behind would bleed the previous sequence's last K-1 tokens into the first K-1 of the next: a small, fluent,
	// entirely wrong prefix.
	for _, st := range r.Conv {
		if st != nil {
			st.ConvWin = nil
		}
	}
	for _, st := range r.Mamba {
		if st != nil {
			st.Reset()
		}
	}
	for _, st := range r.Delta {
		if st != nil {
			st.ConvWin = nil
			for i := range st.S {
				st.S[i] = 0
			}
		}
	}
	// KDA: three per-stream conv windows plus the per-head matrix state, all mutated in place per token, the same leak as above.
	for _, st := range r.KDA {
		if st != nil {
			st.ConvWinQ, st.ConvWinK, st.ConvWinV = nil, nil, nil
			for i := range st.S {
				st.S[i] = 0
			}
		}
	}
	for _, st := range r.Ink {
		if st != nil {
			st.Reset()
		}
	}
}
