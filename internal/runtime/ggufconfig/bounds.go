package ggufconfig

import (
	"fmt"
)

// Generous sanity ceilings for the metadata-derived core dims — orders of
// magnitude above any real checkpoint (largest open models: ~120 layers, hidden
// ~16K, vocab ~256K), but low enough that an untrusted GGUF can't drive a
// pathological allocation. They exist to turn a hostile header into a typed
// error instead of a panic/OOM, not to police legitimate models.
const (
	MaxLayers    = 4096
	MaxHidden    = 1 << 20
	MaxHeads     = 1 << 16
	MaxVocabSize = 1 << 24
	MaxExperts   = 1 << 16
	// MaxIntermediate bounds every feed-forward width. The decode scratch allocates two float32 vectors of IntermediateDim at the first
	// cache whatever the family, so a hostile feed_forward_length makes a multi-terabyte request there (a fatal OOM) with no tensor to
	// give it away. The widest real FFN is about 53K.
	MaxIntermediate = 1 << 20
)

// ggufLayerCount bounds a block_count before any per-layer slice is allocated or iterated. ValidateDims catches an
// out-of-range count too, but only after the family builder has run, and a hostile count would by then have makeslice'd a
// multi-TB array (a fatal OOM, unrecoverable) or, wrapped negative, panicked. Call it right after reading block_count in
// any builder that allocates from it.
func ggufLayerCount(n int) (int, error) {
	if n <= 0 || n > MaxLayers {
		return 0, fmt.Errorf("decoder(gguf): block_count %d out of range (1..%d)", n, MaxLayers)
	}
	return n, nil
}

// ValidateDims rejects core dimensions that are out of range before they reach an allocation. GGUF reads counts as
// uint64 and narrows to int, so a hostile metadata value (e.g. block_count = 1<<63) silently becomes negative and would
// panic make([]LayerWeights, n); a merely-huge positive value would OOM. Both are caught here. Only the dims every
// family sets from metadata are checked (HeadDim is derived per family; IntermediateDim is vestigial or zero for some
// MoE checkpoints).
func ValidateDims(cfg *Config) error {
	switch {
	case cfg.NumLayers <= 0 || cfg.NumLayers > MaxLayers:
		return fmt.Errorf("decoder(gguf): block_count %d out of range (1..%d)", cfg.NumLayers, MaxLayers)
	case cfg.HiddenDim <= 0 || cfg.HiddenDim > MaxHidden:
		return fmt.Errorf("decoder(gguf): embedding_length %d out of range (1..%d)", cfg.HiddenDim, MaxHidden)
	case cfg.NumHeads <= 0 || cfg.NumHeads > MaxHeads:
		return fmt.Errorf("decoder(gguf): head_count %d out of range (1..%d)", cfg.NumHeads, MaxHeads)
	case cfg.NumKVHeads <= 0 || cfg.NumKVHeads > MaxHeads:
		return fmt.Errorf("decoder(gguf): head_count_kv %d out of range (1..%d)", cfg.NumKVHeads, MaxHeads)
	case cfg.VocabSize <= 0 || cfg.VocabSize > MaxVocabSize:
		return fmt.Errorf("decoder(gguf): vocab_size %d out of range (1..%d)", cfg.VocabSize, MaxVocabSize)
	}
	return nil
}
