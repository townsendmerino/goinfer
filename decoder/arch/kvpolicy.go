package arch

// KVFacts is what a backend's KV policy may look at: the architecture's properties that decide whether a KV cache kernel path applies, without the Architecture itself.
type KVFacts struct {
	Family        string // Architecture.Name ("qwen2", "llama", …)
	MLA           bool   // a latent KV cache (DeepSeek, Kimi)
	MoE           bool
	SlidingWindow bool // any sliding-window layer
	PairwiseRoPE  bool // adjacent-pair rotation (Cohere)
	MRoPE         bool // multi-section rotary (the VL families)
	Hybrid        bool // a recurrent or per-layer-geometry mixer: Qwen3.5 DeltaNet, Gemma 4, Laguna, Granite, LFM2
	KVRowNotWords int  // the first K/V row width (elements a position) that is not a multiple of 4, 0 when every layer's is
}

func (a *Architecture) KVFacts() KVFacts {
	return KVFacts{
		Family:        a.Name,
		MLA:           a.MLA != nil,
		MoE:           a.MoE != nil,
		SlidingWindow: a.SlidingWindow > 0,
		PairwiseRoPE:  a.PairwiseRoPE(),
		MRoPE:         len(a.MRopeSection) > 0,
		Hybrid:        a.Qwen35 != nil || a.Gemma4 != nil || a.Laguna != nil || a.Granite != nil || a.LFM2 != nil,
		KVRowNotWords: a.kvRowNotWords(),
	}
}

// kvRowNotWords is the first layer's K/V row width that is not a multiple of 4 elements, 0 when none: an int8 row of such a width is not
// whole 4-byte words, which the CUDA cache's block and host copies move.
func (a *Architecture) kvRowNotWords() int {
	for l := range a.NumLayers {
		if d := a.KVDimAt(l); d%4 != 0 {
			return d
		}
	}
	return 0
}
