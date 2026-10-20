package decoder

// The f16 KV cache is a backend's capability, so the policy for it lives with the backend: a backend registers a KVF16Policy (cuda/kvpolicy.go does), and the decoder asks it.
// Which families a backend covers, which build options it declines for, and whether "auto" means f16 there are all the backend's to change without touching this
// package. What the decoder owns is the resolution order and the facts a policy may look at.

// KVFacts is what a backend's KV policy may look at: the architecture's properties that decide whether a KV cache kernel path applies, without the Architecture itself.
type KVFacts struct {
	Family        string // Architecture.Name ("qwen2", "llama", …)
	MLA           bool   // a latent KV cache (DeepSeek, Kimi)
	MoE           bool
	SlidingWindow bool // any sliding-window layer
	PairwiseRoPE  bool // adjacent-pair rotation (Cohere)
	MRoPE         bool // multi-section rotary (the VL families)
	Hybrid        bool // a recurrent or per-layer-geometry mixer: Qwen3.5 DeltaNet, Gemma 4, Laguna, Granite, LFM2
}

// KVF16Policy is one backend's rule for an f16 KV cache.
type KVF16Policy struct {
	// Decline returns why the backend cannot hold this model's KV cache as f16 under these options, or "" when it can.
	Decline func(f KVFacts, o Options) string
	// Auto reports whether "auto" (and the zero KVPrecision) means f16 for a model Decline lets through.
	Auto func() bool
}

var kvPolicies = map[string]KVF16Policy{}

// RegisterKVF16Policy installs a backend's f16 KV policy under its name. Call it from the backend package's init.
func RegisterKVF16Policy(backend string, p KVF16Policy) { kvPolicies[backend] = p }

func (a *Architecture) kvFacts() KVFacts {
	return KVFacts{
		Family:        a.Name,
		MLA:           a.mla != nil,
		MoE:           a.MoE != nil,
		SlidingWindow: a.SlidingWindow > 0,
		PairwiseRoPE:  a.pairwiseRoPE(),
		MRoPE:         len(a.MRopeSection) > 0,
		Hybrid:        a.qwen35 != nil || a.gemma4 != nil || a.laguna != nil || a.granite != nil || a.lfm2 != nil,
	}
}

// kvF16Decline is the named backend's reason for holding this model's KV cache at f32, "" when it can be f16 or the backend has no policy.
func kvF16Decline(backend string, a *Architecture, o Options) string {
	p, ok := kvPolicies[backend]
	if !ok || a == nil {
		return ""
	}
	return p.Decline(a.kvFacts(), o)
}

// KVF16DeclineReason is the reason a load asked for f16 KV is held at f32 by its backend's policy, "" when it is not held back or the backend has none. serve's
// banner prints it.
func (m *Model) KVF16DeclineReason() string {
	if m == nil || m.be == nil || m.kvF16 {
		return ""
	}
	return m.kvF16Decline
}
