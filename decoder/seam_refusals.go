package decoder

// ownLoopFamily reports whether a is one of the nine families that ran their own layer loop before OP3 (gemma4, the qwen3.5
// family, lfm2, granitemoehybrid, nemotron_h, bailing_hybrid, deepseek_v2/v3 and kimi_k2, llama4_text, gpt-oss). Their layers
// now run through the program walker, and the hidden-state seams serve them; the refusals that are not seams keep their
// pre-OP3 outcome and text: f16 KV's reason, MC3c's and LoRA's.
func (a *Architecture) ownLoopFamily() bool {
	return a.gemma4 != nil || a.qwen35 != nil || a.lfm2 != nil || a.granite != nil || a.nemotron != nil || a.kda != nil ||
		a.mla != nil || a.llama4 != nil || a.gptoss != nil
}
