package arch

// OwnLoopFamily reports whether a is one of the nine families that ran their own layer loop before the program walker (gemma4, the qwen3.5
// family, lfm2, granitemoehybrid, nemotron_h, bailing_hybrid, deepseek_v2/v3 and kimi_k2, llama4_text, gpt-oss). Their layers
// now run through the program walker, and the hidden-state seams serve them; the refusals that are not seams keep their
// outcome and text from before the walker: f16 KV's reason, the batched CPU decode's and LoRA's.
func (a *Architecture) OwnLoopFamily() bool {
	return a.Gemma4 != nil || a.Qwen35 != nil || a.LFM2 != nil || a.Granite != nil || a.Nemotron != nil || a.KDA != nil ||
		a.MLA != nil || a.Llama4 != nil || a.GPTOSS != nil
}
