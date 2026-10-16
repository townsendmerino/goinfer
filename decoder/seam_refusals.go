package decoder

// ownLoopFamily reports whether a is one of the nine families that ran their own layer loop before OP3 (gemma4, the qwen3.5
// family, lfm2, granitemoehybrid, nemotron_h, bailing_hybrid, deepseek_v2/v3 and kimi_k2, llama4_text, gpt-oss). Their layers
// now run through the program walker, but OP3 changes no outcome (docs/tasks/task-forward-architecture-2026-10.md §4.8), so the
// hidden-state seams keep refusing them exactly as their loops did: HiddenLast and ForwardSubCapture for all nine, and
// ForwardCapture for those whose loop did not capture (ownLoopCaptured). Enabling the seams for them is a separate change with a
// gate per seam.
func (a *Architecture) ownLoopFamily() bool {
	return a.gemma4 != nil || a.qwen35 != nil || a.lfm2 != nil || a.granite != nil || a.nemotron != nil || a.kda != nil ||
		a.mla != nil || a.llama4 != nil || a.gptoss != nil
}

// ownLoopCaptured reports whether a's own loop captured the residual for ForwardCapture (gemma4, the qwen3.5 family, gpt-oss).
func (a *Architecture) ownLoopCaptured() bool {
	return a.gemma4 != nil || a.qwen35 != nil || a.gptoss != nil
}
