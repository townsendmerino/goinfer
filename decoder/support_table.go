package decoder

// S8 (docs/tasks/task-multimodal-support-2026-10.md, "S8, plan and gates"): what the README and docs/multimodal.md
// support table's decoder cells are computed from. Each cell answers "after an image, does this family's turn decode on
// this backend's resident?", from the declarations the runtime itself admits on, so the table cannot say more than the
// code does (it did, once, by hand: Qwen3.5+ on Metal).

// HybridImagePrefillDeclared reports whether backend's resident image prefill builds a Gated-DeltaNet hybrid's recurrent
// state itself (ResidentHybridMRoPEPrefill) for a hybrid with (moe) or without MoE layers. Without it a hybrid's image turn
// takes neither the resident reuse path nor the upload bridge, and prefills and decodes on the CPU (GenerateQwenVLDeepstackSpans).
// Declared here, beside the resident features, so the table is computed without the backend modules; each backend's test
// proves its implementation matches. CUDA, dense hybrids only: no MoE hybrid image gate exists (cuda's HybridMRoPEPrefill).
func HybridImagePrefillDeclared(backend string, moe bool) bool { return backend == "cuda" && !moe }

// gemma4ImageDecodeResident is GenerateGemma4VLSpans's rule for decoding on the resident after a Gemma 4 image turn: a
// bidirectional ("vision") checkpoint or an E-model; a causal, non-E Gemma 4 keeps decoding on the CPU.
func gemma4ImageDecodeResident(bidirectional, eModel bool) bool { return bidirectional || eModel }

// gemma4IsEModel is the E-model test GenerateGemma4VLSpans applies.
func gemma4IsEModel(a *Architecture) bool {
	return a.gemma4 != nil && (a.gemma4.HiddenSizePerLayerInput > 0 || a.gemma4.SharedKVLayers > 0)
}

// imageTurnDecodesResident is the support table's decoder cell: the family is resident-eligible on backend, a recurrent
// hybrid also has the backend's hybrid image prefill, and a Gemma 4 passes gemma4ImageDecodeResident.
func imageTurnDecodesResident(cfg *Config, a *Architecture, backend string) bool {
	if !ResidentEligible(a, backend) {
		return false
	}
	if f, own := a.ownForward(); own && f.Recurrent && !HybridImagePrefillDeclared(backend, a.MoE != nil) {
		return false
	}
	if a.gemma4 != nil && !gemma4ImageDecodeResident(cfg.UseBidirectionalAttention != "", gemma4IsEModel(a)) {
		return false
	}
	return true
}
