package cpuengine

import "github.com/townsendmerino/goinfer/internal/cost"

// CacheDiag holds per-sequence operation-cost telemetry, read-only hidden/sublayer capture seams
// (for block drafters and resident parity localization), layer-input overrides, and speculative tree verification masks.
type CacheDiag struct {
	// Cost, when set, counts this stream's operations into the phase CostPhase (cost_cpu.go); nil is off.
	Cost      *cost.Counter
	CostPhase cost.Phase

	// CaptureLayers, when non-nil, requests that runLayersFromEmbed copy the residual stream (the layer output) after each listed
	// layer index into Captured[i]: the read-only hidden-state seam a draft head reads (block drafters). nil = no capture, zero
	// overhead. The copies never feed back into the forward, so the token output is byte-identical.
	CaptureLayers []int
	Captured      [][]float32

	// LayerInputs, when non-nil, replaces the residual entering layer l with LayerInputs[l] wherever that row is non-nil, in the
	// single-token walk (walkToken) only: a resident diff runs each CPU layer on the GPU's own input, so a layer's difference is
	// its own rounding and the K/V it appends are computed from the same input. nil in production, where it costs one nil check
	// per layer.
	LayerInputs [][]float32

	// SubCapture, when true, records the per-sublayer contribution added to the residual —
	// scr.sub after attention (post sandwich-norm) and after the MLP — for every layer, into
	// SubAttn[l]/SubMLP[l]. Diagnostic seam, same contract as CaptureLayers: copies never feed back, so the token output is
	// byte-identical.
	SubCapture bool
	SubAttn    [][]float32 // [layer][hidden] attention contribution (o-proj out, post sandwich-norm)
	SubMLP     [][]float32 // [layer][hidden] MLP contribution (down out, post sandwich-norm)
	SubMLPPre  [][]float32 // [layer][hidden] MLP down output BEFORE the post-MLP sandwich norm
	SubCtx     [][]float32 // [layer][qDim] attention CONTEXT, pre-o-proj (AttendBatchedHeads out)
	SubPLE     [][]float32 // [layer][hidden] Gemma 4's PLE-branch contribution (post-PLE norm); nil unless the layer capture asked for it

	// TreeRowPos / TreeMask, when non-nil, switch the batched verify (forwardN) from a linear causal chain to tree attention: row i
	// takes its RoPE position from TreeRowPos[i] (its depth) instead of startPos+i, and among the K new batch keys attends only to
	// columns j with TreeMask[i][j] true (its ancestor path, including itself), plus the whole committed prefix. nil = the ordinary
	// linear behaviour, byte-identical, zero overhead.
	TreeRowPos []int
	TreeMask   [][]bool
}

func (d *CacheDiag) hasPerPassCapture() bool {
	return d.CaptureLayers != nil || d.SubCapture || d.TreeRowPos != nil || d.TreeMask != nil
}
