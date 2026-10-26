package vision

import "fmt"

// Device-tower export for the SigLIP tower (Gemma 3's), on the pattern of tower_export.go: a device tower (goinfer's
// Metal one) reads the loaded float32 weights through these types, runs the patch embed and the encoder blocks itself
// on GridPatches' output, and hands the last block's output to FinishHidden for the post-layernorm, so the host tail
// has one implementation. Float32 only (LoadEncoder quant=false); the int8 form for the W8A8 device towers is
// GPUWeights (gpu_export.go).

// SiglipBlock is one SigLIP encoder layer: LayerNorm (weight and bias, eps Cfg.LayerNormEps), biased q, k and v,
// attention at scale 1/sqrt(head_dim) over every patch (no RoPE, no mask), biased o, residual; LayerNorm, biased fc1,
// GELU-tanh, biased fc2, residual.
type SiglipBlock struct {
	LN1W, LN1B []float32
	Q, K, V, O VisionProj
	LN2W, LN2B []float32
	FC1, FC2   VisionProj
}

// SiglipWeights is the tower as forwardBlocks runs it: the patch embed ([hidden, C·P·P] and its bias) over
// GridPatches' rows, plus PosEmb ([numPatches, hidden]), then the blocks.
type SiglipWeights struct {
	Cfg                    EncoderConfig
	NumPatches             int
	PatchW, PatchB, PosEmb []float32
	Blocks                 []SiglipBlock
}

// Weights exports the tower's float32 weights (the slices alias the encoder's; do not write them). It errors on a
// tower loaded with quant=true.
func (e *Encoder) Weights() (SiglipWeights, error) {
	if err := e.ensureBlocks(); err != nil {
		return SiglipWeights{}, err
	}
	var err error
	w := SiglipWeights{Cfg: e.Cfg, NumPatches: e.numPatches, PatchW: e.patchW, PatchB: e.patchB, PosEmb: e.posEmb,
		Blocks: make([]SiglipBlock, len(e.layers))}
	for i := range e.layers {
		l := &e.layers[i]
		n := func(s string) string { return fmt.Sprintf("layer %d %s", i, s) }
		w.Blocks[i] = SiglipBlock{LN1W: l.ln1w, LN1B: l.ln1b, LN2W: l.ln2w, LN2B: l.ln2b,
			Q: exportProj(&err, n("q"), l.qw, l.qb), K: exportProj(&err, n("k"), l.kw, l.kb),
			V: exportProj(&err, n("v"), l.vw, l.vb), O: exportProj(&err, n("o"), l.ow, l.ob),
			FC1: exportProj(&err, n("fc1"), l.fc1w, l.fc1b), FC2: exportProj(&err, n("fc2"), l.fc2w, l.fc2b)}
	}
	if err != nil {
		return SiglipWeights{}, err
	}
	return w, nil
}

// FinishHidden is Forward's tail from the last block's output h [numPatches, hidden]: the post-layernorm. It returns
// last_hidden_state, what Forward returns.
func (e *Encoder) FinishHidden(h []float32) ([]float32, error) {
	if want := e.numPatches * e.Cfg.HiddenSize; len(h) != want {
		return nil, fmt.Errorf("vision: %d hidden values, want %d (%d patches of %d)", len(h), want, e.numPatches, e.Cfg.HiddenSize)
	}
	return layerNorm(h, e.postLNw, e.postLNb, e.numPatches, e.Cfg.HiddenSize, e.Cfg.LayerNormEps), nil
}
