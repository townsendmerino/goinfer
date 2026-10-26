package vision

import (
	"fmt"
	"math"
)

// Gemma 4 tower export: a device-resident tower (goinfer's Metal one, for EmbeddingGemma 2 and Gemma 4) reads the
// loaded weights through these types, runs the patch embed and the encoder layers itself, and hands the final hidden
// states back to FinishHidden for the pooling and projection, so the host tail has one implementation. The weights
// are float32 only (LoadGemma4Encoder quant=false); Weights errors otherwise.

// Gemma4Proj is one projection, nn.Linear's [Out, In] weight in float32, and its Gemma4ClippableLinear bounds
// (±Inf when the checkpoint does not clip).
type Gemma4Proj struct {
	W                            []float32
	Out, In                      int
	InMin, InMax, OutMin, OutMax float32
}

// Clipped reports whether any of the projection's bounds is finite.
func (p Gemma4Proj) Clipped() bool {
	for _, v := range []float32{p.InMin, p.InMax, p.OutMin, p.OutMax} {
		if !math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

// Gemma4Layer is one encoder layer: four hidden-wide RMSNorm weights (applied as x·w, no 1+w), the attention and MLP
// projections, and the head-wide q and k norm weights (v's norm has none).
type Gemma4Layer struct {
	InputNorm, PostAttnNorm, PreFFNNorm, PostFFNNorm []float32
	Q, K, V, O                                       Gemma4Proj
	QNorm, KNorm                                     []float32
	Gate, Up, Down                                   Gemma4Proj
}

// Gemma4Weights is the tower as Forward runs it up to the pool. The patches enter as [0, 1] and are mapped to
// 2x-1 before PatchEmbed; PosEmbX and PosEmbY are [PosEmbTableSize, hidden], added by each patch's x and y.
type Gemma4Weights struct {
	Cfg              Gemma4EncoderConfig
	RopeTheta        float64
	PatchEmbed       []float32 // [hidden, 3*patch*patch]
	PosEmbX, PosEmbY []float32
	Layers           []Gemma4Layer
}

// Weights exports the tower's weights (the slices alias the encoder's; do not write them).
func (e *Gemma4Encoder) Weights() (Gemma4Weights, error) {
	if err := e.ensureBlocks(); err != nil {
		return Gemma4Weights{}, err
	}
	var err error
	proj := func(name string, p clippedProj) Gemma4Proj {
		f, ok := p.w.F32()
		if !ok {
			f = widenHalf16(p.w) // a 16-bit-held weight (aliasQMat) exports as its exact f32 widening
			ok = f != nil
		}
		if !ok && err == nil {
			err = fmt.Errorf("vision: %s is %s, the export needs float32 weights (LoadGemma4Encoder quant=false)", name, p.w.Kind())
		}
		return Gemma4Proj{W: f, Out: p.w.Rows(), In: p.w.Cols(), InMin: p.inMin, InMax: p.inMax, OutMin: p.outMin, OutMax: p.outMax}
	}
	w := Gemma4Weights{Cfg: e.Cfg, RopeTheta: e.ropeTheta(), PatchEmbed: e.patchEmbedW, PosEmbX: e.posEmbX, PosEmbY: e.posEmbY,
		Layers: make([]Gemma4Layer, len(e.layers))}
	for l := range e.layers {
		lw := &e.layers[l]
		n := func(s string) string { return fmt.Sprintf("layer %d %s", l, s) }
		w.Layers[l] = Gemma4Layer{InputNorm: lw.inputNormW, PostAttnNorm: lw.postAttnNormW, PreFFNNorm: lw.preFFNNormW,
			PostFFNNorm: lw.postFFNNormW, Q: proj(n("q_proj"), lw.qProj), K: proj(n("k_proj"), lw.kProj),
			V: proj(n("v_proj"), lw.vProj), O: proj(n("o_proj"), lw.oProj), QNorm: lw.qNormW, KNorm: lw.kNormW,
			Gate: proj(n("gate_proj"), lw.gateProj), Up: proj(n("up_proj"), lw.upProj), Down: proj(n("down_proj"), lw.downProj)}
	}
	if err != nil {
		return Gemma4Weights{}, err
	}
	return w, nil
}

func (e *Gemma4Encoder) ropeTheta() float64 {
	if r := e.Cfg.RopeParameters; r != nil && r.RopeTheta != 0 {
		return r.RopeTheta
	}
	return 100.0
}

// Gemma4RopeTables is the tower's axial 2-D RoPE: cos and sin [len(positionIDs), headDim], the first half of each
// head rotating (rotate-half within that half) by x and the second half by y (see gemma4RopeTables).
func Gemma4RopeTables(positionIDs [][2]int, headDim int, theta float64) (cos, sin []float32) {
	return gemma4RopeTables(positionIDs, headDim, theta)
}

// FinishHidden is Forward's tail from the last layer's output h [len(positionIDs), hidden]: the 3x3 average pool, the
// √hidden scale, standardize when the checkpoint has it, the unscaled RMSNorm and the projection to the text width.
// It returns the soft tokens [n, TextHiddenSize].
func (e *Gemma4Encoder) FinishHidden(h []float32, positionIDs [][2]int) ([]float32, error) {
	hidden := e.Cfg.HiddenSize
	np := len(positionIDs)
	if np == 0 || len(h) != np*hidden {
		return nil, fmt.Errorf("vision: %d hidden values for %d patches of %d", len(h), np, hidden)
	}
	return e.finish(h, positionIDs, np)
}
