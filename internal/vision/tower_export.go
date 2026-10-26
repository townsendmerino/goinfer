package vision

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// Device-tower exports for the Qwen3.5+ and GLM-OCR towers, on the pattern of the Gemma 4 one (gemma4_export.go): a
// device-resident tower (goinfer's Metal and CUDA ones) reads the loaded float32 weights through these types, runs the
// patch embed and the encoder blocks itself, and hands the last block's output back to the tower's FinishHidden for
// the tail, so the host tail has one implementation. Host-side tables that depend only on the grid (the 2-D RoPE, the
// Qwen3 interpolated position rows) come from the encoder's own code. The weights are float32 only (a tower loaded
// with quant=false); Weights errors otherwise.

// VisionProj is one biased linear layer: W is nn.Linear's [Out, In] weight in float32, B its [Out] bias (nil when the
// layer has none). Both alias the encoder's (W is a fresh widening for a 16-bit-held weight); do not write them.
type VisionProj struct {
	W, B    []float32
	Out, In int
}

// exportProj wraps one projection, recording the first non-float32 one in *err. A 16-bit-held weight (aliasQMat over a bf16
// or f16 checkpoint) exports as its exact f32 widening, a fresh slice: the f32 values a copying load held.
func exportProj(err *error, name string, w linalg.WeightMat, b []float32) VisionProj {
	f, ok := w.F32()
	if !ok {
		f = widenHalf16(w)
		ok = f != nil
	}
	if !ok && *err == nil {
		*err = fmt.Errorf("vision: %s is %s, the export needs float32 weights (load the tower with quant=false)", name, w.Kind())
	}
	return VisionProj{W: f, B: b, Out: w.Rows(), In: w.Cols()}
}

// VisionSegments is the attention segmentation the Qwen3 and GLM-OCR towers use: cumulative patch offsets, one segment
// per image frame (t·h·w patches each), starting at 0. A patch attends to every patch of its own segment only.
func VisionSegments(gridTHW [][3]int) []int { return cuSeqlensFull(gridTHW) }

// --- Qwen3.5+ ---

// Qwen3Block is one Qwen3 tower block: LayerNorm (weight and bias, eps Qwen3Weights.LNEps), fused biased qkv
// ([3][heads][head_dim] rows), biased proj, LayerNorm, biased fc1, GELU-tanh, biased fc2.
type Qwen3Block struct {
	Norm1W, Norm1B []float32
	QKV, Proj      VisionProj
	Norm2W, Norm2B []float32
	FC1, FC2       VisionProj
}

// Qwen3Weights is the tower as ForwardViT runs it: the patch embed ([hidden, C·T·P·P] and its bias) applied to the
// pre-patchified pixel values, then PositionEmbeds added, then the blocks. Attention scale is 1/sqrt(head_dim), RoPE
// from RopeTables, segments from VisionSegments.
type Qwen3Weights struct {
	Cfg            Qwen3EncoderConfig
	PatchW, PatchB []float32
	Blocks         []Qwen3Block
	LNEps          float64
}

// Weights exports the tower's weights (the slices alias the encoder's; do not write them).
func (e *Qwen3VisionEncoder) Weights() (Qwen3Weights, error) {
	if err := e.ensureBlocks(); err != nil {
		return Qwen3Weights{}, err
	}
	var err error
	w := Qwen3Weights{Cfg: e.Cfg, PatchW: e.patchW, PatchB: e.patchB, Blocks: make([]Qwen3Block, len(e.blocks)), LNEps: qwen3LNEps}
	for i := range e.blocks {
		b := &e.blocks[i]
		n := func(s string) string { return fmt.Sprintf("block %d %s", i, s) }
		w.Blocks[i] = Qwen3Block{Norm1W: b.norm1w, Norm1B: b.norm1b,
			QKV: exportProj(&err, n("qkv"), b.qkvw, b.qkvb), Proj: exportProj(&err, n("proj"), b.projw, b.projb),
			Norm2W: b.norm2w, Norm2B: b.norm2b,
			FC1: exportProj(&err, n("fc1"), b.fc1w, b.fc1b), FC2: exportProj(&err, n("fc2"), b.fc2w, b.fc2b)}
	}
	if err != nil {
		return Qwen3Weights{}, err
	}
	return w, nil
}

// DeepstackFromHidden is DeepStack merger k's tail (Qwen3-VL; goinfer S10): block Cfg.DeepstackVisualIndexes[k]'s output
// h [n_patches, hidden], tapped by a device tower, into that set's rows [n_patches/merge², out_hidden] (the post-shuffle
// LayerNorm, fc1, GELU erf, fc2), on the host, as FinishHidden is the main merger's. A device tower built from Weights
// runs the blocks; a DeepStack tower's caller must tap these blocks and call this for each, or the sets are lost
// (ForwardDeepstack is the CPU reference).
func (e *Qwen3VisionEncoder) DeepstackFromHidden(k int, h []float32, gridTHW [][3]int) ([]float32, error) {
	if k < 0 || k >= len(e.deepstack) {
		return nil, fmt.Errorf("vision: DeepStack set %d of %d", k, len(e.deepstack))
	}
	if _, err := e.checkHidden(h, gridTHW); err != nil {
		return nil, err
	}
	return e.deepMerge(k, h), nil
}

// PositionEmbeds is the learned position table bilinearly resampled to the grids, one [hidden] row per patch in the
// tower's patch order: the term Embed adds after the patch embed and its bias. Computed exactly as Embed does (f32 tap
// coordinates, each product rounded before the fixed-order sum).
func (e *Qwen3VisionEncoder) PositionEmbeds(gridTHW [][3]int) []float32 {
	n := 0
	for _, g := range gridTHW {
		n += g[0] * g[1] * g[2]
	}
	h := make([]float32, n*e.Cfg.HiddenSize)
	e.addPosEmbed(h, gridTHW)
	return h
}

// RopeTables is the tower's 2-D RoPE: cos and sin [n_patches, head_dim], rotate-half over the full head.
func (e *Qwen3VisionEncoder) RopeTables(gridTHW [][3]int) (cos, sin []float32) {
	n := 0
	for _, g := range gridTHW {
		n += g[0] * g[1] * g[2]
	}
	return e.rotaryCosSin(gridTHW, n)
}

// FinishHidden is Forward's tail from the last block's output h [n_patches, hidden]: the patch merger. It returns the
// merged image embeddings [n_patches/merge², out_hidden].
func (e *Qwen3VisionEncoder) FinishHidden(h []float32, gridTHW [][3]int) ([]float32, error) {
	if _, err := e.checkHidden(h, gridTHW); err != nil {
		return nil, err
	}
	return e.merge(h), nil
}

func (e *Qwen3VisionEncoder) checkHidden(h []float32, gridTHW [][3]int) (int, error) {
	return checkTowerHidden(h, gridTHW, e.Cfg.HiddenSize, e.Cfg.SpatialMergeSize)
}

// --- GLM-OCR ---

// GlmOcrBlock is one GLM-OCR tower block: RMSNorm (weight only, eps GlmOcrWeights.RMSEps), fused biased qkv, per-head
// RMSNorm of q and k (QNorm, KNorm [head_dim], before RoPE), biased proj, RMSNorm, then down(silu(gate)·up) with
// biases.
type GlmOcrBlock struct {
	Norm1W, Norm2W []float32
	QKV, Proj      VisionProj
	QNorm, KNorm   []float32
	Gate, Up, Down VisionProj
}

// GlmOcrWeights is the tower as its blocks run: the patch embed ([hidden, C·T·P·P] and its bias) then the blocks, no
// position table. Attention scale is 1/sqrt(head_dim), RoPE from RopeTables, segments from VisionSegments.
type GlmOcrWeights struct {
	Cfg            GlmOcrEncoderConfig
	PatchW, PatchB []float32
	Blocks         []GlmOcrBlock
	RMSEps         float64
}

// Weights exports the tower's weights (the slices alias the encoder's; do not write them).
func (e *GlmOcrVisionEncoder) Weights() (GlmOcrWeights, error) {
	if err := e.ensureBlocks(); err != nil {
		return GlmOcrWeights{}, err
	}
	var err error
	w := GlmOcrWeights{Cfg: e.Cfg, PatchW: e.patchW, PatchB: e.patchB, Blocks: make([]GlmOcrBlock, len(e.blocks)), RMSEps: e.Cfg.RMSNormEps}
	for i := range e.blocks {
		b := &e.blocks[i]
		n := func(s string) string { return fmt.Sprintf("block %d %s", i, s) }
		w.Blocks[i] = GlmOcrBlock{Norm1W: b.norm1w, Norm2W: b.norm2w,
			QKV: exportProj(&err, n("qkv"), b.qkvw, b.qkvb), Proj: exportProj(&err, n("proj"), b.projw, b.projb),
			QNorm: b.qNormW, KNorm: b.kNormW,
			Gate: exportProj(&err, n("gate"), b.gatew, b.gateb), Up: exportProj(&err, n("up"), b.upw, b.upb),
			Down: exportProj(&err, n("down"), b.downw, b.downb)}
	}
	if err != nil {
		return GlmOcrWeights{}, err
	}
	return w, nil
}

// RopeTables is the tower's 2-D RoPE: cos and sin [n_patches, head_dim], rotate-half over the full head.
func (e *GlmOcrVisionEncoder) RopeTables(gridTHW [][3]int) (cos, sin []float32) {
	n := 0
	for _, g := range gridTHW {
		n += g[0] * g[1] * g[2]
	}
	return e.rotaryCosSin(gridTHW, n)
}

// FinishHidden is Forward's tail from the last block's output h [n_patches, hidden]: post_layernorm, the downsample
// conv (channel-major merge groups) and the merger. It returns the merged image embeddings [n_patches/merge²,
// out_hidden].
func (e *GlmOcrVisionEncoder) FinishHidden(h []float32, gridTHW [][3]int) ([]float32, error) {
	np, err := checkTowerHidden(h, gridTHW, e.Cfg.HiddenSize, e.Cfg.SpatialMergeSize)
	if err != nil {
		return nil, err
	}
	post := make([]float32, len(h))
	rmsNormEpsInto(post, h, e.postNormW, np, e.Cfg.HiddenSize, e.Cfg.RMSNormEps)
	return e.merge(e.downsample(post)), nil
}

// checkTowerHidden checks h against the grids: n_patches·hidden values, each grid's h and w multiples of merge.
func checkTowerHidden(h []float32, gridTHW [][3]int, hidden, merge int) (int, error) {
	n := 0
	for _, g := range gridTHW {
		if g[0] < 1 || g[1] < 1 || g[2] < 1 || g[1]%merge != 0 || g[2]%merge != 0 {
			return 0, fmt.Errorf("vision: grid %v is not positive multiples of the merge size %d", g, merge)
		}
		n += g[0] * g[1] * g[2]
	}
	if n == 0 || len(h) != n*hidden {
		return 0, fmt.Errorf("vision: %d hidden values for %d patches of %d", len(h), n, hidden)
	}
	return n, nil
}
