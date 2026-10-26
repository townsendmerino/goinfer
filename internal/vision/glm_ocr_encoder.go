package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// GLM-OCR vision tower — aikit's fourth ViT family (GlmOcrVisionModel, zai-org/GLM-OCR). It is
// structurally the Qwen3.5+ tower (qwen3_encoder.go) with a short list of deltas. Read from the
// GENERATED transformers modeling_glm_ocr.py (5.12.0), NOT modular_glm_ocr.py: in the modular file
// GlmOcrVisionMlp calls its parent's constructor (layers built at out_hidden_size, unbiased) and only
// then resets intermediate_size, so read alone it implies the wrong shapes. The generated file and the
// checkpoint's own tensor shapes agree with each other and with this list:
//
//   - patch embed is a BIASED Conv3d [hidden, C, T, P, P], kernel = stride = [T, P, P]; as a matmul
//     over the flattened (channel, temporal, py, px) patch it is patchW·x + bias (the plain matmul
//     over all C·T·P·P inputs is the reference — no temporal-sum shortcut);
//   - NO learned position table and NO post-conv LayerNorm;
//   - 2D rotary exactly as the Qwen towers (theta 1e4, rotary dim head_dim/2 = h freqs then w freqs,
//     cat(freqs, freqs) over the full head_dim, half-split rotate_half), over (row, col) positions;
//   - FULL attention within one image/frame (no windows), fused biased qkv reshaped
//     (seq, 3, heads, head_dim), scale head_dim^-0.5, biased proj; BUT a per-head RMSNorm on q and on
//     k (weight [head_dim], eps rms_norm_eps) applied BEFORE the rotary;
//   - RMSNorm (weight only, eps rms_norm_eps = 1e-5) for norm1 / norm2 / post_layernorm;
//   - a SiLU-GATED MLP, down(silu(gate(x)) * up(x)), all three linears BIASED, intermediate 4096;
//   - post_layernorm (RMSNorm) after the last block, THEN downsample: a biased Conv2d
//     [out_hidden, hidden, m, m] with stride m = spatial_merge_size. HF does
//     hidden.view(-1, m, m, C).permute(0, 3, 1, 2) then the conv, so the input to merge block g is the
//     m² patches 4g..4g+m²-1 (merge-block order: merge-row, merge-col) arranged CHANNEL-MAJOR —
//     x[g][c·m² + dy·m + dx] = hidden[g·m² + dy·m + dx][c] — against the weight flattened (c, dy, dx);
//   - merger (NO biases anywhere except the LayerNorm's): proj [out, out] → LayerNorm(out) WITH
//     weight and bias (torch default eps 1e-5) → GELU **erf** (nn.GELU()) → SiLU-gated
//     down(silu(gate(x)) * up(x)) with inner width out_hidden·in_channels (HF's context_dim; 4608 for
//     the released checkpoint — a coincidence of 1536·3, the loader checks the tensor shapes).
//
// Patches keep their merge-block order end to end. Forward takes pre-patchified pixel_values exactly
// as Qwen3VisionEncoder does: [n_patches, in_channels·temporal·patch²] in merge-block order with
// per-image grids. Added ADDITIVELY — QwenVisionEncoder, Qwen3VisionEncoder and Encoder are
// untouched (the fused-attention schedule is shared via attendPackedInto, moved verbatim).

// GlmOcrEncoderConfig mirrors the HF GlmOcrVisionConfig fields the forward needs.
type GlmOcrEncoderConfig struct {
	Depth             int     `json:"depth"`
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHeads          int     `json:"num_heads"`
	InChannels        int     `json:"in_channels"`
	PatchSize         int     `json:"patch_size"`
	SpatialMergeSize  int     `json:"spatial_merge_size"`
	TemporalPatchSize int     `json:"temporal_patch_size"`
	OutHiddenSize     int     `json:"out_hidden_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	// AttentionBias is a pointer so an ABSENT key takes HF's default (true) while an explicit false
	// is seen and refused: the loader reads biases unconditionally.
	AttentionBias  *bool           `json:"attention_bias"`
	HiddenAct      string          `json:"hidden_act"`
	RopeParameters *glmOcrRopeJSON `json:"rope_parameters"`
}

// glmOcrRopeJSON is the checkpoint's vision rope_parameters. HF's GlmOcrVisionConfig has no such
// field and the module hard-codes theta 1e4, so the only values that match HF are the released ones.
type glmOcrRopeJSON struct {
	RopeTheta float64 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

const (
	glmOcrRotaryTheta = 10000.0 // GlmOcrVisionRotaryEmbedding default theta
	glmOcrMergerLNEps = 1e-5    // torch nn.LayerNorm default; HF builds the merger norm as LayerNorm(dim)
	glmOcrActMLP      = "silu"
	glmOcrDefaultEps  = 1e-5 // GlmOcrVisionConfig.rms_norm_eps default
)

func (c GlmOcrEncoderConfig) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return fmt.Errorf("hidden_size must be > 0, got %d", c.HiddenSize)
	case c.IntermediateSize <= 0:
		return fmt.Errorf("intermediate_size must be > 0, got %d", c.IntermediateSize)
	case c.Depth < 0:
		return fmt.Errorf("depth must be >= 0, got %d", c.Depth)
	case c.NumHeads <= 0:
		return fmt.Errorf("num_heads must be > 0, got %d", c.NumHeads)
	case c.HiddenSize%c.NumHeads != 0:
		return fmt.Errorf("hidden_size %d not divisible by num_heads %d", c.HiddenSize, c.NumHeads)
	case (c.HiddenSize/c.NumHeads)%4 != 0:
		return fmt.Errorf("head_dim %d (hidden_size/num_heads) must be divisible by 4 for rotary", c.HiddenSize/c.NumHeads)
	case c.InChannels <= 0:
		return fmt.Errorf("in_channels must be > 0, got %d", c.InChannels)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	case c.TemporalPatchSize <= 0:
		return fmt.Errorf("temporal_patch_size must be > 0, got %d", c.TemporalPatchSize)
	case c.SpatialMergeSize <= 0:
		return fmt.Errorf("spatial_merge_size must be > 0, got %d", c.SpatialMergeSize)
	case c.OutHiddenSize <= 0:
		return fmt.Errorf("out_hidden_size must be > 0, got %d", c.OutHiddenSize)
	case c.RMSNormEps < 0 || math.IsNaN(c.RMSNormEps):
		return fmt.Errorf("rms_norm_eps must be >= 0, got %g", c.RMSNormEps)
	case c.AttentionBias != nil && !*c.AttentionBias:
		return fmt.Errorf("attention_bias=false unsupported (the qkv/proj/MLP biases are read unconditionally)")
	case c.HiddenAct != glmOcrActMLP:
		// The block and merger MLPs hard-code SiLU; another declared activation would silently run
		// the wrong one.
		return fmt.Errorf("hidden_act %q unsupported (%s only)", c.HiddenAct, glmOcrActMLP)
	case c.RopeParameters != nil && c.RopeParameters.RopeType != "" && c.RopeParameters.RopeType != "axial":
		return fmt.Errorf("rope_parameters.rope_type %q unsupported (axial only)", c.RopeParameters.RopeType)
	case c.RopeParameters != nil && c.RopeParameters.RopeTheta != 0 && c.RopeParameters.RopeTheta != glmOcrRotaryTheta:
		return fmt.Errorf("rope_parameters.rope_theta %g unsupported (HF's GlmOcrVisionRotaryEmbedding hard-codes %g)", c.RopeParameters.RopeTheta, glmOcrRotaryTheta)
	}
	return nil
}

// withDefaults fills the fields HF defaults when absent (in_channels 3, rms_norm_eps 1e-5, hidden_act
// silu, as GlmOcrVisionConfig does).
func (c GlmOcrEncoderConfig) withDefaults() GlmOcrEncoderConfig {
	if c.InChannels == 0 {
		c.InChannels = 3
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = glmOcrDefaultEps
	}
	if c.HiddenAct == "" {
		c.HiddenAct = glmOcrActMLP
	}
	return c
}

func (c GlmOcrEncoderConfig) patchDim() int {
	return c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
}

// mergerInner is the merger MLP's inner width: HF builds it with context_dim = out_hidden_size ·
// in_channels (4608 = 1536·3 for the released checkpoint).
func (c GlmOcrEncoderConfig) mergerInner() int { return c.OutHiddenSize * c.InChannels }

type glmOcrBlock struct {
	norm1w, norm2w []float32
	qkvw           linalg.WeightMat // [3*hidden, hidden] fused
	qkvb           []float32
	projw          linalg.WeightMat // [hidden, hidden]
	projb          []float32
	qNormW, kNormW []float32        // [head_dim]
	gatew, upw     linalg.WeightMat // [inter, hidden]
	gateb, upb     []float32
	downw          linalg.WeightMat // [hidden, inter]
	downb          []float32
}

// GlmOcrVisionEncoder is a loaded GLM-OCR vision tower (dynamic resolution, full attention).
type GlmOcrVisionEncoder struct {
	Cfg        GlmOcrEncoderConfig
	patchW     []float32 // [hidden, patch_dim] (Conv3d weight flattened, kept f32)
	patchB     []float32 // [hidden]
	blocks     []glmOcrBlock
	postNormW  []float32        // post_layernorm [hidden]
	downW      linalg.WeightMat // [out_hidden, hidden*m*m] (Conv2d weight flattened (c, dy, dx))
	downB      []float32        // [out_hidden]
	mProjW     linalg.WeightMat // merger.proj [out, out]
	mLNw, mLNb []float32        // merger.post_projection_norm [out]
	mGateW     linalg.WeightMat // [inner, out]
	mUpW       linalg.WeightMat // [inner, out]
	mDownW     linalg.WeightMat // [out, inner]
	rotInvFreq []float32        // head_dim/4 rotary frequencies
	rel        *blockRelease[GlmOcrVisionEncoder]
	maps       towerMaps // the checkpoint mappings its aliased weights read (aliasQMat); Close releases them
}

// LoadGlmOcrVisionEncoder reads a GLM-OCR checkpoint (config.json vision_config + safetensors, tensors
// under "model.visual.") and returns a ready encoder. Weights are copied out, so the safetensors file
// is closed before return. quant wraps the projections as int8 W8A8 (patch embed and norms stay f32);
// the fp32 parity gate runs quant=false.
func LoadGlmOcrVisionEncoder(dir string, quant bool) (*GlmOcrVisionEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	var wrap struct {
		VisionConfig *GlmOcrEncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	if wrap.VisionConfig == nil {
		return nil, fmt.Errorf("vision: %s has no vision_config (a text-only checkpoint?)", filepath.Join(dir, "config.json"))
	}
	cfg := wrap.VisionConfig.withDefaults()
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	aliased, kept := false, false // a weight aliases st; the loaded tower keeps st open
	defer func() {
		if !kept {
			st.Close()
		}
	}()

	e := &GlmOcrVisionEncoder{Cfg: cfg}
	pfx := glmOcrTensorPrefix(st)
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx+name, want...)
		if err != nil {
			return nil
		}
		return append([]float32(nil), v...)
	}
	hidden, inter, out := cfg.HiddenSize, cfg.IntermediateSize, cfg.OutHiddenSize
	headDim := hidden / cfg.NumHeads
	qm := func(name string, rows, cols int) linalg.WeightMat {
		if err != nil {
			return linalg.WeightMat{}
		}
		if w, ok, aerr := aliasQMat(st, pfx+name, rows, cols, quant); aerr != nil || ok {
			err, aliased = aerr, aliased || ok
			return w
		}
		w := get(name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	e.patchW = get("patch_embed.proj.weight", hidden, cfg.InChannels, cfg.TemporalPatchSize, cfg.PatchSize, cfg.PatchSize)
	e.patchB = get("patch_embed.proj.bias", hidden)
	e.blocks = make([]glmOcrBlock, cfg.Depth)
	for i := range e.blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		b := &e.blocks[i]
		b.norm1w, b.norm2w = get(p+"norm1.weight", hidden), get(p+"norm2.weight", hidden)
		b.qkvw, b.qkvb = qm(p+"attn.qkv.weight", 3*hidden, hidden), get(p+"attn.qkv.bias", 3*hidden)
		b.projw, b.projb = qm(p+"attn.proj.weight", hidden, hidden), get(p+"attn.proj.bias", hidden)
		b.qNormW, b.kNormW = get(p+"attn.q_norm.weight", headDim), get(p+"attn.k_norm.weight", headDim)
		b.gatew, b.gateb = qm(p+"mlp.gate_proj.weight", inter, hidden), get(p+"mlp.gate_proj.bias", inter)
		b.upw, b.upb = qm(p+"mlp.up_proj.weight", inter, hidden), get(p+"mlp.up_proj.bias", inter)
		b.downw, b.downb = qm(p+"mlp.down_proj.weight", hidden, inter), get(p+"mlp.down_proj.bias", hidden)
	}
	e.postNormW = get("post_layernorm.weight", hidden)
	m := cfg.SpatialMergeSize
	// Conv2d weight [out, hidden, m, m] is already row-major (c, dy, dx) per output row: read it as
	// [out, hidden*m*m] — the shape check names the 4-D shape, the matmul wants the 2-D view.
	dw := get("downsample.weight", out, hidden, m, m)
	if err == nil {
		e.downW = newQMat(dw, out, hidden*m*m, quant)
	}
	e.downB = get("downsample.bias", out)
	inner := cfg.mergerInner()
	e.mProjW = qm("merger.proj.weight", out, out)
	e.mLNw, e.mLNb = get("merger.post_projection_norm.weight", out), get("merger.post_projection_norm.bias", out)
	e.mGateW = qm("merger.gate_proj.weight", inner, out)
	e.mUpW = qm("merger.up_proj.weight", inner, out)
	e.mDownW = qm("merger.down_proj.weight", out, inner)
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}

	// Rotary inv_freq over head_dim/2: 1/theta^(arange(0,dim,2)/dim), head_dim/4 frequencies.
	rdim := headDim / 2
	e.rotInvFreq = make([]float32, rdim/2)
	for i := range e.rotInvFreq {
		e.rotInvFreq[i] = float32(1.0 / math.Pow(glmOcrRotaryTheta, float64(2*i)/float64(rdim)))
	}
	e.rel = newBlockRelease(func() (*GlmOcrVisionEncoder, error) { return LoadGlmOcrVisionEncoder(dir, quant) })
	if aliased {
		e.maps.keep(st)
		kept = true
	}
	return e, nil
}

// glmOcrTensorPrefix is "model.visual." inside a GlmOcrForConditionalGeneration checkpoint (the tower
// shares the shards with the language model); "visual." is accepted for a stripped tower.
func glmOcrTensorPrefix(st *embed.SafetensorsFile) string {
	for _, pfx := range []string{"model.visual.", "visual."} {
		if _, err := st.Tensor(pfx + "patch_embed.proj.weight"); err == nil {
			return pfx
		}
	}
	return "model.visual."
}

// checkGrids validates the per-image grids and returns the total patch count.
func (e *GlmOcrVisionEncoder) checkGrids(pixelValues []float32, gridTHW [][3]int) (int, error) {
	c := e.Cfg
	merge := c.SpatialMergeSize
	patchDim := c.patchDim()
	nPatches := 0
	for _, g := range gridTHW {
		if g[0] <= 0 || g[1] <= 0 || g[2] <= 0 {
			return 0, fmt.Errorf("vision: non-positive grid dim in %v", g)
		}
		if g[1]%merge != 0 || g[2]%merge != 0 {
			return 0, fmt.Errorf("vision: grid %v h/w not divisible by spatial_merge_size %d", g, merge)
		}
		nPatches += g[0] * g[1] * g[2]
	}
	if nPatches == 0 {
		return 0, fmt.Errorf("vision: empty gridTHW (no images/patches)")
	}
	if len(pixelValues) != nPatches*patchDim {
		return 0, fmt.Errorf("vision: pixel_values len %d, want %d (%d patches × %d)", len(pixelValues), nPatches*patchDim, nPatches, patchDim)
	}
	return nPatches, nil
}

// Embed is the tower's first stage — the biased Conv3d patch embed as a matmul — returned
// [n_patches, hidden] in the original patch order. Exported for the stage-isolated parity gate.
func (e *GlmOcrVisionEncoder) Embed(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	nPatches, err := e.checkGrids(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	hidden := e.Cfg.HiddenSize
	h := make([]float32, nPatches*hidden)
	linalg.MatmulBT(pixelValues, e.patchW, h, nPatches, e.Cfg.patchDim(), hidden)
	addBias(h, e.patchB, nPatches, hidden)
	return h, nil
}

// Forward runs the ViT, post_layernorm, downsample and merger on pre-patchified pixel_values
// [n_patches, in_channels·temporal·patch²] (merge-block order) with per-image grids (t, h, w in patch
// units, h/w multiples of spatial_merge_size). It returns the merged image embeddings
// [n_merged, out_hidden_size] — HF's pooler_output, the rows that replace the decoder's <|image|>
// placeholders, n_merged = Σ t·h·w / merge².
func (e *GlmOcrVisionEncoder) Forward(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	ds, err := e.ForwardDownsampled(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	return e.merge(ds), nil
}

// ForwardDownsampled is HF's last_hidden_state: blocks, post_layernorm, then the downsample conv —
// [n_merged, out_hidden], i.e. everything before the merger.
func (e *GlmOcrVisionEncoder) ForwardDownsampled(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	h, err := e.ForwardViT(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	return e.downsample(h), nil
}

// ForwardViT runs embed + the transformer blocks + post_layernorm: the pre-downsample hidden state
// [n_patches, hidden] (the parity gate's pre-merge stage).
func (e *GlmOcrVisionEncoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	h, err := e.forwardBlocks(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	nPatches := len(h) / e.Cfg.HiddenSize
	post := make([]float32, len(h))
	rmsNormEpsInto(post, h, e.postNormW, nPatches, e.Cfg.HiddenSize, e.Cfg.RMSNormEps)
	return post, nil
}

// forwardBlocks runs embed + the transformer blocks: the last block's output [n_patches, hidden], before
// post_layernorm (what a device tower hands to FinishHidden).
func (e *GlmOcrVisionEncoder) forwardBlocks(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	if err := e.maps.check(); err != nil {
		return nil, err
	}
	if err := e.ensureBlocks(); err != nil {
		return nil, err
	}
	h, err := e.Embed(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	c := e.Cfg
	hidden, inter := c.HiddenSize, c.IntermediateSize
	nPatches := len(h) / hidden
	headDim := hidden / c.NumHeads
	cos, sin := e.rotaryCosSin(gridTHW, nPatches)
	cu := cuSeqlensFull(gridTHW)
	s := newQwenScratch(nPatches, hidden, inter, headDim, maxSegment(cu), c.NumHeads)
	att := s.att[:nPatches*hidden]
	o := s.o[:nPatches*hidden]
	mlpOut := s.mlpOut[:nPatches*hidden]
	n1 := s.n1[:nPatches*hidden]
	n2 := s.n2[:nPatches*hidden]
	eps := c.RMSNormEps
	for li := range e.blocks {
		b := &e.blocks[li]
		rmsNormEpsInto(n1, h, b.norm1w, nPatches, hidden, eps)
		if err := e.attentionInto(att, n1, b, nPatches, cos, sin, cu, s); err != nil {
			return nil, err
		}
		b.projw.MatmulBT(att, o, nPatches)
		addBias(o, b.projb, nPatches, hidden)
		addResidual(h, o)
		rmsNormEpsInto(n2, h, b.norm2w, nPatches, hidden, eps)
		gatedSiLUMLPInto(mlpOut, n2, b.gatew, b.gateb, b.upw, b.upb, b.downw, b.downb, nPatches, hidden, inter, s.gate, s.up)
		addResidual(h, mlpOut)
	}
	return h, nil
}

// attentionInto is the GLM-OCR attention body: biased fused qkv → split (seq, 3, heads, head_dim) →
// per-head q/k RMSNorm → rotary → full attention per cu_seqlens segment. The output projection,
// residual and norms stay with the caller.
func (e *GlmOcrVisionEncoder) attentionInto(out, x []float32, b *glmOcrBlock, seq int, cos, sin []float32, cu []int, s *qwenScratch) error {
	c := e.Cfg
	hidden, nH := c.HiddenSize, c.NumHeads
	hd := hidden / nH
	qkv := s.qkv[:seq*3*hidden]
	b.qkvw.MatmulBT(x, qkv, seq)
	addBias(qkv, b.qkvb, seq, 3*hidden)
	q := s.q[:seq*hidden]
	k := s.k[:seq*hidden]
	v := s.v[:seq*hidden]
	for i := range seq {
		base := i * 3 * hidden
		copy(q[i*hidden:(i+1)*hidden], qkv[base:base+hidden])
		copy(k[i*hidden:(i+1)*hidden], qkv[base+hidden:base+2*hidden])
		copy(v[i*hidden:(i+1)*hidden], qkv[base+2*hidden:base+3*hidden])
	}
	// Per-head RMSNorm BEFORE the rotary: q and k are [seq*nH, head_dim] row-major, so one call with
	// rows = seq*nH normalises every head of every patch independently.
	rmsNormEpsInto(q, q, b.qNormW, seq*nH, hd, c.RMSNormEps)
	rmsNormEpsInto(k, k, b.kNormW, seq*nH, hd, c.RMSNormEps)
	for i := range seq {
		co, si := cos[i*hd:i*hd+hd], sin[i*hd:i*hd+hd]
		for head := range nH {
			off := i*hidden + head*hd
			applyRotaryVision(q[off:off+hd], co, si)
			applyRotaryVision(k[off:off+hd], co, si)
		}
	}
	return attendPackedInto(out, q, k, v, hidden, nH, cu, s)
}

// gatedSiLUMLPInto computes down(silu(gate(x)) * up(x)) for x [rows, inDim] into dst [rows, outDim]:
// gate/up are [inner, inDim] with biases (nil = none), down is [outDim, inner]. gate/up are caller
// scratch of at least rows*inner.
func gatedSiLUMLPInto(dst, x []float32, gatew linalg.WeightMat, gateb []float32, upw linalg.WeightMat, upb []float32, downw linalg.WeightMat, downb []float32, rows, outDim, inner int, gateS, upS []float32) {
	gate := gateS[:rows*inner]
	gatew.MatmulBT(x, gate, rows)
	addBias(gate, gateb, rows, inner)
	up := upS[:rows*inner]
	upw.MatmulBT(x, up, rows)
	addBias(up, upb, rows, inner)
	parallelChunks(len(gate), func(lo, hi int) {
		g := gate[lo:hi]
		linalg.SiLUContractInto(g, g)
		u := up[lo:hi]
		for i := range g {
			g[i] *= u[i]
		}
	})
	downw.MatmulBT(gate, dst, rows)
	addBias(dst, downb, rows, outDim)
}

// rotaryCosSin builds per-patch cos/sin over the full head_dim (emb = cat(freqs, freqs)) from the
// (row, col) grid coordinates: freqs row = [row·inv_freq..., col·inv_freq...] (head_dim/2).
func (e *GlmOcrVisionEncoder) rotaryCosSin(gridTHW [][3]int, nPatches int) (cos, sin []float32) {
	headDim := e.Cfg.HiddenSize / e.Cfg.NumHeads
	nf := len(e.rotInvFreq)
	cos = make([]float32, nPatches*headDim)
	sin = make([]float32, nPatches*headDim)
	for i, p := range patchCoords(gridTHW, e.Cfg.SpatialMergeSize) {
		c, s := cos[i*headDim:(i+1)*headDim], sin[i*headDim:(i+1)*headDim]
		for k, f := range e.rotInvFreq {
			hf, wf := float64(float32(p.row)*f), float64(float32(p.col)*f)
			c[k], s[k] = float32(math.Cos(hf)), float32(math.Sin(hf))
			c[nf+k], s[nf+k] = float32(math.Cos(wf)), float32(math.Sin(wf))
		}
		copy(c[2*nf:], c[:2*nf])
		copy(s[2*nf:], s[:2*nf])
	}
	return cos, sin
}

// downsample is the Conv2d(hidden → out_hidden, kernel = stride = m): for each merge block of m²
// consecutive patches the input row is the patches' hidden vectors arranged channel-major,
// x[g][c·m² + j] = h[g·m² + j][c] (j = dy·m + dx) — HF's view(-1, m, m, C).permute(0, 3, 1, 2) —
// against the Conv2d weight flattened (c, dy, dx); then + bias.
func (e *GlmOcrVisionEncoder) downsample(h []float32) []float32 {
	c := e.Cfg
	hidden, out := c.HiddenSize, c.OutHiddenSize
	mu := c.SpatialMergeSize * c.SpatialMergeSize
	groups := len(h) / hidden / mu
	x := make([]float32, groups*hidden*mu)
	parallelRows(groups, groups*hidden*mu, func(start, end int) {
		for g := start; g < end; g++ {
			dst := x[g*hidden*mu : (g+1)*hidden*mu]
			for j := range mu {
				src := h[(g*mu+j)*hidden : (g*mu+j+1)*hidden]
				for ch, v := range src {
					dst[ch*mu+j] = v
				}
			}
		}
	})
	y := make([]float32, groups*out)
	e.downW.MatmulBT(x, y, groups)
	addBias(y, e.downB, groups, out)
	return y
}

// merge runs the patch merger on the downsampled rows [n_merged, out_hidden]: proj (no bias) →
// LayerNorm(weight, bias, eps 1e-5) → GELU(erf) → down(silu(gate) * up), all unbiased.
func (e *GlmOcrVisionEncoder) merge(x []float32) []float32 {
	c := e.Cfg
	out, inner := c.OutHiddenSize, c.mergerInner()
	rows := len(x) / out
	p := make([]float32, rows*out)
	e.mProjW.MatmulBT(x, p, rows)
	nrm := layerNorm(p, e.mLNw, e.mLNb, rows, out, glmOcrMergerLNEps)
	geluErf(nrm)
	res := make([]float32, rows*out)
	gate := make([]float32, rows*inner)
	up := make([]float32, rows*inner)
	gatedSiLUMLPInto(res, nrm, e.mGateW, nil, e.mUpW, nil, e.mDownW, nil, rows, out, inner, gate, up)
	return res
}

// rmsNormEpsInto is the weight-only RMSNorm with a caller-chosen eps (rmsNormRows hard-codes the Qwen
// towers' 1e-6): variance over the last dim in f64, no mean subtraction, y = w · float32(x · rsqrt).
// out may alias x: each row is read fully (sum of squares) before it is written.
func rmsNormEpsInto(out, x, w []float32, rows, dim int, eps float64) {
	parallelRows(rows, rows*dim, func(start, end int) {
		for r := start; r < end; r++ {
			xr := x[r*dim : r*dim+dim]
			var ss float64
			for _, v := range xr {
				ss += float64(v) * float64(v)
			}
			inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
			dst := out[r*dim : r*dim+dim]
			for d, v := range xr {
				dst[d] = float32(float64(v)*inv) * w[d]
			}
		}
	})
}

// Close releases the checkpoint mappings the tower's weights alias (aliasQMat); its forwards return an error afterward.
// Idempotent.
func (e *GlmOcrVisionEncoder) Close() {
	e.blocks = nil
	e.mProjW, e.mGateW, e.mUpW, e.mDownW = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
	e.maps.close()
}
