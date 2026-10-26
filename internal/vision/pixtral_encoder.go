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

// Pixtral's vision tower (Mistral 3 / Ministral 3's image encoder; HF PixtralVisionModel, transformers 5.15
// models/pixtral/modeling_pixtral.py). The language-model side — the multimodal projector (RMSNorm, the 2x2 patch
// merger, the GELU MLP), the [IMG]/[IMG_BREAK]/[IMG_END] layout and the splice — is the consumer's; this is the tower:
//
//   - patch_conv: Conv2d(channels -> hidden, kernel = stride = patch, no bias), as a matmul over patchified pixels;
//   - ln_pre: RMSNorm (eps 1e-5);
//   - num_hidden_layers pre-norm blocks: RMSNorm -> attention (separate q/k/v/o, no bias, 2-D rotate-half RoPE) ->
//     residual; RMSNorm -> SiLU-gated MLP (no bias) -> residual. No final norm: the output is the last block's.
//
// The 2-D RoPE (PixtralRotaryEmbedding): freqs = 1/theta^(arange(0, head_dim, 2)/head_dim) (head_dim/2 values); a patch
// at (row, col) rotates by [row * freqs[0::2] ‖ col * freqs[1::2]] (head_dim/4 each), doubled to head_dim. Rows take the
// EVEN-indexed frequencies and columns the ODD-indexed ones — not Qwen2.5-VL's tower, which uses one list for both.
//
// Several images run as one sequence with a block-diagonal mask (each image attends only within itself), every image's
// patches row-major over its own (rows, cols) patch grid.

// PixtralEncoderConfig is the vision_config of a Mistral3ForConditionalGeneration checkpoint.
type PixtralEncoderConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumChannels       int     `json:"num_channels"`
	PatchSize         int     `json:"patch_size"`
	ImageSize         int     `json:"image_size"`
	HeadDim           int     `json:"head_dim"`
	HiddenAct         string  `json:"hidden_act"`
	RopeTheta         float64 `json:"rope_theta"`
	RopeParameters    *struct {
		RopeTheta float64 `json:"rope_theta"`
		RopeType  string  `json:"rope_type"`
	} `json:"rope_parameters"`
}

func (c *PixtralEncoderConfig) validate() error {
	if c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 ||
		c.PatchSize <= 0 || c.ImageSize < c.PatchSize {
		return fmt.Errorf("pixtral: incomplete vision_config %+v", *c)
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.HeadDim*c.NumAttentionHeads != c.HiddenSize || c.HeadDim%4 != 0 {
		return fmt.Errorf("pixtral: head_dim %d x %d heads must be the hidden size %d, and head_dim a multiple of 4", c.HeadDim, c.NumAttentionHeads, c.HiddenSize)
	}
	if c.HiddenAct != "" && c.HiddenAct != "silu" {
		return fmt.Errorf("pixtral: hidden_act %q (only silu is implemented)", c.HiddenAct)
	}
	if c.RopeParameters != nil {
		if c.RopeParameters.RopeType != "" && c.RopeParameters.RopeType != "default" {
			return fmt.Errorf("pixtral: rope_type %q (only default is implemented)", c.RopeParameters.RopeType)
		}
		if c.RopeParameters.RopeTheta > 0 {
			c.RopeTheta = c.RopeParameters.RopeTheta
		}
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.NumChannels == 0 {
		c.NumChannels = 3
	}
	return nil
}

type pixtralBlock struct {
	attnNormW, ffnNormW []float32
	q, k, v, o          linalg.WeightMat // [hidden, hidden]
	gate, up            linalg.WeightMat // [inter, hidden]
	down                linalg.WeightMat // [hidden, inter]
}

// PixtralVisionEncoder is a loaded Pixtral vision tower.
type PixtralVisionEncoder struct {
	Cfg     PixtralEncoderConfig
	patchW  []float32 // [hidden, channels*patch*patch]
	lnPreW  []float32
	blocks  []pixtralBlock
	invFreq []float32 // head_dim/2 frequencies, float32 as HF computes them
	maps    towerMaps // the checkpoint mappings its aliased weights read (aliasQMat); Close releases them
}

// pixtralNormEps is the tower's RMSNorm eps (PixtralRMSNorm, eps=1e-5).
const pixtralNormEps = 1e-5

// LoadPixtralVisionEncoder reads the tower from a Mistral3 checkpoint directory (config.json's vision_config and the
// vision_tower.* tensors). quant wraps the projections as int8 W8A8; the parity gate runs quant=false.
func LoadPixtralVisionEncoder(dir string, quant bool) (*PixtralVisionEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("pixtral: read config: %w", err)
	}
	var wrap struct {
		PixtralEncoderConfig
		VisionConfig *PixtralEncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("pixtral: parse config: %w", err)
	}
	cfg := wrap.PixtralEncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("pixtral: open safetensors: %w", err)
	}
	aliased, kept := false, false // a weight aliases st; the loaded tower keeps st open
	defer func() {
		if !kept {
			st.Close()
		}
	}()
	pfx := pixtralTensorPrefix(st)
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
	hidden, inter, p := cfg.HiddenSize, cfg.IntermediateSize, cfg.PatchSize
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
	e := &PixtralVisionEncoder{Cfg: cfg}
	e.patchW = get("patch_conv.weight", hidden, cfg.NumChannels, p, p)
	e.lnPreW = get("ln_pre.weight", hidden)
	e.blocks = make([]pixtralBlock, cfg.NumHiddenLayers)
	for i := range e.blocks {
		l := fmt.Sprintf("transformer.layers.%d.", i)
		b := &e.blocks[i]
		b.attnNormW = get(l+"attention_norm.weight", hidden)
		b.q = qm(l+"attention.q_proj.weight", hidden, hidden)
		b.k = qm(l+"attention.k_proj.weight", hidden, hidden)
		b.v = qm(l+"attention.v_proj.weight", hidden, hidden)
		b.o = qm(l+"attention.o_proj.weight", hidden, hidden)
		b.ffnNormW = get(l+"ffn_norm.weight", hidden)
		b.gate = qm(l+"feed_forward.gate_proj.weight", inter, hidden)
		b.up = qm(l+"feed_forward.up_proj.weight", inter, hidden)
		b.down = qm(l+"feed_forward.down_proj.weight", hidden, inter)
	}
	if err != nil {
		return nil, fmt.Errorf("pixtral: load weights: %w", err)
	}
	e.invFreq = pixtralInvFreq(cfg.HeadDim, cfg.RopeTheta)
	if aliased {
		e.maps.keep(st)
		kept = true
	}
	return e, nil
}

// pixtralTensorPrefix is where the tower sits: "vision_tower." in a Mistral3ForConditionalGeneration save (and the tiny
// fixture), "model.vision_tower." in a save that nests the whole model.
func pixtralTensorPrefix(st *embed.SafetensorsFile) string {
	for _, pfx := range []string{"vision_tower.", "model.vision_tower."} {
		if _, err := st.Tensor(pfx + "patch_conv.weight"); err == nil {
			return pfx
		}
	}
	return "vision_tower."
}

// pixtralInvFreq is HF's freqs = 1/theta^(arange(0, dim, 2)/dim) in float32: head_dim/2 values.
func pixtralInvFreq(headDim int, theta float64) []float32 {
	f := make([]float32, headDim/2)
	for i := range f {
		ex := float32(2*i) / float32(headDim)
		f[i] = float32(1.0 / math.Pow(theta, float64(ex)))
	}
	return f
}

// pixtralColFreqOffset selects the column frequencies: 1 takes the odd-indexed ones, as HF does. A test seam (the planted
// defect "columns from the even list"); production leaves it at 1.
var pixtralColFreqOffset = 1

// pixtralSwapRowCol, when set, puts the column angles first: the planted defect "row and column halves swapped".
var pixtralSwapRowCol bool

// rotaryCosSin returns per-patch cos/sin over the full head_dim for every image's row-major patch grid.
func (e *PixtralVisionEncoder) rotaryCosSin(grids [][2]int, nPatches int) (cos, sin []float32) {
	hd := e.Cfg.HeadDim
	quarter := hd / 4
	cos, sin = make([]float32, nPatches*hd), make([]float32, nPatches*hd)
	ang := make([]float32, hd/2)
	i := 0
	for _, g := range grids {
		for r := range g[0] {
			for c := range g[1] {
				for j := range quarter {
					row := float32(r) * e.invFreq[2*j]
					col := float32(c) * e.invFreq[2*j+pixtralColFreqOffset]
					if pixtralSwapRowCol {
						row, col = col, row
					}
					ang[j], ang[quarter+j] = row, col
				}
				for d := range hd {
					a := float64(ang[d%(hd/2)]) // emb = cat(angles, angles)
					cos[i*hd+d] = float32(math.Cos(a))
					sin[i*hd+d] = float32(math.Sin(a))
				}
				i++
			}
		}
	}
	return cos, sin
}

// PixtralPatchify turns one image, channels-first [channels, h, w] (h and w multiples of patch), into its patches
// row-major over the (h/patch, w/patch) grid, each patch [channels, patch, patch] flattened as patch_conv's weight is.
func PixtralPatchify(img []float32, channels, h, w, patch int) ([]float32, [2]int, error) {
	if h%patch != 0 || w%patch != 0 || len(img) != channels*h*w {
		return nil, [2]int{}, fmt.Errorf("pixtral: image %dx%dx%d (len %d) is not patchable by %d", channels, h, w, len(img), patch)
	}
	rows, cols := h/patch, w/patch
	pd := channels * patch * patch
	out := make([]float32, rows*cols*pd)
	for r := range rows {
		for c := range cols {
			dst := out[(r*cols+c)*pd:]
			n := 0
			for ch := range channels {
				for dy := range patch {
					src := img[ch*h*w+(r*patch+dy)*w+c*patch:]
					copy(dst[n:n+patch], src[:patch])
					n += patch
				}
			}
		}
	}
	return out, [2]int{rows, cols}, nil
}

// pixtralNoBlockMask, when set, runs every image as one attention segment: the planted defect "the block mask dropped".
var pixtralNoBlockMask bool

// Forward runs the tower over every image's patches (PixtralPatchify's layout, images concatenated) and returns the
// last block's hidden state, [Σ rows·cols, hidden], in the same order: what the projector consumes.
func (e *PixtralVisionEncoder) Forward(patches []float32, grids [][2]int) ([]float32, error) {
	return e.forward(patches, grids, nil)
}

// ForwardStages is Forward that also returns each stage for a parity gate: the patch conv's output, ln_pre's, then each
// block's, in order (len = 2 + layers).
func (e *PixtralVisionEncoder) ForwardStages(patches []float32, grids [][2]int) ([][]float32, error) {
	var st [][]float32
	_, err := e.forward(patches, grids, func(x []float32) { st = append(st, append([]float32(nil), x...)) })
	return st, err
}

func (e *PixtralVisionEncoder) forward(patches []float32, grids [][2]int, stage func([]float32)) ([]float32, error) {
	if err := e.maps.check(); err != nil {
		return nil, err
	}
	c := e.Cfg
	hidden, nH, hd, inter := c.HiddenSize, c.NumAttentionHeads, c.HeadDim, c.IntermediateSize
	pd := c.NumChannels * c.PatchSize * c.PatchSize
	n := 0
	cu := []int{0}
	for _, g := range grids {
		if g[0] <= 0 || g[1] <= 0 {
			return nil, fmt.Errorf("pixtral: non-positive patch grid %v", g)
		}
		if g[0]*c.PatchSize > c.ImageSize || g[1]*c.PatchSize > c.ImageSize {
			return nil, fmt.Errorf("pixtral: patch grid %v is past the %d-pixel image size the RoPE table covers", g, c.ImageSize)
		}
		n += g[0] * g[1]
		cu = append(cu, n)
	}
	if n == 0 {
		return nil, fmt.Errorf("pixtral: no images")
	}
	if len(patches) != n*pd {
		return nil, fmt.Errorf("pixtral: patches len %d, want %d (%d patches x %d)", len(patches), n*pd, n, pd)
	}
	if pixtralNoBlockMask {
		cu = []int{0, n}
	}
	h := make([]float32, n*hidden)
	linalg.MatmulBT(patches, e.patchW, h, n, pd, hidden)
	if stage != nil {
		stage(h)
	}
	rmsNormEpsInto(h, h, e.lnPreW, n, hidden, pixtralNormEps)
	if stage != nil {
		stage(h)
	}
	cos, sin := e.rotaryCosSin(grids, n)
	s := newQwenScratch(n, hidden, inter, hd, maxSegment(cu), nH)
	nrm, att, o, mlp := s.n1[:n*hidden], s.att[:n*hidden], s.o[:n*hidden], s.mlpOut[:n*hidden]
	q, k, v := s.q[:n*hidden], s.k[:n*hidden], s.v[:n*hidden]
	for li := range e.blocks {
		b := &e.blocks[li]
		rmsNormEpsInto(nrm, h, b.attnNormW, n, hidden, pixtralNormEps)
		b.q.MatmulBT(nrm, q, n)
		b.k.MatmulBT(nrm, k, n)
		b.v.MatmulBT(nrm, v, n)
		for i := range n {
			co, si := cos[i*hd:i*hd+hd], sin[i*hd:i*hd+hd]
			for head := range nH {
				off := i*hidden + head*hd
				applyRotaryVision(q[off:off+hd], co, si)
				applyRotaryVision(k[off:off+hd], co, si)
			}
		}
		if err := attendPackedInto(att, q, k, v, hidden, nH, cu, s); err != nil {
			return nil, err
		}
		b.o.MatmulBT(att, o, n)
		addResidual(h, o)
		rmsNormEpsInto(nrm, h, b.ffnNormW, n, hidden, pixtralNormEps)
		gatedSiLUMLPInto(mlp, nrm, b.gate, nil, b.up, nil, b.down, nil, n, hidden, inter, s.gate, s.up)
		addResidual(h, mlp)
		if stage != nil {
			stage(h)
		}
	}
	return h, nil
}

// Close releases the checkpoint mappings the tower's weights alias (aliasQMat); its forwards return an error afterward.
// Idempotent.
func (e *PixtralVisionEncoder) Close() {
	e.blocks = nil
	e.maps.close()
}
