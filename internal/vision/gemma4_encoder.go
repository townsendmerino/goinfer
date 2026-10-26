package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Gemma 4's vision tower (the "gemma4" plain family — E2B/E4B/26B-A4B/31B, NOT the
// separate encoder-free "gemma4_unified" 12B checkpoint). aikit's third ViT family,
// after SigLIP (encoder.go) and Qwen2.5-VL (qwen_encoder.go) — and architecturally
// closer to this codebase's own text decoders than to either: Gemma2/3-style
// SANDWICH RMSNorm (four independent norms per layer, not SigLIP's two LayerNorms),
// a GATED SwiGLU-tanh MLP (not SigLIP's plain 2-layer GELU), axial 2-D RoPE on Q/K
// (learned absolute position table ALSO applied, additively, at the patch-embed
// stage — both are real, neither is dead code), q/k/v all independently RMSNorm'd
// per head (v unscaled), a hardcoded attention scale of 1.0 (NOT head_dim**-0.5),
// and every attention/MLP projection wrapped in a real, checkpoint-supplied
// input/output clamp (Gemma4ClippableLinear) — confirmed load-bearing for the real
// E2B checkpoint (use_clipped_linears=true there, with genuine finite bounds like
// [-6.375,6.3125], not the harmless ±inf default), not something safe to skip.
//
// Pooling (3×3 average, "48px effective patch") runs ONCE, after all 16 layers —
// the encoder itself runs at full 16px-patch resolution. There is no separate
// final norm: the last layer's own post_feedforward_layernorm is the last op.
//
// Parity: cosine vs a real HF Gemma4VisionModel forward hook (P7 Phase A gate).
// Source citations throughout are to huggingface/transformers'
// modeling_gemma4.py, fetched 2026-09-09 (goinfer docs/multimodal.md's P7 entry).

// Gemma4EncoderConfig mirrors the HF Gemma4VisionConfig fields the forward needs.
type Gemma4EncoderConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	HeadDim           int     `json:"head_dim"`
	PatchSize         int     `json:"patch_size"`
	PoolingKernelSize int     `json:"pooling_kernel_size"`
	PosEmbTableSize   int     `json:"position_embedding_size"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	UseClippedLinears bool    `json:"use_clipped_linears"`
	// Standardize gates an extra (bias,scale) affine after pooling: false on the
	// real E2B checkpoint, true on the real 26B-A4B checkpoint (both confirmed
	// directly against real config.json — this is a real per-checkpoint split,
	// not an edge case). Gemma4VisionModel.forward applies it to the pooler's
	// already root-hidden_size-scaled output, before the embedder's RMSNorm+
	// projection: `(hidden_states - std_bias) * std_scale`, both [hidden_size]
	// (`model.vision_tower.std_bias`/`std_scale` in the real checkpoint's
	// safetensors header). LoadGemma4Encoder loads the two buffers only when
	// this is set.
	Standardize    bool `json:"standardize"`
	RopeParameters *struct {
		RopeTheta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

func (c Gemma4EncoderConfig) validate() error {
	switch {
	case c.HiddenSize <= 0:
		return fmt.Errorf("hidden_size must be > 0, got %d", c.HiddenSize)
	case c.IntermediateSize <= 0:
		return fmt.Errorf("intermediate_size must be > 0, got %d", c.IntermediateSize)
	case c.NumHiddenLayers < 0:
		return fmt.Errorf("num_hidden_layers must be >= 0, got %d", c.NumHiddenLayers)
	case c.NumAttentionHeads <= 0:
		return fmt.Errorf("num_attention_heads must be > 0, got %d", c.NumAttentionHeads)
	case c.HeadDim <= 0:
		return fmt.Errorf("head_dim must be > 0, got %d", c.HeadDim)
	case c.HeadDim%4 != 0:
		// Axial rope splits head_dim into 2 axis-chunks of head_dim/2, each further
		// split at head_dim/4 for the rotate-half pairing (gemma4RopeTables) — a
		// head_dim not divisible by 4 divides unevenly and silently drops channels.
		return fmt.Errorf("head_dim %d must be divisible by 4 (axial rope: two %d-wide rotate-half chunks)", c.HeadDim, c.HeadDim/2)
	case c.NumAttentionHeads*c.HeadDim != c.HiddenSize:
		return fmt.Errorf("num_attention_heads*head_dim (%d*%d=%d) must equal hidden_size %d — no GQA in this tower",
			c.NumAttentionHeads, c.HeadDim, c.NumAttentionHeads*c.HeadDim, c.HiddenSize)
	case c.PatchSize <= 0:
		return fmt.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	case c.PoolingKernelSize <= 0:
		return fmt.Errorf("pooling_kernel_size must be > 0, got %d", c.PoolingKernelSize)
	case c.PosEmbTableSize <= 0:
		return fmt.Errorf("position_embedding_size must be > 0, got %d", c.PosEmbTableSize)
	}
	return nil
}

// clippedProj is one Gemma4ClippableLinear projection: a bias-free matmul weight
// plus the real (checkpoint-supplied, per-tensor scalar) input/output clamp
// bounds HF applies around it. Bounds default to ±inf (a no-op clamp) when the
// checkpoint has use_clipped_linears=false — see loadClipped.
type clippedProj struct {
	w                            linalg.WeightMat
	inMin, inMax, outMin, outMax float32
}

func (p *clippedProj) matmulInto(ws *linalg.Workspace, x, dst []float32, rows int) {
	clampInto(x, p.inMin, p.inMax) // in place: HF clamps the shared activation tensor before each of q/k/v read it independently, so this must not mutate a value another projection still needs to read unclamped — callers pass a fresh copy per projection when x is shared (see attentionInto).
	p.w.MatmulBTInto(ws, x, dst, rows)
	clampInto(dst, p.outMin, p.outMax)
}

func clampInto(x []float32, lo, hi float32) {
	for i, v := range x {
		switch {
		case v < lo:
			x[i] = lo
		case v > hi:
			x[i] = hi
		}
	}
}

type gemma4EncLayer struct {
	inputNormW, postAttnNormW, preFFNNormW, postFFNNormW []float32 // RMSNorm weights, hidden-wide, no +1 offset
	qProj, kProj, vProj, oProj                           clippedProj
	qNormW, kNormW                                       []float32 // scaled RMSNorm, head_dim-wide (v_norm is unscaled — no weight)
	gateProj, upProj, downProj                           clippedProj
}

// Gemma4Encoder is a loaded Gemma 4 vision tower.
type Gemma4Encoder struct {
	Cfg            Gemma4EncoderConfig
	TextHiddenSize int       // text_config.hidden_size — the projector's output width
	patchEmbedW    []float32 // [hidden, 3*patch*patch], bias-free plain Linear
	posEmbX        []float32 // [PosEmbTableSize, hidden] — position_embedding_table[0]
	posEmbY        []float32 // [PosEmbTableSize, hidden] — position_embedding_table[1]
	layers         []gemma4EncLayer
	embedderProjW  linalg.WeightMat // [TextHiddenSize, hidden], plain Linear (no bias, no clip)
	stdBias        []float32        // [hidden]; nil unless Cfg.Standardize
	stdScale       []float32        // [hidden]; nil unless Cfg.Standardize
	rel            *blockRelease[Gemma4Encoder]
	maps           towerMaps // the checkpoint mappings its aliased weights read (aliasQMat); Close releases them
}

// LoadGemma4Encoder reads a Gemma 4 checkpoint (config.json + safetensors) and
// returns a ready vision tower. Weights are copied out, so the safetensors file
// is closed before return (no retained mmap).
func LoadGemma4Encoder(dir string, quant bool) (*Gemma4Encoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	var wrap struct {
		Gemma4EncoderConfig
		VisionConfig *Gemma4EncoderConfig `json:"vision_config"`
		TextConfig   *struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	cfg := wrap.Gemma4EncoderConfig
	if wrap.VisionConfig != nil {
		cfg = *wrap.VisionConfig
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	textHidden := 0
	if wrap.TextConfig != nil {
		textHidden = wrap.TextConfig.HiddenSize
	}
	if textHidden <= 0 {
		return nil, fmt.Errorf("vision: text_config.hidden_size missing or <= 0 (needed for the embed_vision projector's output width)")
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

	// Real checkpoints nest everything under "model." (confirmed against a real
	// gemma-4-E2B-it model.safetensors header); a stripped tiny fixture may not.
	pfx := tensorPrefix(st, "vision_tower.patch_embedder.input_proj.weight", "model.")

	e := &Gemma4Encoder{Cfg: cfg, TextHiddenSize: textHidden}
	hidden, inter, headDim := cfg.HiddenSize, cfg.IntermediateSize, cfg.HeadDim

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
	scalar := func(name string) float32 {
		v := get(name) // 0-dim tensor: no shape check, just want [1]
		if len(v) != 1 {
			if err == nil {
				err = fmt.Errorf("vision: scalar tensor %s has %d elements, want 1", name, len(v))
			}
			return 0
		}
		return v[0]
	}
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
	// clipped loads one Gemma4ClippableLinear: the wrapped Linear's weight plus,
	// only when use_clipped_linears is set, its four real scalar clamp bounds
	// (confirmed genuinely finite and load-bearing on the real E2B checkpoint —
	// NOT the harmless ±inf a disabled clamp would carry).
	clipped := func(base string, rows, cols int) clippedProj {
		p := clippedProj{w: qm(base+".linear.weight", rows, cols), inMin: float32(math.Inf(-1)), inMax: float32(math.Inf(1)), outMin: float32(math.Inf(-1)), outMax: float32(math.Inf(1))}
		if cfg.UseClippedLinears {
			p.inMin, p.inMax = scalar(base+".input_min"), scalar(base+".input_max")
			p.outMin, p.outMax = scalar(base+".output_min"), scalar(base+".output_max")
		}
		return p
	}

	e.patchEmbedW = get("vision_tower.patch_embedder.input_proj.weight", hidden, 3*cfg.PatchSize*cfg.PatchSize)
	posTable := get("vision_tower.patch_embedder.position_embedding_table", 2, cfg.PosEmbTableSize, hidden)
	if err == nil && len(posTable) == 2*cfg.PosEmbTableSize*hidden {
		half := cfg.PosEmbTableSize * hidden
		e.posEmbX = posTable[:half]
		e.posEmbY = posTable[half:]
	}

	e.layers = make([]gemma4EncLayer, cfg.NumHiddenLayers)
	for l := range e.layers {
		p := fmt.Sprintf("vision_tower.encoder.layers.%d.", l)
		lw := &e.layers[l]
		lw.inputNormW = get(p+"input_layernorm.weight", hidden)
		lw.postAttnNormW = get(p+"post_attention_layernorm.weight", hidden)
		lw.preFFNNormW = get(p+"pre_feedforward_layernorm.weight", hidden)
		lw.postFFNNormW = get(p+"post_feedforward_layernorm.weight", hidden)
		lw.qProj = clipped(p+"self_attn.q_proj", hidden, hidden)
		lw.kProj = clipped(p+"self_attn.k_proj", hidden, hidden)
		lw.vProj = clipped(p+"self_attn.v_proj", hidden, hidden)
		lw.oProj = clipped(p+"self_attn.o_proj", hidden, hidden)
		lw.qNormW = get(p+"self_attn.q_norm.weight", headDim)
		lw.kNormW = get(p+"self_attn.k_norm.weight", headDim)
		// v_norm is unscaled (with_scale=False) — no weight tensor exists in the
		// checkpoint (confirmed absent from the real E2B safetensors header).
		lw.gateProj = clipped(p+"mlp.gate_proj", inter, hidden)
		lw.upProj = clipped(p+"mlp.up_proj", inter, hidden)
		lw.downProj = clipped(p+"mlp.down_proj", hidden, inter)
	}
	e.embedderProjW = qm("embed_vision.embedding_projection.weight", textHidden, hidden)
	if cfg.Standardize {
		e.stdBias = get("vision_tower.std_bias", hidden)
		e.stdScale = get("vision_tower.std_scale", hidden)
	}
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}
	e.rel = newBlockRelease(func() (*Gemma4Encoder, error) { return LoadGemma4Encoder(dir, quant) })
	if aliased {
		e.maps.keep(st)
		kept = true
	}
	return e, nil
}

// Forward runs the vision tower on pre-unfolded patches [numPatches, 3*patch*patch]
// (the [0,1]-rescaled, NOT [-1,1] — that remap happens here, matching HF's
// Gemma4VisionPatchEmbedder.forward, not the preprocessor) with per-patch (x,y)
// grid coordinates, and returns the pooled+projected soft-token embeddings
// [numPooled, TextHiddenSize] in row-major pooled-grid order — what a caller
// splices into the text decoder's image-token positions.
func (e *Gemma4Encoder) Forward(patches []float32, positionIDs [][2]int) ([]float32, error) {
	if err := e.maps.check(); err != nil {
		return nil, err
	}
	if err := e.ensureBlocks(); err != nil {
		return nil, err
	}
	c := e.Cfg
	hidden, inter, nH, hd := c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.HeadDim
	np := len(positionIDs)
	patchDim := 3 * c.PatchSize * c.PatchSize
	if len(patches) != np*patchDim {
		return nil, fmt.Errorf("vision: patches len %d, want %d (%d patches × %d)", len(patches), np*patchDim, np, patchDim)
	}
	if np == 0 {
		return nil, fmt.Errorf("vision: no patches")
	}

	// Patch embed: [-1,1] rescale (model-side, not preprocessing — HF does
	// `2*(x-0.5)` inside Gemma4VisionPatchEmbedder.forward) then a plain matmul,
	// plus the learned additive x/y position table lookup.
	scaled := make([]float32, len(patches))
	for i, v := range patches {
		scaled[i] = 2 * (v - 0.5)
	}
	h := make([]float32, np*hidden)
	linalg.MatmulBT(scaled, e.patchEmbedW, h, np, patchDim, hidden)
	for i, pos := range positionIDs {
		x, y := pos[0], pos[1]
		if x < 0 || x >= c.PosEmbTableSize || y < 0 || y >= c.PosEmbTableSize {
			return nil, fmt.Errorf("vision: position id (%d,%d) out of range [0,%d)", x, y, c.PosEmbTableSize)
		}
		row := h[i*hidden : i*hidden+hidden]
		px, py := e.posEmbX[x*hidden:x*hidden+hidden], e.posEmbY[y*hidden:y*hidden+hidden]
		for d := range hidden {
			row[d] += px[d] + py[d]
		}
	}

	cos, sin := gemma4RopeTables(positionIDs, hd, e.ropeTheta())

	s := newGemma4EncScratch(np, hidden, inter, hd, nH)
	for l := range e.layers {
		lw := &e.layers[l]
		rmsNormWInto(s.n1, h, lw.inputNormW, np, hidden, c.RMSNormEps)
		if err := e.attentionInto(s.att, s.n1, lw, np, hd, nH, cos, sin, s); err != nil {
			return nil, err
		}
		rmsNormWInto(s.attNormed, s.att, lw.postAttnNormW, np, hidden, c.RMSNormEps)
		addResidual(h, s.attNormed)

		rmsNormWInto(s.n2, h, lw.preFFNNormW, np, hidden, c.RMSNormEps)
		e.mlpInto(s.mlpOut, s.n2, lw, np, s)
		rmsNormWInto(s.mlpNormed, s.mlpOut, lw.postFFNNormW, np, hidden, c.RMSNormEps)
		addResidual(h, s.mlpNormed)
	}
	return e.finish(h, positionIDs, np)
}

// finish is Forward's tail after the last layer (FinishHidden exports it).
func (e *Gemma4Encoder) finish(h []float32, positionIDs [][2]int, np int) ([]float32, error) {
	c := e.Cfg
	hidden := c.HiddenSize
	pooled, _, _, err := e.averagePool(h, positionIDs, np)
	if err != nil {
		return nil, err
	}
	sq := float32(math.Sqrt(float64(hidden)))
	for i := range pooled {
		pooled[i] *= sq
	}
	// standardize: (x - std_bias) * std_scale, applied to the pooler's already
	// root-hidden_size-scaled output — matches Gemma4VisionModel.forward's own
	// placement exactly (after pooling, before the embedder's RMSNorm+projection
	// below). Both buffers are [hidden], broadcast per pooled row.
	if c.Standardize {
		for i := range pooled {
			d := i % hidden
			pooled[i] = (pooled[i] - e.stdBias[d]) * e.stdScale[d]
		}
	}

	nPooled := len(pooled) / hidden
	normed := make([]float32, len(pooled))
	rmsNormUnscaledInto(normed, pooled, nPooled, hidden, c.RMSNormEps)
	out := make([]float32, nPooled*e.TextHiddenSize)
	e.embedderProjW.MatmulBT(normed, out, nPooled)
	return out, nil
}

// averagePool implements Gemma4VisionPooler's one-hot-bucket-matmul as a literal
// 3×3 (PoolingKernelSize²) bucket mean — mathematically identical for fully-valid
// interior windows (no padding patches in this Phase A single-image path).
// Requires the full patch grid's width/height (derived as max(x)+1,max(y)+1 over
// positionIDs) to each be divisible by PoolingKernelSize — true by construction
// when patches come from Gemma4AspectRatioSize's 48px-multiple resize.
func (e *Gemma4Encoder) averagePool(h []float32, positionIDs [][2]int, np int) (pooled []float32, poolW, poolH int, err error) {
	hidden, k := e.Cfg.HiddenSize, e.Cfg.PoolingKernelSize
	maxX, maxY := 0, 0
	for _, p := range positionIDs {
		if p[0] > maxX {
			maxX = p[0]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}
	gridW, gridH := maxX+1, maxY+1
	if gridW%k != 0 || gridH%k != 0 {
		return nil, 0, 0, fmt.Errorf("vision: patch grid %dx%d not divisible by pooling_kernel_size %d", gridW, gridH, k)
	}
	poolW, poolH = gridW/k, gridH/k
	nPooled := poolW * poolH
	sum := make([]float32, nPooled*hidden)
	count := make([]int, nPooled)
	for i, p := range positionIDs {
		bx, by := p[0]/k, p[1]/k
		bucket := by*poolW + bx
		count[bucket]++
		dst := sum[bucket*hidden : bucket*hidden+hidden]
		src := h[i*hidden : i*hidden+hidden]
		for d := range hidden {
			dst[d] += src[d]
		}
	}
	// Divide by the bucket's ACTUAL occupancy, not k*k (audit C-06). The
	// divisibility check above guarantees the grid splits into whole buckets; it
	// does NOT guarantee that positionIDs densely covers that grid. A sparse or
	// padded patch set leaves a bucket under-filled, and dividing it by k*k
	// scales it toward zero — a mean that is quietly wrong rather than an error.
	// Identical to the old constant whenever every bucket is full, which is the
	// only case the single-image path produces today.
	for b := range nPooled {
		if count[b] == 0 {
			continue // no patches landed here; leave the bucket at zero
		}
		inv := float32(1.0 / float64(count[b]))
		row := sum[b*hidden : b*hidden+hidden]
		for d := range row {
			row[d] *= inv
		}
	}
	return sum, poolW, poolH, nil
}

// gemma4RopeTables builds per-patch axial 2-D rope cos/sin, length headDim each.
// Channels [0,headDim/2) rotate on x using `sub=headDim/4` shared frequencies
// (duplicated across the chunk, standard rotate-half doubling); channels
// [headDim/2,headDim) rotate on y using the SAME frequency set. Exact formula:
// modeling_gemma4.py's compute_axial_rope_parameters + recomposition_frequencies
// (cat([freq_h,freq_h,freq_w,freq_w])).
func gemma4RopeTables(positionIDs [][2]int, headDim int, theta float64) (cos, sin []float32) {
	half := headDim / 2
	sub := half / 2
	invFreq := make([]float32, sub)
	for i := range invFreq {
		invFreq[i] = float32(1.0 / math.Pow(theta, float64(2*i)/float64(half)))
	}
	np := len(positionIDs)
	cos = make([]float32, np*headDim)
	sin = make([]float32, np*headDim)
	for i, p := range positionIDs {
		x, y := float32(p[0]), float32(p[1])
		base := i * headDim
		for j, f := range invFreq {
			ax, ay := x*f, y*f
			cx, sx := float32(math.Cos(float64(ax))), float32(math.Sin(float64(ax)))
			cy, sy := float32(math.Cos(float64(ay))), float32(math.Sin(float64(ay)))
			cos[base+j], cos[base+sub+j] = cx, cx
			sin[base+j], sin[base+sub+j] = sx, sx
			cos[base+half+j], cos[base+half+sub+j] = cy, cy
			sin[base+half+j], sin[base+half+sub+j] = sy, sy
		}
	}
	return cos, sin
}

// gemma4EncScratch is the Gemma 4 vision tower's per-Forward arena (mirrors
// encScratch/qwenScratch: one allocation per Forward, reused across all 16
// layers — the residual stream h is owned by the caller, everything else here
// is scratch).
type gemma4EncScratch struct {
	n1, att, attNormed, n2, mlpOut, mlpNormed []float32
	q, k, v, gate, up                         []float32
	ws                                        linalg.Workspace
	headPool                                  []encHeadScratch
	loFull, hiFull                            []int
}

func newGemma4EncScratch(np, hidden, inter, hd, nH int) *gemma4EncScratch {
	workers := max(min(runtime.GOMAXPROCS(0), nH), 1)
	pool := make([]encHeadScratch, workers)
	for i := range pool {
		pool[i] = newEncHeadScratch(np, hd)
	}
	loFull, hiFull := make([]int, np), make([]int, np)
	for i := range hiFull {
		hiFull[i] = np - 1
	}
	return &gemma4EncScratch{
		n1: make([]float32, np*hidden), att: make([]float32, np*hidden),
		attNormed: make([]float32, np*hidden), n2: make([]float32, np*hidden),
		mlpOut: make([]float32, np*hidden), mlpNormed: make([]float32, np*hidden),
		q: make([]float32, np*hidden), k: make([]float32, np*hidden), v: make([]float32, np*hidden),
		gate: make([]float32, np*inter), up: make([]float32, np*inter),
		headPool: pool, loFull: loFull, hiFull: hiFull,
	}
}

// attentionInto: full bidirectional multi-head attention, scale hardcoded to 1.0
// (NOT head_dim**-0.5 — confirmed at modeling_gemma4.py's Gemma4VisionAttention,
// self.scaling = 1.0). q/k/v each pass through their own ClippableLinear, then a
// per-head RMSNorm (q_norm/k_norm scaled, v_norm unscaled — v IS normed, easy to
// silently skip), then axial rope on q/k only (not v), then the same
// AttendTileFused fused schedule SigLIP/Qwen already use here (encHeadScratch is
// shared across all three towers in this package).
func (e *Gemma4Encoder) attentionInto(att, x []float32, lw *gemma4EncLayer, np, hd, nH int, cos, sin []float32, s *gemma4EncScratch) error {
	hidden := e.Cfg.HiddenSize
	const scale = 1.0 // Gemma4VisionAttention.scaling — hardcoded, not head_dim**-0.5

	qIn, kIn, vIn := append([]float32(nil), x...), append([]float32(nil), x...), append([]float32(nil), x...)
	lw.qProj.matmulInto(&s.ws, qIn, s.q, np)
	lw.kProj.matmulInto(&s.ws, kIn, s.k, np)
	lw.vProj.matmulInto(&s.ws, vIn, s.v, np)

	rmsNormHeadInto(s.q, lw.qNormW, np, hidden, nH, hd, e.Cfg.RMSNormEps)
	rmsNormHeadInto(s.k, lw.kNormW, np, hidden, nH, hd, e.Cfg.RMSNormEps)
	rmsNormHeadInto(s.v, nil, np, hidden, nH, hd, e.Cfg.RMSNormEps) // v_norm: unscaled

	half := hd / 2
	for i := range np {
		base := i * hd
		for head := range nH {
			off := i*hidden + head*hd
			applyRotaryVision(s.q[off:off+half], cos[base:base+half], sin[base:base+half])
			applyRotaryVision(s.q[off+half:off+hd], cos[base+half:base+hd], sin[base+half:base+hd])
			applyRotaryVision(s.k[off:off+half], cos[base:base+half], sin[base:base+half])
			applyRotaryVision(s.k[off+half:off+hd], cos[base+half:base+hd], sin[base+half:base+hd])
		}
	}

	attendHead := func(ws *encHeadScratch, mm func(a, b, dst []float32, M, K, N int), head int) error {
		off := head * hd
		linalg.GatherVBlockMajor(ws.kh, ws.vBlk, s.k, s.v, head, hd, hidden, np)
		for i := range np {
			copy(ws.qh[i*hd:(i+1)*hd], s.q[i*hidden+off:i*hidden+off+hd])
		}
		if !linalg.AttendTileFusedContractExp(mm, ws.qh, ws.kh, ws.vBlk, ws.ch, ws.fused, np, hd, np, scale, s.loFull, s.hiFull) {
			return fmt.Errorf("vision: AttendTileFused declined for np=%d hd=%d — scratch sizing invariant violated", np, hd)
		}
		for i := range np {
			copy(att[i*hidden+off:i*hidden+off+hd], ws.ch[i*hd:(i+1)*hd])
		}
		return nil
	}

	workers := len(s.headPool)
	if workers <= 1 {
		ws := &s.headPool[0]
		for head := range nH {
			if err := attendHead(ws, linalg.MatmulBT, head); err != nil {
				return err
			}
		}
	} else {
		var wg sync.WaitGroup
		errs := make([]error, workers)
		headsPer := (nH + workers - 1) / workers
		for w := range workers {
			h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
			if h0 >= h1 {
				continue
			}
			wg.Add(1)
			go func(w, h0, h1 int) {
				defer wg.Done()
				ws := &s.headPool[w]
				for head := h0; head < h1; head++ {
					if err := attendHead(ws, ws.mmWS.MatmulBT, head); err != nil {
						errs[w] = err
						return
					}
				}
			}(w, h0, h1)
		}
		wg.Wait()
		for _, err := range errs {
			if err != nil {
				return err
			}
		}
	}

	// o_proj on the concatenated attention output.
	oOut := make([]float32, np*hidden)
	lw.oProj.matmulInto(&s.ws, att, oOut, np)
	copy(att, oOut)
	return nil
}

// mlpInto: down(gelu_tanh(gate(x)) * up(x)) — gated, tanh-GELU (gelu_pytorch_tanh,
// this repo's already-proven cuda/glue.cu formula), each projection clamped by
// its own ClippableLinear bounds.
func (e *Gemma4Encoder) mlpInto(down, x []float32, lw *gemma4EncLayer, np int, s *gemma4EncScratch) {
	inter := e.Cfg.IntermediateSize
	gateIn, upIn := append([]float32(nil), x...), append([]float32(nil), x...)
	gate := s.gate[:np*inter]
	up := s.up[:np*inter]
	lw.gateProj.matmulInto(&s.ws, gateIn, gate, np)
	lw.upProj.matmulInto(&s.ws, upIn, up, np)
	geluTanh(gate)
	n := len(gate)
	if n > 0 {
		_ = up[n-1]
		i := 0
		for ; i+3 < n; i += 4 {
			gate[i+0] *= up[i+0]
			gate[i+1] *= up[i+1]
			gate[i+2] *= up[i+2]
			gate[i+3] *= up[i+3]
		}
		for ; i < n; i++ {
			gate[i] *= up[i]
		}
	}
	lw.downProj.matmulInto(&s.ws, gate, down, np)
}

// rmsNormHeadInto normalizes each head_dim-wide head slice of x independently
// (shared weight w across every head; w == nil for an unscaled norm — v_norm).
// In place.
// Row-split across cores (audit M-10): each row's per-head f64 sum-of-squares
// fold is row-local, so the split is bit-identical.
func rmsNormHeadInto(x []float32, w []float32, rows, hidden, nH, hd int, eps float64) {
	parallelRows(rows, rows*hidden, func(start, end int) {
		rmsNormHeadRows(x, w, start, end, hidden, nH, hd, eps)
	})
}

func rmsNormHeadRows(x []float32, w []float32, start, end, hidden, nH, hd int, eps float64) {
	if hd <= 0 {
		return
	}
	if w != nil {
		_ = w[hd-1]
	}
	hd4 := hd &^ 3
	for r := start; r < end; r++ {
		base := r * hidden
		for hIdx := range nH {
			off := base + hIdx*hd
			seg := x[off : off+hd]
			_ = seg[hd-1]
			var ss float64
			d := 0
			for ; d < hd4; d += 4 {
				v0 := float64(seg[d+0])
				v1 := float64(seg[d+1])
				v2 := float64(seg[d+2])
				v3 := float64(seg[d+3])
				ss += v0 * v0
				ss += v1 * v1
				ss += v2 * v2
				ss += v3 * v3
			}
			for ; d < hd; d++ {
				v := float64(seg[d])
				ss += v * v
			}
			inv := 1.0 / math.Sqrt(ss/float64(hd)+eps)
			d = 0
			if w != nil {
				for ; d < hd4; d += 4 {
					seg[d+0] = float32(float64(seg[d+0])*inv) * w[d+0]
					seg[d+1] = float32(float64(seg[d+1])*inv) * w[d+1]
					seg[d+2] = float32(float64(seg[d+2])*inv) * w[d+2]
					seg[d+3] = float32(float64(seg[d+3])*inv) * w[d+3]
				}
				for ; d < hd; d++ {
					seg[d] = float32(float64(seg[d])*inv) * w[d]
				}
			} else {
				for ; d < hd4; d += 4 {
					seg[d+0] = float32(float64(seg[d+0]) * inv)
					seg[d+1] = float32(float64(seg[d+1]) * inv)
					seg[d+2] = float32(float64(seg[d+2]) * inv)
					seg[d+3] = float32(float64(seg[d+3]) * inv)
				}
				for ; d < hd; d++ {
					seg[d] = float32(float64(seg[d]) * inv)
				}
			}
		}
	}
}

// rmsNormWInto is plain weight-only RMSNorm — Gemma4RMSNorm's `x_norm * weight`,
// NO `1+weight` offset (confirmed: modeling_gemma4.py:197-215). Writes into dst
// (reused per-layer scratch).
// Row-split across cores (audit M-10); bit-identical, the fold is row-local.
func rmsNormWInto(dst, x, w []float32, rows, dim int, eps float64) {
	parallelRows(rows, rows*dim, func(start, end int) {
		rmsNormWRows(dst, x, w, start, end, dim, eps)
	})
}

func rmsNormWRows(dst, x, w []float32, start, end, dim int, eps float64) {
	if dim <= 0 {
		return
	}
	_ = w[dim-1]
	dim4 := dim &^ 3
	for r := start; r < end; r++ {
		xr := x[r*dim : r*dim+dim]
		_ = xr[dim-1]
		var ss float64
		d := 0
		for ; d < dim4; d += 4 {
			v0 := float64(xr[d+0])
			v1 := float64(xr[d+1])
			v2 := float64(xr[d+2])
			v3 := float64(xr[d+3])
			ss += v0 * v0
			ss += v1 * v1
			ss += v2 * v2
			ss += v3 * v3
		}
		for ; d < dim; d++ {
			v := float64(xr[d])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
		dSlice := dst[r*dim : r*dim+dim]
		_ = dSlice[dim-1]
		d = 0
		for ; d < dim4; d += 4 {
			dSlice[d+0] = float32(float64(xr[d+0])*inv) * w[d+0]
			dSlice[d+1] = float32(float64(xr[d+1])*inv) * w[d+1]
			dSlice[d+2] = float32(float64(xr[d+2])*inv) * w[d+2]
			dSlice[d+3] = float32(float64(xr[d+3])*inv) * w[d+3]
		}
		for ; d < dim; d++ {
			dSlice[d] = float32(float64(xr[d])*inv) * w[d]
		}
	}
}

// rmsNormUnscaledInto is Gemma4RMSNorm(with_scale=False) — variance-normalize
// only, no weight multiply (the embed_vision projector's pre-projection norm).
// Row-split across cores (audit M-10); bit-identical, the fold is row-local.
func rmsNormUnscaledInto(dst, x []float32, rows, dim int, eps float64) {
	parallelRows(rows, rows*dim, func(start, end int) {
		rmsNormUnscaledRows(dst, x, start, end, dim, eps)
	})
}

func rmsNormUnscaledRows(dst, x []float32, start, end, dim int, eps float64) {
	if dim <= 0 {
		return
	}
	dim4 := dim &^ 3
	for r := start; r < end; r++ {
		xr := x[r*dim : r*dim+dim]
		_ = xr[dim-1]
		var ss float64
		d := 0
		for ; d < dim4; d += 4 {
			v0 := float64(xr[d+0])
			v1 := float64(xr[d+1])
			v2 := float64(xr[d+2])
			v3 := float64(xr[d+3])
			ss += v0 * v0
			ss += v1 * v1
			ss += v2 * v2
			ss += v3 * v3
		}
		for ; d < dim; d++ {
			v := float64(xr[d])
			ss += v * v
		}
		inv := 1.0 / math.Sqrt(ss/float64(dim)+eps)
		dSlice := dst[r*dim : r*dim+dim]
		_ = dSlice[dim-1]
		d = 0
		for ; d < dim4; d += 4 {
			dSlice[d+0] = float32(float64(xr[d+0]) * inv)
			dSlice[d+1] = float32(float64(xr[d+1]) * inv)
			dSlice[d+2] = float32(float64(xr[d+2]) * inv)
			dSlice[d+3] = float32(float64(xr[d+3]) * inv)
		}
		for ; d < dim; d++ {
			dSlice[d] = float32(float64(xr[d]) * inv)
		}
	}
}

// Close releases the checkpoint mappings the tower's weights alias (aliasQMat); its forwards return an error afterward.
// Idempotent.
func (e *Gemma4Encoder) Close() {
	e.layers = nil
	e.maps.close()
}
