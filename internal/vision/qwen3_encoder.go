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

// Qwen3.5+ vision tower — aikit's third ViT family. It is the Qwen3-VL tower
// (Qwen3VLVisionModel) with DeepStack removed: Qwen3_5VisionModel deletes
// deepstack_visual_indexes / deepstack_merger_list, and every released Qwen3.5 / 3.6 / 3.8
// config carries deepstack_visual_indexes = []. Read from transformers 5.15.0's
// modeling_qwen3_5.py (self-contained, not just its modular file), and checked against the real
// Qwen3.5-0.8B / 9B / Qwen3.6-35B-A3B configs.
//
// What differs from the Qwen2.5-VL tower (qwen_encoder.go), each item transcribed rather than
// inferred:
//   - patch embed is a BIASED Conv3d (kernel = stride = [temporal, patch, patch]); as a matmul over
//     the flattened (channel, temporal, py, px) patch it is patchW·x + bias;
//   - a LEARNED position table pos_embed [G², hidden] (G = sqrt(num_position_embeddings), 48 in
//     every released config) is resampled to each image's (h, w) grid by bilinear interpolation with
//     align_corners=True and ADDED after the patch embed. Source coordinate = i·(G−1)/max(n−1, 1),
//     computed in f32; the four taps are weighted and summed in the order (h0w0, h0w1, h1w0, h1w1);
//   - 2D rotary as Qwen2.5-VL (theta 1e4 over head_dim/2, h then w, cat twice), but
//   - FULL attention over the whole image everywhere (no windows, no window reorder), fused biased
//     qkv (reshape (seq, 3, heads, head_dim)), biased proj;
//   - LayerNorm (weight AND bias), eps 1e-6, in place of RMSNorm;
//   - a NON-gated MLP: linear_fc1 → gelu_pytorch_tanh → linear_fc2, both biased;
//   - merger: LayerNorm(hidden) over each patch token, THEN the 2×2 merge as a contiguous view
//     (use_postshuffle_norm=False — the norm is before the view, over hidden, not merge²·hidden),
//     linear_fc1 → GELU **erf** (nn.GELU(), NOT the tanh form the block MLP uses) → linear_fc2.
//
// Patches keep their original (merge-block) order end to end, so unlike the Qwen2.5-VL tower there is
// no window permutation to undo. Forward takes pre-patchified pixel_values exactly as
// QwenVisionEncoder does. Added ADDITIVELY: QwenVisionEncoder and Encoder are untouched (the
// fused-attention body they used is shared via packedAttentionInto, moved verbatim).
//
// DeepStack (Qwen3-VL proper, deepstack_visual_indexes non-empty; goinfer S10): at each listed block a separate merger
// turns that block's output into one row per merged token. Its LayerNorm runs AFTER the merge-unit shuffle, over the
// hidden·merge² row (use_postshuffle_norm=True), unlike the main merger's. ForwardDeepstack returns those rows beside the
// main ones; the decoder adds the i-th set to the image positions' hidden state after its i-th layer. Forward still
// returns the main rows alone, so a caller that ignores DeepStack gets exactly what it got before.

// Qwen3EncoderConfig mirrors the HF Qwen3_5VisionConfig fields the forward needs.
type Qwen3EncoderConfig struct {
	Depth                  int    `json:"depth"`
	HiddenSize             int    `json:"hidden_size"`
	IntermediateSize       int    `json:"intermediate_size"`
	NumHeads               int    `json:"num_heads"`
	InChannels             int    `json:"in_channels"`
	PatchSize              int    `json:"patch_size"`
	SpatialMergeSize       int    `json:"spatial_merge_size"`
	TemporalPatchSize      int    `json:"temporal_patch_size"`
	OutHiddenSize          int    `json:"out_hidden_size"`
	NumPositionEmbeddings  int    `json:"num_position_embeddings"`
	HiddenAct              string `json:"hidden_act"`
	DeepstackVisualIndexes []int  `json:"deepstack_visual_indexes"`
}

const (
	qwen3LNEps       = 1e-6
	qwen3RotaryTheta = 10000.0 // Qwen3VLVisionRotaryEmbedding default theta
	qwen3ActBlockMLP = "gelu_pytorch_tanh"
)

func (c Qwen3EncoderConfig) validate() error {
	switch {
	case !deepstackIndexesOK(c.DeepstackVisualIndexes, c.Depth):
		return fmt.Errorf("deepstack_visual_indexes %v must be distinct, increasing block indexes below depth %d", c.DeepstackVisualIndexes, c.Depth)
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
	case c.NumPositionEmbeddings <= 0 || c.gridSide()*c.gridSide() != c.NumPositionEmbeddings:
		return fmt.Errorf("num_position_embeddings %d must be a positive perfect square (the learned table is a square grid)", c.NumPositionEmbeddings)
	case c.HiddenAct != qwen3ActBlockMLP:
		// blockMLP hardcodes gelu_pytorch_tanh; another declared activation would silently run
		// the wrong one (the same "known activation" check Qwen2.5-VL's validate makes).
		return fmt.Errorf("hidden_act %q unsupported (%s only)", c.HiddenAct, qwen3ActBlockMLP)
	}
	return nil
}

// deepstackIndexesOK: strictly increasing, each a block index (HF looks each block up in the list, so the order of the
// returned sets is the order of the blocks).
func deepstackIndexesOK(idx []int, depth int) bool {
	for i, v := range idx {
		if v < 0 || v >= depth || (i > 0 && v <= idx[i-1]) {
			return false
		}
	}
	return true
}

// gridSide is G = sqrt(num_position_embeddings), int(x**0.5) as HF computes it.
func (c Qwen3EncoderConfig) gridSide() int {
	g := int(math.Sqrt(float64(c.NumPositionEmbeddings)))
	for g*g > c.NumPositionEmbeddings {
		g--
	}
	for (g+1)*(g+1) <= c.NumPositionEmbeddings {
		g++
	}
	return g
}

type qwen3Block struct {
	norm1w, norm1b []float32
	qkvw           linalg.WeightMat // [3*hidden, hidden] fused
	qkvb           []float32
	projw          linalg.WeightMat // [hidden, hidden]
	projb          []float32
	norm2w, norm2b []float32
	fc1w           linalg.WeightMat // [inter, hidden]
	fc1b           []float32
	fc2w           linalg.WeightMat // [hidden, inter]
	fc2b           []float32
}

// Qwen3VisionEncoder is a loaded Qwen3.5+ vision tower (dynamic resolution, full attention).
type Qwen3VisionEncoder struct {
	Cfg        Qwen3EncoderConfig
	patchW     []float32 // [hidden, patch_dim] (Conv3d weight flattened, kept f32)
	patchB     []float32 // [hidden]
	posEmbed   []float32 // [G*G, hidden]
	blocks     []qwen3Block
	mergerLNw  []float32
	mergerLNb  []float32
	merger1w   linalg.WeightMat // fc1 [hidden*merge², hidden*merge²]
	merger1b   []float32
	merger2w   linalg.WeightMat // fc2 [out_hidden, hidden*merge²]
	merger2b   []float32
	rotInvFreq []float32                         // head_dim/4 rotary frequencies
	deepstack  []qwen3DSMerger                   // one per Cfg.DeepstackVisualIndexes entry (Qwen3-VL); empty for Qwen3.5+
	rel        *blockRelease[Qwen3VisionEncoder] // nil for a tower from a caller's TensorSource
	maps       towerMaps                         // the checkpoint mappings its aliased weights read (aliasQMat); Close releases them
	aliased    bool                              // LoadQwen3VisionEncoderFrom wrapped a weight in place (an ownedSource only)
}

// qwen3DSMerger is one DeepStack merger: LayerNorm over the merged hidden·merge² row (post-shuffle), fc1, GELU (erf),
// fc2, all biased.
type qwen3DSMerger struct {
	lnW, lnB []float32 // [hidden*merge²]
	fc1w     linalg.WeightMat
	fc1b     []float32
	fc2w     linalg.WeightMat
	fc2b     []float32
}

// LoadQwen3VisionEncoder reads a Qwen3.5+ checkpoint (config.json vision_config + safetensors) and
// returns a ready encoder. Weights are copied out, so the safetensors file is closed before return.
// quant wraps the projections as int8 W8A8 (patch embed, pos table and norms stay f32); the fp32
// parity gate runs quant=false.
func LoadQwen3VisionEncoder(dir string, quant bool) (*Qwen3VisionEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	var wrap struct {
		VisionConfig *Qwen3EncoderConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	if wrap.VisionConfig == nil {
		return nil, fmt.Errorf("vision: %s has no vision_config (a text-only checkpoint?)", filepath.Join(dir, "config.json"))
	}
	cfg := *wrap.VisionConfig
	if cfg.InChannels == 0 {
		cfg.InChannels = 3
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("vision: open safetensors: %w", err)
	}
	e, err := LoadQwen3VisionEncoderFrom(cfg, ownedSource{prefixedSource{st, qwen3TensorPrefix(st)}, st}, quant)
	if err != nil {
		st.Close()
		return nil, err
	}
	if e.aliased {
		e.maps.keep(st)
	} else {
		st.Close()
	}
	e.rel = newBlockRelease(func() (*Qwen3VisionEncoder, error) { return LoadQwen3VisionEncoder(dir, quant) })
	return e, nil
}

// TensorSource supplies a vision tower's tensors as float32 by their Hugging Face name (without a checkpoint's
// "model.visual." prefix) and shape, so a tower can load from a safetensors directory or from another container, such
// as a GGUF mmproj (LoadQwen3VisionEncoderMMProj). The returned slice may alias the source; loaders copy it.
type TensorSource interface {
	TensorF32(name string, want ...int) ([]float32, error)
}

// prefixedSource is a safetensors checkpoint's tensors under a name prefix.
type prefixedSource struct {
	st interface {
		TensorF32(name string, want ...int) ([]float32, error)
	}
	pfx string
}

func (p prefixedSource) TensorF32(name string, want ...int) ([]float32, error) {
	return p.st.TensorF32(p.pfx+name, want...)
}

// ownedSource is prefixedSource over a checkpoint the loading tower owns (LoadQwen3VisionEncoder). Only through it does
// LoadQwen3VisionEncoderFrom alias tensors in place (aliasQMat), since the tower then keeps the file open; a caller's
// TensorSource, even a safetensors file, is copied from.
type ownedSource struct {
	prefixedSource
	file *embed.SafetensorsFile
}

func (o ownedSource) Tensor(name string) (embed.Tensor, error) { return o.file.Tensor(o.pfx + name) }

// LoadQwen3VisionEncoderFrom builds a Qwen3.5+ tower from cfg and a tensor source (LoadQwen3VisionEncoder's body for
// any container). cfg is validated here.
func LoadQwen3VisionEncoderFrom(cfg Qwen3EncoderConfig, src TensorSource, quant bool) (*Qwen3VisionEncoder, error) {
	if cfg.InChannels == 0 {
		cfg.InChannels = 3
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("vision: %w", err)
	}
	var err error
	e := &Qwen3VisionEncoder{Cfg: cfg}
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = src.TensorF32(name, want...)
		if err != nil {
			return nil
		}
		return append([]float32(nil), v...)
	}
	hidden, inter := cfg.HiddenSize, cfg.IntermediateSize
	_, owned := src.(ownedSource)
	qm := func(name string, rows, cols int) linalg.WeightMat {
		if err != nil {
			return linalg.WeightMat{}
		}
		if owned {
			if w, ok, aerr := aliasQMat(src, name, rows, cols, quant); aerr != nil || ok {
				err, e.aliased = aerr, e.aliased || ok
				return w
			}
		}
		w := get(name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	e.patchW = get("patch_embed.proj.weight", hidden, cfg.InChannels, cfg.TemporalPatchSize, cfg.PatchSize, cfg.PatchSize)
	e.patchB = get("patch_embed.proj.bias", hidden)
	e.posEmbed = get("pos_embed.weight", cfg.NumPositionEmbeddings, hidden)
	e.blocks = make([]qwen3Block, cfg.Depth)
	for i := range e.blocks {
		p := fmt.Sprintf("blocks.%d.", i)
		b := &e.blocks[i]
		b.norm1w, b.norm1b = get(p+"norm1.weight", hidden), get(p+"norm1.bias", hidden)
		b.qkvw, b.qkvb = qm(p+"attn.qkv.weight", 3*hidden, hidden), get(p+"attn.qkv.bias", 3*hidden)
		b.projw, b.projb = qm(p+"attn.proj.weight", hidden, hidden), get(p+"attn.proj.bias", hidden)
		b.norm2w, b.norm2b = get(p+"norm2.weight", hidden), get(p+"norm2.bias", hidden)
		b.fc1w, b.fc1b = qm(p+"mlp.linear_fc1.weight", inter, hidden), get(p+"mlp.linear_fc1.bias", inter)
		b.fc2w, b.fc2b = qm(p+"mlp.linear_fc2.weight", hidden, inter), get(p+"mlp.linear_fc2.bias", hidden)
	}
	mh := hidden * cfg.SpatialMergeSize * cfg.SpatialMergeSize
	e.mergerLNw, e.mergerLNb = get("merger.norm.weight", hidden), get("merger.norm.bias", hidden)
	e.merger1w, e.merger1b = qm("merger.linear_fc1.weight", mh, mh), get("merger.linear_fc1.bias", mh)
	e.merger2w, e.merger2b = qm("merger.linear_fc2.weight", cfg.OutHiddenSize, mh), get("merger.linear_fc2.bias", cfg.OutHiddenSize)
	for i := range cfg.DeepstackVisualIndexes {
		p := fmt.Sprintf("deepstack_merger_list.%d.", i)
		e.deepstack = append(e.deepstack, qwen3DSMerger{
			lnW: get(p+"norm.weight", mh), lnB: get(p+"norm.bias", mh),
			fc1w: qm(p+"linear_fc1.weight", mh, mh), fc1b: get(p+"linear_fc1.bias", mh),
			fc2w: qm(p+"linear_fc2.weight", cfg.OutHiddenSize, mh), fc2b: get(p+"linear_fc2.bias", cfg.OutHiddenSize),
		})
	}
	if err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", err)
	}

	// Rotary inv_freq over head_dim/2: 1/theta^(arange(0,dim,2)/dim), head_dim/4 frequencies.
	headDim := hidden / cfg.NumHeads
	rdim := headDim / 2
	e.rotInvFreq = make([]float32, rdim/2)
	for i := range e.rotInvFreq {
		e.rotInvFreq[i] = float32(1.0 / math.Pow(qwen3RotaryTheta, float64(2*i)/float64(rdim)))
	}
	return e, nil
}

// qwen3TensorPrefix is "model.visual." inside a Qwen3_5ForConditionalGeneration checkpoint (the
// tower shares the shards with the language model); "visual." is accepted for a stripped tower.
func qwen3TensorPrefix(st *embed.SafetensorsFile) string {
	for _, pfx := range []string{"model.visual.", "visual."} {
		if _, err := st.Tensor(pfx + "patch_embed.proj.weight"); err == nil {
			return pfx
		}
	}
	return "model.visual."
}

// checkGrids validates the per-image grids and returns the total patch count.
func (e *Qwen3VisionEncoder) checkGrids(pixelValues []float32, gridTHW [][3]int) (int, error) {
	c := e.Cfg
	merge := c.SpatialMergeSize
	patchDim := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
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

// Embed is the tower's first stage — patch embed (+bias) plus the interpolated learned position
// embedding — returned [n_patches, hidden] in the original patch order. Exported for the
// stage-isolated parity gate (S1).
func (e *Qwen3VisionEncoder) Embed(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	nPatches, err := e.checkGrids(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	c := e.Cfg
	hidden := c.HiddenSize
	patchDim := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	h := make([]float32, nPatches*hidden)
	linalg.MatmulBT(pixelValues, e.patchW, h, nPatches, patchDim, hidden)
	addBias(h, e.patchB, nPatches, hidden)
	e.addPosEmbed(h, gridTHW)
	return h, nil
}

// patchPos is one patch's grid coordinates and its image's grid size.
type patchPos struct{ row, col, gh, gw int }

// patchCoords returns every patch's position in the tower's patch order: per image, per frame,
// spatial-merge-block order (block-row, block-col, in-row, in-col) — HF get_vision_position_ids.
func patchCoords(gridTHW [][3]int, merge int) []patchPos {
	var out []patchPos
	for _, g := range gridTHW {
		t, gh, gw := g[0], g[1], g[2]
		for range t {
			for br := 0; br < gh/merge; br++ {
				for bc := 0; bc < gw/merge; bc++ {
					for ir := range merge {
						for ic := range merge {
							out = append(out, patchPos{br*merge + ir, bc*merge + ic, gh, gw})
						}
					}
				}
			}
		}
	}
	return out
}

// addPosEmbed adds the bilinearly-resampled position embedding to h (in place), per patch.
// hidden_states = hidden_states + pos_embeds, with pos_embeds = (pos_embed(idx) * w[:, :, None]).sum(1).
func (e *Qwen3VisionEncoder) addPosEmbed(h []float32, gridTHW [][3]int) {
	c := e.Cfg
	hidden, side := c.HiddenSize, c.gridSide()
	var idx [4]int
	var wt [4]float32
	for i, p := range patchCoords(gridTHW, c.SpatialMergeSize) {
		interpTaps(p.row, p.gh, side, p.col, p.gw, &idx, &wt)
		row := h[i*hidden : i*hidden+hidden]
		// Sum over the 4 taps IN ORDER, each product rounded to f32 first, then accumulated: the
		// order torch's .sum(1) takes over a length-4 axis of the product tensor.
		for d := range hidden {
			v := e.posEmbed[idx[0]*hidden+d] * wt[0]
			v += float32(e.posEmbed[idx[1]*hidden+d] * wt[1])
			v += float32(e.posEmbed[idx[2]*hidden+d] * wt[2])
			v += float32(e.posEmbed[idx[3]*hidden+d] * wt[3])
			row[d] += v
		}
	}
}

// interpTaps fills the four (h0w0, h0w1, h1w0, h1w1) flat table indices and weights for one patch at
// (row, col) of an (gh × gw) grid resampled from a side×side table, align_corners=True bilinear —
// vision_utils._interpolation_axis_taps_weights, bilinear branch, all in f32:
//
//	src   = index * (side-1) / clamp(size-1, min=1)
//	tap0  = clamp(floor(src),   0, side-1)    w0 = clamp(1 - |src - floor(src)|,       min 0)
//	tap1  = clamp(floor(src)+1, 0, side-1)    w1 = clamp(1 - |src - (floor(src)+1)|,   min 0)
//
// then the 2D weights are the outer product (h weight × w weight) and the index is h_tap*side + w_tap.
func interpTaps(row, gh, side, col, gw int, idx *[4]int, wt *[4]float32) {
	var ht, wtp [2]int
	var hw, ww [2]float32
	axisTaps(row, gh, side, &ht, &hw)
	axisTaps(col, gw, side, &wtp, &ww)
	n := 0
	for a := range 2 {
		for b := range 2 {
			idx[n] = ht[a]*side + wtp[b]
			wt[n] = float32(hw[a] * ww[b])
			n++
		}
	}
}

func axisTaps(index, size, side int, taps *[2]int, w *[2]float32) {
	den := size - 1
	if den < 1 {
		den = 1
	}
	// f32 throughout, each step rounded: index * (side-1) then / den.
	src := float32(float32(index)*float32(side-1)) / float32(den)
	fl := float32(math.Floor(float64(src)))
	for o := range 2 {
		t := int(fl) + o
		if t < 0 {
			t = 0
		}
		if t > side-1 {
			t = side - 1
		}
		taps[o] = t
		d := float32(src - fl - float32(o))
		if d < 0 {
			d = -d
		}
		wv := float32(1) - d
		if wv < 0 {
			wv = 0
		}
		w[o] = wv
	}
}

// Forward runs the ViT + merger on pre-patchified pixel_values [n_patches, patch_dim] with per-image
// grids (t,h,w in patch units, h/w multiples of spatial_merge_size). It returns the merged image
// embeddings [n_merged, out_hidden_size] — the rows that replace the decoder's <image_pad>
// placeholders, n_merged = Σ t·h·w / merge².
func (e *Qwen3VisionEncoder) Forward(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	hid, err := e.ForwardViT(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	return e.merge(hid), nil
}

// ForwardViT runs embed + the transformer blocks (no merger): the pre-merge hidden state
// [n_patches, hidden], HF's last_hidden_state — the parity gate's S2 stage.
func (e *Qwen3VisionEncoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) ([]float32, error) {
	return e.forwardBlocks(pixelValues, gridTHW, nil)
}

// ForwardDeepstack is Forward for a DeepStack tower (Qwen3-VL): the main merged rows [n_merged, out_hidden] and, per
// Cfg.DeepstackVisualIndexes entry in order, that block's DeepStack rows [n_merged, out_hidden]. A tower without
// DeepStack returns no sets.
func (e *Qwen3VisionEncoder) ForwardDeepstack(pixelValues []float32, gridTHW [][3]int) (merged []float32, deep [][]float32, err error) {
	deep = make([][]float32, len(e.deepstack))
	h, err := e.forwardBlocks(pixelValues, gridTHW, func(li int, h []float32) {
		for k, idx := range e.Cfg.DeepstackVisualIndexes {
			if idx == li {
				deep[k] = e.deepMerge(k, h)
			}
		}
	})
	if err != nil {
		return nil, nil, err
	}
	return e.merge(h), deep, nil
}

// deepMerge is DeepStack merger k on a block's output h [n_patches, hidden]: the merge² consecutive patches viewed as
// one hidden·merge² row, LayerNorm over that row (post-shuffle), fc1 → GELU(erf) → fc2.
func (e *Qwen3VisionEncoder) deepMerge(k int, h []float32) []float32 {
	c, m := e.Cfg, &e.deepstack[k]
	mh := c.HiddenSize * c.SpatialMergeSize * c.SpatialMergeSize
	groups := len(h) / mh
	nrm := layerNorm(h, m.lnW, m.lnB, groups, mh, qwen3LNEps)
	mid := make([]float32, groups*mh)
	m.fc1w.MatmulBT(nrm, mid, groups)
	addBias(mid, m.fc1b, groups, mh)
	geluErf(mid)
	out := make([]float32, groups*c.OutHiddenSize)
	m.fc2w.MatmulBT(mid, out, groups)
	addBias(out, m.fc2b, groups, c.OutHiddenSize)
	return out
}

// forwardBlocks is ForwardViT's body; tap, when set, sees each block's output (li, h) right after the block, before the
// next one overwrites h.
func (e *Qwen3VisionEncoder) forwardBlocks(pixelValues []float32, gridTHW [][3]int, tap func(li int, h []float32)) ([]float32, error) {
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
	fc1 := s.gate[:nPatches*inter]
	for li := range e.blocks {
		b := &e.blocks[li]
		layerNormInto(n1, h, b.norm1w, b.norm1b, nPatches, hidden, qwen3LNEps)
		if err := packedAttentionInto(att, n1, b.qkvw, b.qkvb, hidden, c.NumHeads, nPatches, cos, sin, cu, s); err != nil {
			return nil, err
		}
		b.projw.MatmulBT(att, o, nPatches)
		addBias(o, b.projb, nPatches, hidden)
		addResidual(h, o)
		layerNormInto(n2, h, b.norm2w, b.norm2b, nPatches, hidden, qwen3LNEps)
		b.fc1w.MatmulBT(n2, fc1, nPatches)
		addBias(fc1, b.fc1b, nPatches, inter)
		geluTanh(fc1)
		b.fc2w.MatmulBT(fc1, mlpOut, nPatches)
		addBias(mlpOut, b.fc2b, nPatches, hidden)
		addResidual(h, mlpOut)
		if tap != nil {
			tap(li, h)
		}
	}
	return h, nil
}

// rotaryCosSin builds per-patch cos/sin over the full head_dim (emb = cat(freqs, freqs)) from the
// (h, w) grid coordinates.
func (e *Qwen3VisionEncoder) rotaryCosSin(gridTHW [][3]int, nPatches int) (cos, sin []float32) {
	headDim := e.Cfg.HiddenSize / e.Cfg.NumHeads
	nf := len(e.rotInvFreq)
	cos = make([]float32, nPatches*headDim)
	sin = make([]float32, nPatches*headDim)
	for i, p := range patchCoords(gridTHW, e.Cfg.SpatialMergeSize) {
		// freqs row = [h*inv_freq..., w*inv_freq...] (head_dim/2), emb = cat(row, row).
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

// merge runs the patch merger on the ViT hidden: LayerNorm over hidden per patch, the merge² patches
// of a block viewed as one hidden·merge² row (they are consecutive), fc1 → GELU(erf) → fc2.
func (e *Qwen3VisionEncoder) merge(hidden []float32) []float32 {
	c := e.Cfg
	H := c.HiddenSize
	mergeUnit := c.SpatialMergeSize * c.SpatialMergeSize
	mh := H * mergeUnit
	nPatches := len(hidden) / H
	groups := nPatches / mergeUnit
	nrm := layerNorm(hidden, e.mergerLNw, e.mergerLNb, nPatches, H, qwen3LNEps)
	mid := make([]float32, groups*mh)
	e.merger1w.MatmulBT(nrm, mid, groups)
	addBias(mid, e.merger1b, groups, mh)
	geluErf(mid)
	out := make([]float32, groups*c.OutHiddenSize)
	e.merger2w.MatmulBT(mid, out, groups)
	addBias(out, e.merger2b, groups, c.OutHiddenSize)
	return out
}

// Close releases the checkpoint mappings the tower's weights alias (aliasQMat); its forwards return an error afterward.
// Idempotent.
func (e *Qwen3VisionEncoder) Close() {
	e.blocks, e.merger1w, e.merger2w, e.deepstack = nil, linalg.WeightMat{}, linalg.WeightMat{}, nil
	e.maps.close()
}
