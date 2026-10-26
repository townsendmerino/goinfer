package vision

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/townsendmerino/aikit/linalg"
)

// Siglip2NaFlexEncoder is SigLIP2's NaFlex vision tower (transformers' Siglip2VisionModel, without its pooling head), the
// tower of LFM2-VL (goinfer S10). It differs from SigLIP (Encoder) in its input only: a tile of any patch grid, embedded
// by a linear layer over each patch's (row, column, channel) vector, plus a learned square position table resized to the
// tile's grid with antialiased bilinear interpolation. The blocks and the post-layernorm are SigLIP's (runBlocks). One
// tile per call: HF batches tiles with a key mask over the padding, which leaves each tile's rows exactly what it gives
// alone.
type Siglip2NaFlexEncoder struct {
	Cfg      EncoderConfig
	enc      *Encoder  // the blocks and post-layernorm, through SigLIP's own code
	patchW   []float32 // [hidden, C*P*P], the linear patch embedding
	patchB   []float32
	posTable []float32 // [side, side, hidden], channels last
	posSide  int
}

// Test seams for the parity gate's planted defects (siglip2_naflex_test.go); production never sets them.
var (
	naflexPlainBilinear   bool // the position table resized without antialias
	naflexChannelMajor    bool // patches flattened (channel, row, column)
	naflexNoPostLayerNorm bool // post_layernorm dropped
)

// LoadSiglip2NaFlexEncoder reads a SigLIP2 NaFlex tower from a checkpoint directory: a flat siglip2_vision_model
// config.json (the tiny fixture) or a VL model's vision_config, with the tower's tensors bare, under
// vision_tower.vision_model., or under model.vision_tower.vision_model. (an LFM2-VL save). quant weight-quantizes the
// blocks' matmuls as Encoder does; the patch embedding stays float32.
func LoadSiglip2NaFlexEncoder(dir string, quant bool) (*Siglip2NaFlexEncoder, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("vision: read config: %w", err)
	}
	type naflexCfg struct {
		EncoderConfig
		NumPatches int `json:"num_patches"`
	}
	var wrap struct {
		naflexCfg
		VisionConfig *naflexCfg `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &wrap); err != nil {
		return nil, fmt.Errorf("vision: parse config: %w", err)
	}
	nc := wrap.naflexCfg
	if wrap.VisionConfig != nil {
		nc = *wrap.VisionConfig
	}
	cfg := nc.EncoderConfig
	if cfg.LayerNormEps == 0 {
		cfg.LayerNormEps = 1e-6
	}
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 3
	}
	side := int(math.Round(math.Sqrt(float64(nc.NumPatches))))
	if nc.NumPatches <= 0 || side*side != nc.NumPatches {
		return nil, fmt.Errorf("vision: siglip2 num_patches %d is not a square", nc.NumPatches)
	}
	// NaFlex has no fixed image size; SigLIP's validation wants one, so give it the position table's.
	cfg.ImageSize = side * cfg.PatchSize
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
	r := &siglipReader{st: st, pfx: tensorPrefix(st, "embeddings.patch_embedding.weight", "vision_tower.vision_model.")}
	if _, err := st.Tensor("model.vision_tower.vision_model.embeddings.patch_embedding.weight"); err == nil {
		r.pfx = "model.vision_tower.vision_model."
	}
	hidden, cpp := cfg.HiddenSize, cfg.NumChannels*cfg.PatchSize*cfg.PatchSize
	e := &Siglip2NaFlexEncoder{Cfg: cfg, posSide: side}
	e.patchW = r.get("embeddings.patch_embedding.weight", hidden, cpp)
	e.patchB = r.get("embeddings.patch_embedding.bias", hidden)
	e.posTable = r.get("embeddings.position_embedding.weight", nc.NumPatches, hidden) // [side*side, hidden] = [side, side, hidden]
	enc := &Encoder{Cfg: cfg, dir: dir, quant: quant}
	enc.postLNw, enc.postLNb = r.get("post_layernorm.weight", hidden), r.get("post_layernorm.bias", hidden)
	if r.err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", r.err)
	}
	layers := make([]encLayer, cfg.NumHiddenLayers)
	qm := func(name string, rows, cols int) linalg.WeightMat {
		if r.err != nil {
			return linalg.WeightMat{}
		}
		if w, ok, err := aliasQMat(st, r.pfx+name, rows, cols, quant); err != nil || ok {
			if err != nil {
				r.err = err
			}
			aliased = aliased || ok
			return w
		}
		w := r.get(name, rows, cols)
		if r.err != nil {
			return linalg.WeightMat{}
		}
		return newQMat(w, rows, cols, quant)
	}
	inter := cfg.IntermediateSize
	for l := range layers {
		p := fmt.Sprintf("encoder.layers.%d.", l)
		lw := &layers[l]
		lw.ln1w, lw.ln1b = r.get(p+"layer_norm1.weight", hidden), r.get(p+"layer_norm1.bias", hidden)
		lw.ln2w, lw.ln2b = r.get(p+"layer_norm2.weight", hidden), r.get(p+"layer_norm2.bias", hidden)
		lw.qw, lw.qb = qm(p+"self_attn.q_proj.weight", hidden, hidden), r.get(p+"self_attn.q_proj.bias", hidden)
		lw.kw, lw.kb = qm(p+"self_attn.k_proj.weight", hidden, hidden), r.get(p+"self_attn.k_proj.bias", hidden)
		lw.vw, lw.vb = qm(p+"self_attn.v_proj.weight", hidden, hidden), r.get(p+"self_attn.v_proj.bias", hidden)
		lw.ow, lw.ob = qm(p+"self_attn.out_proj.weight", hidden, hidden), r.get(p+"self_attn.out_proj.bias", hidden)
		lw.fc1w, lw.fc1b = qm(p+"mlp.fc1.weight", inter, hidden), r.get(p+"mlp.fc1.bias", inter)
		lw.fc2w, lw.fc2b = qm(p+"mlp.fc2.weight", hidden, inter), r.get(p+"mlp.fc2.bias", hidden)
	}
	if r.err != nil {
		return nil, fmt.Errorf("vision: load weights: %w", r.err)
	}
	enc.layers = layers
	e.enc = enc
	if aliased {
		enc.maps.keep(st)
		kept = true
	}
	return e, nil
}

// PatchifyNaFlex cuts a channels-first image (c x h x w, h and w multiples of patch) into SigLIP2's patch rows: patches
// row-major over the (h/patch, w/patch) grid, each flattened (row, column, channel) with the channel innermost, as HF's
// convert_image_to_patches lays them out. It returns the rows and the grid.
func PatchifyNaFlex(img []float32, c, h, w, patch int) ([]float32, [2]int, error) {
	if patch <= 0 || h%patch != 0 || w%patch != 0 || len(img) != c*h*w {
		return nil, [2]int{}, fmt.Errorf("vision: naflex patchify: a %dx%dx%d image (%d values) does not tile into %d-pixel patches", c, h, w, len(img), patch)
	}
	gh, gw := h/patch, w/patch
	cpp := c * patch * patch
	out := make([]float32, gh*gw*cpp)
	for py := range gh {
		for px := range gw {
			row := out[(py*gw+px)*cpp:]
			for y := range patch {
				for x := range patch {
					for ch := range c {
						v := img[ch*h*w+(py*patch+y)*w+px*patch+x]
						if naflexChannelMajor {
							row[ch*patch*patch+y*patch+x] = v
						} else {
							row[(y*patch+x)*c+ch] = v
						}
					}
				}
			}
		}
	}
	return out, [2]int{gh, gw}, nil
}

// positions is the position table resized to grid (rows, cols): [rows*cols, hidden], row-major.
func (e *Siglip2NaFlexEncoder) positions(grid [2]int) []float32 {
	hidden := e.Cfg.HiddenSize
	if naflexPlainBilinear {
		return resizeBilinearPlainFloat(e.posTable, e.posSide, e.posSide, hidden, grid[0], grid[1])
	}
	return ResizeBilinearAAFloat(e.posTable, e.posSide, e.posSide, hidden, grid[0], grid[1])
}

// Forward runs one tile: patches [rows*cols, C*P*P] (PatchifyNaFlex's layout) over grid (rows, cols), to the
// post-layernormed hidden rows [rows*cols, hidden].
func (e *Siglip2NaFlexEncoder) Forward(patches []float32, grid [2]int) ([]float32, error) {
	return e.forward(patches, grid, nil)
}

// ForwardStages is Forward that also returns each stage for a parity gate: the embeddings (patch embedding plus
// positions), each block's output, then the post-layernorm's (len = 2 + layers).
func (e *Siglip2NaFlexEncoder) ForwardStages(patches []float32, grid [2]int) ([][]float32, error) {
	var st [][]float32
	_, err := e.forward(patches, grid, func(x []float32) { st = append(st, append([]float32(nil), x...)) })
	return st, err
}

func (e *Siglip2NaFlexEncoder) forward(patches []float32, grid [2]int, stage func([]float32)) ([]float32, error) {
	if err := e.enc.maps.check(); err != nil {
		return nil, err
	}
	c := e.Cfg
	hidden, cpp := c.HiddenSize, c.NumChannels*c.PatchSize*c.PatchSize
	n := grid[0] * grid[1]
	if n <= 0 || len(patches) != n*cpp {
		return nil, fmt.Errorf("vision: siglip2 tile %v with %d patch values, want %d", grid, len(patches), n*cpp)
	}
	h := make([]float32, n*hidden)
	linalg.MatmulBT(patches, e.patchW, h, n, cpp, hidden)
	addBias(h, e.patchB, n, hidden)
	addResidual(h, e.positions(grid))
	if stage != nil {
		stage(h)
	}
	if err := e.enc.runBlocks(h, n, stage); err != nil {
		return nil, err
	}
	if naflexNoPostLayerNorm {
		if stage != nil {
			stage(h)
		}
		return h, nil
	}
	out := layerNorm(h, e.enc.postLNw, e.enc.postLNb, n, hidden, c.LayerNormEps)
	if stage != nil {
		stage(out)
	}
	return out, nil
}

// resizeBilinearPlainFloat is torch's bilinear without antialias (align_corners false): two taps per axis at
// (i+0.5)·scale - 0.5, clamped at the edges. Only the planted defect uses it.
func resizeBilinearPlainFloat(src []float32, h, w, c, th, tw int) []float32 {
	idx := func(i, in, out int) (int, int, float64) {
		x := (float64(i)+0.5)*float64(in)/float64(out) - 0.5
		x = max(x, 0)
		x0 := min(int(x), in-1)
		x1 := min(x0+1, in-1)
		return x0, x1, x - float64(x0)
	}
	dst := make([]float32, th*tw*c)
	for y := range th {
		y0, y1, fy := idx(y, h, th)
		for x := range tw {
			x0, x1, fx := idx(x, w, tw)
			for k := range c {
				v := (1-fy)*((1-fx)*float64(src[(y0*w+x0)*c+k])+fx*float64(src[(y0*w+x1)*c+k])) +
					fy*((1-fx)*float64(src[(y1*w+x0)*c+k])+fx*float64(src[(y1*w+x1)*c+k]))
				dst[(y*tw+x)*c+k] = float32(v)
			}
		}
	}
	return dst
}

// Close releases the checkpoint mappings the tower's block weights alias (aliasQMat); its forwards return an error afterward.
// Idempotent.
func (e *Siglip2NaFlexEncoder) Close() { e.enc.Close() }
