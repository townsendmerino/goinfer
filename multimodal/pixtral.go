package multimodal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/vision"
)

// Ministral 3's image path (Mistral3ForConditionalGeneration; S10 of docs/tasks/task-multimodal-support-2026-10.md):
// the preprocessing (HF's PixtralImageProcessor), the prompt block ([IMG]/[IMG_BREAK]/[IMG_END]) and the multimodal
// projector (Mistral3MultiModalProjector). The tower is aikit's vision.PixtralVisionEncoder.

// Pixtral's image special tokens.
const (
	PixtralImageToken = "[IMG]"
	PixtralImageBreak = "[IMG_BREAK]"
	PixtralImageEnd   = "[IMG_END]"
)

// PixtralPreprocessConfig is the processor's image settings.
type PixtralPreprocessConfig struct {
	PatchSize   int // the tower's patch (14)
	MergeSize   int // the projector's spatial merge (2)
	LongestEdge int // the resize bound (1540)
	Mean, Std   [3]float32
}

// PixtralDefaultPreprocess is Ministral 3's released processor_config.json.
func PixtralDefaultPreprocess() PixtralPreprocessConfig {
	return PixtralPreprocessConfig{PatchSize: 14, MergeSize: 2, LongestEdge: 1540,
		Mean: [3]float32{0.48145466, 0.4578275, 0.40821073}, Std: [3]float32{0.26862954, 0.26130258, 0.27577711}}
}

// LoadPixtralPreprocessConfig reads processor_config.json (or preprocessor_config.json) and config.json's
// spatial_merge_size from a checkpoint directory; a missing processor file keeps the defaults.
func LoadPixtralPreprocessConfig(dir string) (PixtralPreprocessConfig, error) {
	c := PixtralDefaultPreprocess()
	var top struct {
		SpatialMergeSize int `json:"spatial_merge_size"`
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		if err := json.Unmarshal(raw, &top); err != nil {
			return c, fmt.Errorf("pixtral: parse config.json: %w", err)
		}
		if top.SpatialMergeSize > 0 {
			c.MergeSize = top.SpatialMergeSize
		}
	}
	type ip struct {
		PatchSize int       `json:"patch_size"`
		Mean      []float32 `json:"image_mean"`
		Std       []float32 `json:"image_std"`
		Size      struct {
			LongestEdge int `json:"longest_edge"`
		} `json:"size"`
	}
	for _, name := range []string{"processor_config.json", "preprocessor_config.json"} {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		var w struct {
			ImageProcessor *ip `json:"image_processor"`
			ip
		}
		if err := json.Unmarshal(raw, &w); err != nil {
			return c, fmt.Errorf("pixtral: parse %s: %w", name, err)
		}
		p := w.ip
		if w.ImageProcessor != nil {
			p = *w.ImageProcessor
		}
		if p.PatchSize > 0 {
			c.PatchSize = p.PatchSize
		}
		if p.Size.LongestEdge > 0 {
			c.LongestEdge = p.Size.LongestEdge
		}
		if len(p.Mean) == 3 && len(p.Std) == 3 {
			copy(c.Mean[:], p.Mean)
			copy(c.Std[:], p.Std)
		}
		break
	}
	return c, nil
}

// PixtralResizeTarget is HF's resize rule: scale down so neither side exceeds LongestEdge (floor), then round each side UP
// to a multiple of patch*merge, so the image tiles into whole merge units (a small image is upscaled, never cropped).
func PixtralResizeTarget(h, w int, c PixtralPreprocessConfig) (int, int) {
	f := c.PatchSize * c.MergeSize
	ratio := math.Max(float64(h)/float64(c.LongestEdge), float64(w)/float64(c.LongestEdge))
	if ratio > 1 {
		h = int(math.Floor(float64(h) / ratio))
		w = int(math.Floor(float64(w) / ratio))
	}
	return ((h-1)/f + 1) * f, ((w-1)/f + 1) * f
}

// PixtralPreprocess decodes an image and returns it resized and normalised, channels-first [3, h, w], with h and w; its
// patch grid is (h/patch, w/patch) and its merged grid half that.
func PixtralPreprocess(data []byte, c PixtralPreprocessConfig) ([]float32, int, int, error) {
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("pixtral: decode image header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 || ic.Width > 16384 || ic.Height > 16384 {
		return nil, 0, 0, fmt.Errorf("pixtral: image %dx%d out of range", ic.Width, ic.Height)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("pixtral: decode image: %w", err)
	}
	h, w := ic.Height, ic.Width
	rgb := qwenExtractRGB(img, h, w)
	src := make([]uint8, len(rgb))
	for i, v := range rgb {
		src[i] = uint8(v)
	}
	th, tw := PixtralResizeTarget(h, w, c)
	res := vision.ResizeBicubicAA(src, h, w, th, tw) // torchvision's antialiased bicubic on uint8, as HF's fast processor
	out := make([]float32, 3*th*tw)
	for y := range th {
		for x := range tw {
			i := (y*tw + x) * 3
			for ch := range 3 {
				v := float32(res[i+ch]) * (1.0 / 255.0)
				out[ch*th*tw+y*tw+x] = (v - c.Mean[ch]) / c.Std[ch]
			}
		}
	}
	return out, th, tw, nil
}

// PixtralImageBlock is one image's prompt text for a merged grid of rows x cols: each merged row is cols [IMG] tokens
// followed by [IMG_BREAK], the last row's break being [IMG_END] (HF's PixtralProcessor). The image features replace the
// [IMG] tokens only; the breaks and the end are ordinary embeddings.
func PixtralImageBlock(rows, cols int) string {
	var b strings.Builder
	row := strings.Repeat(PixtralImageToken, cols)
	for r := range rows {
		b.WriteString(row)
		if r == rows-1 {
			b.WriteString(PixtralImageEnd)
		} else {
			b.WriteString(PixtralImageBreak)
		}
	}
	return b.String()
}

// PixtralProjector is Mistral3MultiModalProjector: RMSNorm over the tower's hidden, the 2x2 patch merger (channel-major
// unfold over each image's row-major patch grid, then a linear 4*hidden -> hidden), then linear_1, exact GELU, linear_2.
type PixtralProjector struct {
	visionHidden, textHidden, merge int
	eps                             float64
	normW                           []float32
	mergeW                          []float32 // [visionHidden, visionHidden*merge*merge]
	l1W                             []float32 // [textHidden, visionHidden]
	l2W                             []float32 // [textHidden, textHidden]
}

// TextHidden is the projector's output width (the language model's hidden size).
func (p *PixtralProjector) TextHidden() int { return p.textHidden }

// LoadPixtralProjector reads the projector from a Mistral3 checkpoint directory.
func LoadPixtralProjector(dir string) (*PixtralProjector, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("pixtral: read config: %w", err)
	}
	var c struct {
		SpatialMergeSize int    `json:"spatial_merge_size"`
		ProjectorAct     string `json:"projector_hidden_act"`
		ProjectorBias    bool   `json:"multimodal_projector_bias"`
		TextConfig       struct {
			HiddenSize int     `json:"hidden_size"`
			RMSNormEps float64 `json:"rms_norm_eps"`
		} `json:"text_config"`
		VisionConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("pixtral: parse config: %w", err)
	}
	if c.SpatialMergeSize <= 0 || c.TextConfig.HiddenSize <= 0 || c.VisionConfig.HiddenSize <= 0 {
		return nil, fmt.Errorf("pixtral: incomplete config (spatial_merge_size %d, text hidden %d, vision hidden %d)", c.SpatialMergeSize, c.TextConfig.HiddenSize, c.VisionConfig.HiddenSize)
	}
	if c.ProjectorBias {
		return nil, fmt.Errorf("pixtral: multimodal_projector_bias is not implemented (every released checkpoint has none)")
	}
	if c.ProjectorAct != "" && c.ProjectorAct != "gelu" {
		return nil, fmt.Errorf("pixtral: projector_hidden_act %q (only gelu)", c.ProjectorAct)
	}
	p := &PixtralProjector{visionHidden: c.VisionConfig.HiddenSize, textHidden: c.TextConfig.HiddenSize, merge: c.SpatialMergeSize, eps: c.TextConfig.RMSNormEps}
	if p.eps == 0 {
		p.eps = 1e-5
	}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("pixtral: open safetensors: %w", err)
	}
	defer st.Close()
	vh, th, mm := p.visionHidden, p.textHidden, p.merge*p.merge
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32("multi_modal_projector."+name, want...)
		return append([]float32(nil), v...)
	}
	p.normW = get("norm.weight", vh)
	p.mergeW = get("patch_merger.merging_layer.weight", vh, vh*mm)
	p.l1W = get("linear_1.weight", th, vh)
	p.l2W = get("linear_2.weight", th, th)
	if err != nil {
		return nil, fmt.Errorf("pixtral: load projector: %w", err)
	}
	return p, nil
}

// pixtralMergerPositionMajor, when set, lays each merge unit out position-major (the 4 patches' vectors one after the
// other) instead of HF's channel-major unfold: S10's planted defect for the merger.
var pixtralMergerPositionMajor bool

// Forward projects the tower's output ([Σ rows·cols, visionHidden], each image's patches row-major over grids[i] =
// (rows, cols) in patches) to [Σ (rows/merge)·(cols/merge), textHidden], each image's merged units row-major: the rows
// that replace its [IMG] tokens, in prompt order.
func (p *PixtralProjector) Forward(hidden []float32, grids [][2]int) ([]float32, error) {
	return p.forward(hidden, grids, nil)
}

// ForwardStages is Forward that also returns each stage for a parity gate (HF's module outputs): the norm's, the patch
// merger's, linear_1's (before the GELU) and linear_2's.
func (p *PixtralProjector) ForwardStages(hidden []float32, grids [][2]int) ([][]float32, error) {
	var st [][]float32
	_, err := p.forward(hidden, grids, func(x []float32) { st = append(st, append([]float32(nil), x...)) })
	return st, err
}

func (p *PixtralProjector) forward(hidden []float32, grids [][2]int, stage func([]float32)) ([]float32, error) {
	if stage == nil {
		stage = func([]float32) {}
	}
	vh, m := p.visionHidden, p.merge
	n, units := 0, 0
	for _, g := range grids {
		if g[0]%m != 0 || g[1]%m != 0 {
			return nil, fmt.Errorf("pixtral: patch grid %v is not a multiple of the merge %d", g, m)
		}
		n += g[0] * g[1]
		units += g[0] / m * g[1] / m
	}
	if len(hidden) != n*vh {
		return nil, fmt.Errorf("pixtral: projector input %d values, want %d (%d patches x %d)", len(hidden), n*vh, n, vh)
	}
	// RMSNorm (float64 variance, as the towers' own helpers).
	x := make([]float32, len(hidden))
	for r := range n {
		row := hidden[r*vh : (r+1)*vh]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(vh)+p.eps)
		for d, v := range row {
			x[r*vh+d] = float32(float64(v)*inv) * p.normW[d]
		}
	}
	stage(x)
	// The patch merger: unit (R, C) of an image takes the patches (m*R+ky, m*C+kx); HF's unfold lays the vector
	// channel-major: v[ch*m*m + ky*m + kx].
	mm := m * m
	unf := make([]float32, units*vh*mm)
	u, base := 0, 0
	for _, g := range grids {
		rows, cols := g[0], g[1]
		for R := range rows / m {
			for C := range cols / m {
				dst := unf[u*vh*mm : (u+1)*vh*mm]
				for ky := range m {
					for kx := range m {
						src := x[(base+(m*R+ky)*cols+m*C+kx)*vh:]
						k := ky*m + kx
						for ch := range vh {
							if pixtralMergerPositionMajor {
								dst[k*vh+ch] = src[ch]
							} else {
								dst[ch*mm+k] = src[ch]
							}
						}
					}
				}
				u++
			}
		}
		base += rows * cols
	}
	merged := make([]float32, units*vh)
	linalg.MatmulBT(unf, p.mergeW, merged, units, vh*mm, vh)
	stage(merged)
	h1 := make([]float32, units*p.textHidden)
	linalg.MatmulBT(merged, p.l1W, h1, units, vh, p.textHidden)
	stage(h1)
	for i, v := range h1 {
		h1[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	}
	out := make([]float32, units*p.textHidden)
	linalg.MatmulBT(h1, p.l2W, out, units, p.textHidden, p.textHidden)
	stage(out)
	return out, nil
}
