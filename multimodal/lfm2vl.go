package multimodal

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/aikit/vision"
)

// LFM2-VL's image path (Lfm2VlForConditionalGeneration; S10 of docs/tasks/task-multimodal-support-2026-10.md): the
// preprocessing (HF's Lfm2VlImageProcessorFast: a large image becomes a grid of 512-pixel tiles plus a thumbnail, a small
// one a single smart-resized tile), the prompt block and the multimodal projector (Lfm2VlMultiModalProjector). The tower
// is aikit's vision.Siglip2NaFlexEncoder, one tile per call.

// LFM2-VL's image special tokens.
const (
	Lfm2VLImageToken = "<image>"
	Lfm2VLImageStart = "<|image_start|>"
	Lfm2VLImageEnd   = "<|image_end|>"
	Lfm2VLThumbnail  = "<|img_thumbnail|>"
)

// Lfm2VLRowColToken is the marker before tile (row, col) of a multi-tile image, 0-based arguments.
func Lfm2VLRowColToken(row, col int) string {
	return fmt.Sprintf("<|img_row_%d_col_%d|>", row+1, col+1)
}

// Lfm2VLPreprocessConfig is the processor's image settings (processor_config.json's image_processor).
type Lfm2VLPreprocessConfig struct {
	TileSize           int       `json:"tile_size"`
	PatchSize          int       `json:"encoder_patch_size"`
	Downsample         int       `json:"downsample_factor"`
	MinImageTokens     int       `json:"min_image_tokens"`
	MaxImageTokens     int       `json:"max_image_tokens"`
	MinTiles           int       `json:"min_tiles"`
	MaxTiles           int       `json:"max_tiles"`
	UseThumbnail       bool      `json:"use_thumbnail"`
	DoImageSplitting   bool      `json:"do_image_splitting"`
	MaxPixelsTolerance float64   `json:"max_pixels_tolerance"`
	Mean               []float32 `json:"image_mean"`
	Std                []float32 `json:"image_std"`
}

// Lfm2VLDefaultPreprocess is LFM2.5-VL-1.6B's released processor_config.json.
func Lfm2VLDefaultPreprocess() Lfm2VLPreprocessConfig {
	return Lfm2VLPreprocessConfig{TileSize: 512, PatchSize: 16, Downsample: 2, MinImageTokens: 64, MaxImageTokens: 256,
		MinTiles: 2, MaxTiles: 10, UseThumbnail: true, DoImageSplitting: true, MaxPixelsTolerance: 2.0,
		Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}}
}

// LoadLfm2VLPreprocessConfig reads processor_config.json's image_processor from a checkpoint directory (the defaults
// when the file is absent), overlaying only the fields it sets.
func LoadLfm2VLPreprocessConfig(dir string) (Lfm2VLPreprocessConfig, error) {
	c := Lfm2VLDefaultPreprocess()
	raw, err := os.ReadFile(filepath.Join(dir, "processor_config.json"))
	if err != nil {
		return c, nil
	}
	var w struct {
		ImageProcessor json.RawMessage `json:"image_processor"`
	}
	if err := json.Unmarshal(raw, &w); err != nil {
		return c, fmt.Errorf("lfm2-vl: parse processor_config.json: %w", err)
	}
	if len(w.ImageProcessor) > 0 {
		if err := json.Unmarshal(w.ImageProcessor, &c); err != nil {
			return c, fmt.Errorf("lfm2-vl: parse processor_config.json image_processor: %w", err)
		}
	}
	if c.TileSize <= 0 || c.PatchSize <= 0 || c.Downsample <= 0 || c.TileSize%(c.PatchSize*c.Downsample) != 0 ||
		len(c.Mean) != 3 || len(c.Std) != 3 || c.MinImageTokens <= 0 || c.MaxImageTokens < c.MinImageTokens {
		return c, fmt.Errorf("lfm2-vl: unusable image processor settings %+v", c)
	}
	return c, nil
}

// lfm2VLRoundByFactor is HF's round_by_factor: Python's round, which rounds half to even.
func lfm2VLRoundByFactor(x float64, f int) int { return int(math.RoundToEven(x/float64(f))) * f }

// lfm2VLTooLarge is _is_image_too_large: the image rounded to the merge unit exceeds the token budget times the
// tolerance, so it is tiled.
func lfm2VLTooLarge(h, w int, c Lfm2VLPreprocessConfig) bool {
	f := c.PatchSize * c.Downsample
	hb := max(c.PatchSize, lfm2VLRoundByFactor(float64(h), f))
	wb := max(c.PatchSize, lfm2VLRoundByFactor(float64(w), f))
	return float64(hb*wb) > float64(c.MaxImageTokens*f*f)*c.MaxPixelsTolerance
}

// Lfm2VLSmartResize is HF's smart_resize: each side a multiple of patch*downsample, the token count between the minimum
// and maximum. It returns (height, width).
func Lfm2VLSmartResize(h, w int, c Lfm2VLPreprocessConfig) (int, int) {
	f := c.PatchSize * c.Downsample
	minPx, maxPx := float64(c.MinImageTokens*f*f), float64(c.MaxImageTokens*f*f)
	hb := max(f, lfm2VLRoundByFactor(float64(h), f))
	wb := max(f, lfm2VLRoundByFactor(float64(w), f))
	switch {
	case float64(hb*wb) > maxPx:
		beta := math.Sqrt(float64(h*w) / maxPx)
		hb = max(f, int(math.Floor(float64(h)/beta/float64(f)))*f)
		wb = max(f, int(math.Floor(float64(w)/beta/float64(f)))*f)
	case float64(hb*wb) < minPx:
		beta := math.Sqrt(minPx / float64(h*w))
		hb = int(math.Ceil(float64(h)*beta/float64(f))) * f
		wb = int(math.Ceil(float64(w)*beta/float64(f))) * f
	}
	return hb, wb
}

// lfm2VLTargetRatios is HF's _target_ratios: every (w, h) with minT <= w·h <= maxT, sorted by tile count. HF sorts a
// Python set, so the order within one count is CPython's; it never decides a layout. Two grids of one count can only tie
// at an aspect midway between theirs, and the square grid of that range (2x2, 3x3) is always nearer. What does decide a
// tie is the area rule in Lfm2VLGrid, which the 1200x1000 and 1000x1250 layout goldens exercise.
func lfm2VLTargetRatios(minT, maxT int) [][2]int {
	var r [][2]int
	for w := 1; w <= maxT; w++ {
		for h := 1; h <= maxT; h++ {
			if n := w * h; n >= minT && n <= maxT {
				r = append(r, [2]int{w, h})
			}
		}
	}
	sort.SliceStable(r, func(i, j int) bool { return r[i][0]*r[i][1] < r[j][0]*r[j][1] })
	return r
}

// lfm2VLNoAreaRule, when set, keeps the first of equally close grids (HF's area test dropped): S10's planted defect for
// the tie-break.
var lfm2VLNoAreaRule bool

// Lfm2VLGrid is _get_grid_layout: the tile grid (columns, rows) for an h x w image.
func Lfm2VLGrid(h, w int, c Lfm2VLPreprocessConfig) (cols, rows int) {
	aspect := float64(w) / float64(h)
	best, bestDiff := [2]int{1, 1}, math.Inf(1)
	area := float64(w * h)
	for _, r := range lfm2VLTargetRatios(c.MinTiles, c.MaxTiles) {
		d := math.Abs(aspect - float64(r[0])/float64(r[1]))
		if d < bestDiff {
			best, bestDiff = r, d
		} else if d == bestDiff && !lfm2VLNoAreaRule && area > 0.5*float64(c.TileSize*c.TileSize*r[0]*r[1]) {
			best = r
		}
	}
	return best[0], best[1]
}

// Lfm2VLTile is one tile (or the thumbnail, or a single-tile image): normalised channels-first pixels and its size.
type Lfm2VLTile struct {
	CHW  []float32
	H, W int
}

// Lfm2VLLayout is an image's preprocessing: its tiles in prompt order (row-major, then the thumbnail), the grid (1x1 for
// a single tile) and whether the last tile is a thumbnail.
type Lfm2VLLayout struct {
	Tiles      []Lfm2VLTile
	Rows, Cols int
	Thumbnail  bool
}

// Tokens is each tile's image-token count, in prompt order.
func (l Lfm2VLLayout) Tokens(c Lfm2VLPreprocessConfig) []int {
	f := c.PatchSize * c.Downsample
	n := make([]int, len(l.Tiles))
	for i, t := range l.Tiles {
		n[i] = (t.H / f) * (t.W / f)
	}
	return n
}

// Lfm2VLPreprocess decodes an image and lays it out as HF's processor does: resize_and_split, then rescale and
// normalise.
func Lfm2VLPreprocess(data []byte, c Lfm2VLPreprocessConfig) (Lfm2VLLayout, error) {
	ic, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return Lfm2VLLayout{}, fmt.Errorf("lfm2-vl: decode image header: %w", err)
	}
	if ic.Width <= 0 || ic.Height <= 0 || ic.Width > 16384 || ic.Height > 16384 {
		return Lfm2VLLayout{}, fmt.Errorf("lfm2-vl: image %dx%d out of range", ic.Width, ic.Height)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return Lfm2VLLayout{}, fmt.Errorf("lfm2-vl: decode image: %w", err)
	}
	h, w := ic.Height, ic.Width
	rgb := qwenExtractRGB(img, h, w)
	src := make([]uint8, len(rgb))
	for i, v := range rgb {
		src[i] = uint8(v)
	}
	return lfm2VLLayoutRGB(src, h, w, c), nil
}

// lfm2VLPlan is resize_and_split's decision for an h x w image, without the pixels: the grid (1x1 when not tiled),
// whether a thumbnail follows the tiles, and each tile's size (height, width) in prompt order.
func lfm2VLPlan(h, w int, c Lfm2VLPreprocessConfig) (rows, cols int, thumb bool, sizes [][2]int) {
	th, tw := Lfm2VLSmartResize(h, w, c)
	splitting := c.DoImageSplitting && !(c.MinTiles == 1 && c.MaxTiles == 1)
	if !(splitting && lfm2VLTooLarge(h, w, c)) {
		return 1, 1, false, [][2]int{{th, tw}}
	}
	cols, rows = Lfm2VLGrid(h, w, c)
	for range rows * cols {
		sizes = append(sizes, [2]int{c.TileSize, c.TileSize})
	}
	if c.UseThumbnail && rows*cols != 1 {
		thumb = true
		sizes = append(sizes, [2]int{th, tw})
	}
	return rows, cols, thumb, sizes
}

// lfm2VLLayoutRGB is Lfm2VLPreprocess on decoded 8-bit RGB (HWC).
func lfm2VLLayoutRGB(src []uint8, h, w int, c Lfm2VLPreprocessConfig) Lfm2VLLayout {
	rows, cols, thumb, sizes := lfm2VLPlan(h, w, c)
	l := Lfm2VLLayout{Rows: rows, Cols: cols, Thumbnail: thumb}
	if rows*cols == 1 {
		th, tw := sizes[0][0], sizes[0][1]
		l.Tiles = []Lfm2VLTile{lfm2VLNormalise(vision.ResizeBilinearAA(src, h, w, th, tw), th, tw, c)}
		return l
	}
	ts := c.TileSize
	big := vision.ResizeBilinearAA(src, h, w, ts*rows, ts*cols)
	tile := make([]uint8, ts*ts*3)
	for r := range rows {
		for col := range cols {
			for y := range ts {
				copy(tile[y*ts*3:(y+1)*ts*3], big[((r*ts+y)*ts*cols+col*ts)*3:((r*ts+y)*ts*cols+(col+1)*ts)*3])
			}
			l.Tiles = append(l.Tiles, lfm2VLNormalise(tile, ts, ts, c))
		}
	}
	if thumb {
		th, tw := sizes[len(sizes)-1][0], sizes[len(sizes)-1][1]
		l.Tiles = append(l.Tiles, lfm2VLNormalise(vision.ResizeBilinearAA(src, h, w, th, tw), th, tw, c))
	}
	return l
}

// lfm2VLNormalise is rescale (1/255) and normalise to channels-first float32.
func lfm2VLNormalise(rgb []uint8, h, w int, c Lfm2VLPreprocessConfig) Lfm2VLTile {
	out := make([]float32, 3*h*w)
	for y := range h {
		for x := range w {
			for ch := range 3 {
				v := float32(rgb[(y*w+x)*3+ch]) * (1.0 / 255.0)
				out[ch*h*w+y*w+x] = (v - c.Mean[ch]) / c.Std[ch]
			}
		}
	}
	return Lfm2VLTile{CHW: out, H: h, W: w}
}

// Lfm2VLImageBlock is one image's prompt text: <|image_start|>, each tile's marker and its <image> run (row-major), the
// thumbnail's marker and run, <|image_end|>. A single tile has no markers. The features replace the <image> tokens only.
func Lfm2VLImageBlock(l Lfm2VLLayout, c Lfm2VLPreprocessConfig) string {
	n := l.Tokens(c)
	var b strings.Builder
	b.WriteString(Lfm2VLImageStart)
	if l.Rows*l.Cols > 1 {
		k := 0
		for r := range l.Rows {
			for col := range l.Cols {
				b.WriteString(Lfm2VLRowColToken(r, col))
				b.WriteString(strings.Repeat(Lfm2VLImageToken, n[k]))
				k++
			}
		}
		if l.Thumbnail {
			b.WriteString(Lfm2VLThumbnail)
			b.WriteString(strings.Repeat(Lfm2VLImageToken, n[k]))
		}
	} else {
		b.WriteString(strings.Repeat(Lfm2VLImageToken, n[0]))
	}
	b.WriteString(Lfm2VLImageEnd)
	return b.String()
}

// Lfm2VLProjector is Lfm2VlMultiModalProjector: a pixel unshuffle by the downsample factor over a tile's patch grid,
// then linear_1 (bias), GELU (erf), linear_2 (bias). The released checkpoints have no LayerNorm.
type Lfm2VLProjector struct {
	visionHidden, textHidden, mid, factor int
	l1W, l1B, l2W, l2B                    []float32
}

// TextHidden is the projector's output width (the decoder's hidden size).
func (p *Lfm2VLProjector) TextHidden() int { return p.textHidden }

// lfm2VLUnshuffleChannelMajor, when set, lays each unit out channel-major (Pixtral's merger order) instead of HF's
// position-major pixel unshuffle: S10's planted defect for this projector.
var lfm2VLUnshuffleChannelMajor bool

// LoadLfm2VLProjector reads the projector from a checkpoint directory (config.json and the safetensors).
func LoadLfm2VLProjector(dir string) (*Lfm2VLProjector, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("lfm2-vl: read config: %w", err)
	}
	var c struct {
		Downsample   int    `json:"downsample_factor"`
		Hidden       int    `json:"projector_hidden_size"`
		Act          string `json:"projector_hidden_act"`
		Bias         *bool  `json:"projector_bias"`
		UseLayerNorm bool   `json:"projector_use_layernorm"`
		TextConfig   struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
		VisionConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("lfm2-vl: parse config: %w", err)
	}
	switch {
	case c.Downsample <= 0 || c.Hidden <= 0 || c.TextConfig.HiddenSize <= 0 || c.VisionConfig.HiddenSize <= 0:
		return nil, fmt.Errorf("lfm2-vl: incomplete projector config (downsample %d, hidden %d, text %d, vision %d)", c.Downsample, c.Hidden, c.TextConfig.HiddenSize, c.VisionConfig.HiddenSize)
	case c.UseLayerNorm:
		return nil, fmt.Errorf("lfm2-vl: projector_use_layernorm is not implemented (the released checkpoints have none)")
	case c.Bias != nil && !*c.Bias:
		return nil, fmt.Errorf("lfm2-vl: a bias-free projector is not implemented (the released checkpoints have biases)")
	case c.Act != "" && c.Act != "gelu":
		return nil, fmt.Errorf("lfm2-vl: projector_hidden_act %q (only gelu)", c.Act)
	}
	p := &Lfm2VLProjector{visionHidden: c.VisionConfig.HiddenSize, textHidden: c.TextConfig.HiddenSize, mid: c.Hidden, factor: c.Downsample}
	st, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("lfm2-vl: open safetensors: %w", err)
	}
	defer st.Close()
	pfx := "model.multi_modal_projector."
	if _, err := st.Tensor(pfx + "linear_1.weight"); err != nil {
		pfx = "multi_modal_projector."
	}
	in := p.visionHidden * p.factor * p.factor
	get := func(name string, want ...int) []float32 {
		if err != nil {
			return nil
		}
		var v []float32
		v, err = st.TensorF32(pfx+name, want...)
		return append([]float32(nil), v...)
	}
	p.l1W, p.l1B = get("linear_1.weight", p.mid, in), get("linear_1.bias", p.mid)
	p.l2W, p.l2B = get("linear_2.weight", p.textHidden, p.mid), get("linear_2.bias", p.textHidden)
	if err != nil {
		return nil, fmt.Errorf("lfm2-vl: load projector: %w", err)
	}
	return p, nil
}

// Forward projects one tile's tower output ([rows*cols, visionHidden] over its patch grid) to [rows/f*cols/f,
// textHidden], the units row-major: the rows that replace the tile's <image> tokens.
func (p *Lfm2VLProjector) Forward(hidden []float32, grid [2]int) ([]float32, error) {
	_, _, out, err := p.forwardStages(hidden, grid)
	return out, err
}

// forwardStages is Forward returning each stage for the parity gate (HF's module outputs): the pixel unshuffle's,
// linear_1's (before the GELU) and linear_2's.
func (p *Lfm2VLProjector) forwardStages(hidden []float32, grid [2]int) (unshuffled, linear1, out []float32, err error) {
	vh, f := p.visionHidden, p.factor
	rows, cols := grid[0], grid[1]
	if rows%f != 0 || cols%f != 0 {
		return nil, nil, nil, fmt.Errorf("lfm2-vl: patch grid %v is not a multiple of the downsample %d", grid, f)
	}
	if len(hidden) != rows*cols*vh {
		return nil, nil, nil, fmt.Errorf("lfm2-vl: projector input %d values, want %d (%dx%d x %d)", len(hidden), rows*cols*vh, rows, cols, vh)
	}
	ur, uc, ff := rows/f, cols/f, f*f
	units := ur * uc
	in := vh * ff
	unf := make([]float32, units*in)
	// HF's pixel_unshuffle: unit (R, C) takes patches (f*R+j, f*C+k), laid out [(j*f+k)*vh + channel].
	for R := range ur {
		for C := range uc {
			dst := unf[(R*uc+C)*in : (R*uc+C+1)*in]
			for j := range f {
				for k := range f {
					src := hidden[((f*R+j)*cols+f*C+k)*vh : ((f*R+j)*cols+f*C+k+1)*vh]
					pos := j*f + k
					if lfm2VLUnshuffleChannelMajor {
						for ch, v := range src {
							dst[ch*ff+pos] = v
						}
					} else {
						copy(dst[pos*vh:(pos+1)*vh], src)
					}
				}
			}
		}
	}
	h1 := make([]float32, units*p.mid)
	linalg.MatmulBT(unf, p.l1W, h1, units, in, p.mid)
	for u := range units {
		row := h1[u*p.mid : (u+1)*p.mid]
		for i := range row {
			row[i] += p.l1B[i]
		}
	}
	linear1 = append([]float32(nil), h1...)
	for i, x := range h1 {
		v := float64(x)
		h1[i] = float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2)))
	}
	out = make([]float32, units*p.textHidden)
	linalg.MatmulBT(h1, p.l2W, out, units, p.mid, p.textHidden)
	for u := range units {
		row := out[u*p.textHidden : (u+1)*p.textHidden]
		for i := range row {
			row[i] += p.l2B[i]
		}
	}
	return unf, linear1, out, nil
}
