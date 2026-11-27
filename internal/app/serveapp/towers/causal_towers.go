package towers

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Lfm2VLTower holds LFM2-VL's SigLIP2 NaFlex encoder, projector, and preprocessing config.
type Lfm2VLTower struct {
	Enc    *vision.Siglip2NaFlexEncoder
	Proj   *multimodal.Lfm2VLProjector
	PP     multimodal.Lfm2VLPreprocessConfig
	ImgTok int
}

func (t *Lfm2VLTower) Family() string { return "LFM2-VL" }
func (t *Lfm2VLTower) Kind() string   { return "lfm2vl" }
func (t *Lfm2VLTower) ImageTok() int  { return t.ImgTok }

func (t *Lfm2VLTower) Close() {
	if t.Enc != nil {
		t.Enc.Close()
	}
}

func (t *Lfm2VLTower) Prep(hidden int, img media.ImageRef) (ImagePrep, error) {
	layout, err := multimodal.Lfm2VLPreprocess(img.Data, t.PP)
	if err != nil {
		return ImagePrep{}, err
	}
	runLens := layout.Tokens(t.PP)
	n := 0
	for _, k := range runLens {
		n += k
	}
	p := ImagePrep{N: n, RunLens: runLens, Hash: multimodal.HashImageBytes(img.Data), Block: multimodal.Lfm2VLImageBlock(layout, t.PP)}
	p.Features = func() ([]float32, error) {
		out := make([]float32, 0, n*hidden)
		for i, tile := range layout.Tiles {
			patches, grid, err := vision.PatchifyNaFlex(tile.CHW, 3, tile.H, tile.W, t.PP.PatchSize)
			if err != nil {
				return nil, err
			}
			hid, err := t.Enc.Forward(patches, grid)
			if err != nil {
				return nil, fmt.Errorf("LFM2-VL vision tower, tile %d of %d: %w", i+1, len(layout.Tiles), err)
			}
			f, err := t.Proj.Forward(hid, grid)
			if err != nil {
				return nil, err
			}
			if len(f) != runLens[i]*hidden {
				return nil, fmt.Errorf("LFM2-VL tile %d: the projector emitted %d values, want %d", i+1, len(f), runLens[i]*hidden)
			}
			out = append(out, f...)
		}
		return out, nil
	}
	return p, nil
}

// LoadLfm2VL loads LFM2-VL's encoder, projector, and preprocessing config from dir.
func LoadLfm2VL(dir string) (*vision.Siglip2NaFlexEncoder, *multimodal.Lfm2VLProjector, multimodal.Lfm2VLPreprocessConfig, error) {
	enc, err := vision.LoadSiglip2NaFlexEncoder(dir, false)
	if err != nil {
		return nil, nil, multimodal.Lfm2VLPreprocessConfig{}, fmt.Errorf("load LFM2-VL vision tower (%s): %w", dir, err)
	}
	proj, err := multimodal.LoadLfm2VLProjector(dir)
	if err != nil {
		enc.Close()
		return nil, nil, multimodal.Lfm2VLPreprocessConfig{}, fmt.Errorf("load LFM2-VL projector (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadLfm2VLPreprocessConfig(dir)
	if err != nil {
		enc.Close()
		return nil, nil, pp, err
	}
	if pp.PatchSize != enc.Cfg.PatchSize {
		enc.Close()
		return nil, nil, pp, fmt.Errorf("lfm2-vl: the processor's patch %d is not the tower's %d", pp.PatchSize, enc.Cfg.PatchSize)
	}
	return enc, proj, pp, nil
}

// PixtralTower holds Ministral 3's Pixtral encoder, projector, and preprocessing config.
type PixtralTower struct {
	Enc    *vision.PixtralVisionEncoder
	Proj   *multimodal.PixtralProjector
	PP     multimodal.PixtralPreprocessConfig
	ImgTok int
}

func (px *PixtralTower) Family() string { return "Pixtral" }
func (px *PixtralTower) Kind() string   { return "pixtral" }
func (px *PixtralTower) ImageTok() int  { return px.ImgTok }

func (px *PixtralTower) Close() {
	if px.Enc != nil {
		px.Enc.Close()
	}
}

func (px *PixtralTower) Prep(hidden int, img media.ImageRef) (ImagePrep, error) {
	chw, h, w, err := multimodal.PixtralPreprocess(img.Data, px.PP)
	if err != nil {
		return ImagePrep{}, err
	}
	patch := px.PP.PatchSize
	grid := [2]int{h / patch, w / patch}
	rows, cols := grid[0]/px.PP.MergeSize, grid[1]/px.PP.MergeSize
	runLens := make([]int, rows)
	for i := range runLens {
		runLens[i] = cols
	}
	p := ImagePrep{N: rows * cols, RunLens: runLens, Hash: multimodal.HashImageBytes(img.Data), Block: multimodal.PixtralImageBlock(rows, cols)}
	p.Features = func() ([]float32, error) {
		patches, g, err := vision.PixtralPatchify(chw, 3, h, w, patch)
		if err != nil {
			return nil, err
		}
		hid, err := px.Enc.Forward(patches, [][2]int{g})
		if err != nil {
			return nil, fmt.Errorf("pixtral vision tower: %w", err)
		}
		f, err := px.Proj.Forward(hid, [][2]int{g})
		if err != nil {
			return nil, err
		}
		if len(f) != rows*cols*hidden {
			return nil, fmt.Errorf("pixtral projector emitted %d values, want %d", len(f), rows*cols*hidden)
		}
		return f, nil
	}
	return p, nil
}

// LoadPixtral loads Ministral 3's Pixtral encoder, projector, and preprocessing config from dir.
func LoadPixtral(dir string) (*vision.PixtralVisionEncoder, *multimodal.PixtralProjector, multimodal.PixtralPreprocessConfig, error) {
	enc, err := vision.LoadPixtralVisionEncoder(dir, false)
	if err != nil {
		return nil, nil, multimodal.PixtralPreprocessConfig{}, fmt.Errorf("load pixtral vision tower (%s): %w", dir, err)
	}
	proj, err := multimodal.LoadPixtralProjector(dir)
	if err != nil {
		enc.Close()
		return nil, nil, multimodal.PixtralPreprocessConfig{}, fmt.Errorf("load mistral3 projector (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadPixtralPreprocessConfig(dir)
	if err != nil {
		enc.Close()
		return nil, nil, pp, err
	}
	if pp.PatchSize != enc.Cfg.PatchSize {
		enc.Close()
		return nil, nil, pp, fmt.Errorf("pixtral: the processor's patch %d is not the tower's %d", pp.PatchSize, enc.Cfg.PatchSize)
	}
	return enc, proj, pp, nil
}
