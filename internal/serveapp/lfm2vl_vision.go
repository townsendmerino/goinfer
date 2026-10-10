package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// lfm2vlTower is LFM2-VL's image path (S10, docs/tasks/task-multimodal-support-2026-10.md): aikit's SigLIP2 NaFlex tower,
// the projector, the processor's settings and the <image> token id. The tower runs on the CPU in float32, one tile at a
// time, and the decoder (lfm2) runs on the CPU only, so the turn goes through decoder.GenerateVLCausalSpans with one span
// per tile and thumbnail.
type lfm2vlTower struct {
	enc    *vision.Siglip2NaFlexEncoder
	proj   *multimodal.Lfm2VLProjector
	pp     multimodal.Lfm2VLPreprocessConfig
	imgTok int
}

// loadLfm2VLVisionTower attaches LFM2-VL's tower to the single loaded model.
func (s *server) loadLfm2VLVisionTower(dir string) error {
	enc, err := vision.LoadSiglip2NaFlexEncoder(dir, false)
	if err != nil {
		return fmt.Errorf("load LFM2-VL vision tower (%s): %w", dir, err)
	}
	proj, err := multimodal.LoadLfm2VLProjector(dir)
	if err != nil {
		return fmt.Errorf("load LFM2-VL projector (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadLfm2VLPreprocessConfig(dir)
	if err != nil {
		return err
	}
	if pp.PatchSize != enc.Cfg.PatchSize {
		return fmt.Errorf("lfm2-vl: the processor's patch %d is not the tower's %d", pp.PatchSize, enc.Cfg.PatchSize)
	}
	for _, lm := range s.models {
		if h := lm.model.Config().HiddenDim; h != proj.TextHidden() {
			return fmt.Errorf("lfm2-vl: the projector emits %d-wide rows and %q's decoder takes %d", proj.TextHidden(), lm.name, h)
		}
		id, ok := lm.tk.TokenID(multimodal.Lfm2VLImageToken)
		if !ok {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.Lfm2VLImageToken)
		}
		for _, t := range []string{multimodal.Lfm2VLImageStart, multimodal.Lfm2VLImageEnd, multimodal.Lfm2VLThumbnail, multimodal.Lfm2VLRowColToken(0, 0)} {
			if _, ok := lm.tk.TokenID(t); !ok {
				return fmt.Errorf("vision: tokenizer has no %q token", t)
			}
		}
		lm.lfm2vl = &lfm2vlTower{enc: enc, proj: proj, pp: pp, imgTok: id}
		fmt.Fprintf(os.Stderr, "loaded LFM2-VL vision tower for %q (SigLIP2 NaFlex, %d layers, image token id %d, tower on CPU, float32) from %s\n", lm.name, enc.Cfg.NumHiddenLayers, id, dir)
	}
	return nil
}

// lfm2vlPrep prepares one image: its tiles (row-major, then the thumbnail), one <image> run per tile, and the lazy tower
// and projector over each tile in turn.
func (lm *loadedModel) lfm2vlPrep(img imageRef) (imagePrep, error) {
	t := lm.lfm2vl
	layout, err := multimodal.Lfm2VLPreprocess(img.data, t.pp)
	if err != nil {
		return imagePrep{}, err
	}
	runLens := layout.Tokens(t.pp)
	n := 0
	for _, k := range runLens {
		n += k
	}
	hidden := lm.model.Config().HiddenDim
	p := imagePrep{n: n, runLens: runLens, hash: multimodal.HashImageBytes(img.data), block: multimodal.Lfm2VLImageBlock(layout, t.pp)}
	p.features = func() ([]float32, error) {
		out := make([]float32, 0, n*hidden)
		for i, tile := range layout.Tiles {
			patches, grid, err := vision.PatchifyNaFlex(tile.CHW, 3, tile.H, tile.W, t.pp.PatchSize)
			if err != nil {
				return nil, err
			}
			hid, err := t.enc.Forward(patches, grid)
			if err != nil {
				return nil, fmt.Errorf("LFM2-VL vision tower, tile %d of %d: %w", i+1, len(layout.Tiles), err)
			}
			f, err := t.proj.Forward(hid, grid)
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
