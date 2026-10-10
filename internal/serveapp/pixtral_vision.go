package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// pixtralTower is Ministral 3's image path: aikit's Pixtral tower, the multimodal projector, the processor's settings and the
// [IMG] token id. The tower runs on the CPU in float32 (no backend declares a Pixtral device tower:
// multimodal.DeclaredTowers); the image tokens are causal, so the turn goes through decoder.GenerateVLCausalSpans, one span
// per merged row (docs/tasks/task-multimodal-support-2026-10.md).
type pixtralTower struct {
	enc    *vision.PixtralVisionEncoder
	proj   *multimodal.PixtralProjector
	pp     multimodal.PixtralPreprocessConfig
	imgTok int
}

// loadPixtralVisionTower attaches Ministral 3's tower to the single loaded model.
func (s *server) loadPixtralVisionTower(dir string) error {
	enc, err := vision.LoadPixtralVisionEncoder(dir, false)
	if err != nil {
		return fmt.Errorf("load pixtral vision tower (%s): %w", dir, err)
	}
	proj, err := multimodal.LoadPixtralProjector(dir)
	if err != nil {
		return fmt.Errorf("load mistral3 projector (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadPixtralPreprocessConfig(dir)
	if err != nil {
		return err
	}
	if pp.PatchSize != enc.Cfg.PatchSize {
		return fmt.Errorf("pixtral: the processor's patch %d is not the tower's %d", pp.PatchSize, enc.Cfg.PatchSize)
	}
	for _, lm := range s.models {
		if h := lm.model.Config().HiddenDim; h != proj.TextHidden() {
			return fmt.Errorf("pixtral: the projector emits %d-wide rows and %q's decoder takes %d", proj.TextHidden(), lm.name, h)
		}
		id, ok := lm.tk.TokenID(multimodal.PixtralImageToken)
		if !ok {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.PixtralImageToken)
		}
		for _, t := range []string{multimodal.PixtralImageBreak, multimodal.PixtralImageEnd} {
			if _, ok := lm.tk.TokenID(t); !ok {
				return fmt.Errorf("vision: tokenizer has no %q token", t)
			}
		}
		lm.pixtral = &pixtralTower{enc: enc, proj: proj, pp: pp, imgTok: id}
		fmt.Fprintf(os.Stderr, "loaded Pixtral vision tower for %q (%d layers, image token id %d, tower on CPU, float32) from %s\n", lm.name, enc.Cfg.NumHiddenLayers, id, dir)
	}
	return nil
}

// pixtralPrep prepares one image: its merged grid, one [IMG] run per merged row, and the lazy tower and projector.
func (lm *loadedModel) pixtralPrep(img imageRef) (imagePrep, error) {
	px := lm.pixtral
	chw, h, w, err := multimodal.PixtralPreprocess(img.data, px.pp)
	if err != nil {
		return imagePrep{}, err
	}
	patch := px.pp.PatchSize
	grid := [2]int{h / patch, w / patch}
	rows, cols := grid[0]/px.pp.MergeSize, grid[1]/px.pp.MergeSize
	hidden := lm.model.Config().HiddenDim
	p := imagePrep{n: rows * cols, runs: rows, runLen: cols, hash: multimodal.HashImageBytes(img.data), block: multimodal.PixtralImageBlock(rows, cols)}
	p.features = func() ([]float32, error) {
		patches, g, err := vision.PixtralPatchify(chw, 3, h, w, patch)
		if err != nil {
			return nil, err
		}
		hid, err := px.enc.Forward(patches, [][2]int{g})
		if err != nil {
			return nil, fmt.Errorf("pixtral vision tower: %w", err)
		}
		f, err := px.proj.Forward(hid, [][2]int{g})
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
