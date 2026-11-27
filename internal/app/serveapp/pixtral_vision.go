package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/multimodal"
)

func (s *server) loadPixtralVisionTower(dir string) error {
	enc, proj, pp, err := towers.LoadPixtral(dir)
	if err != nil {
		return err
	}
	for _, lm := range s.models {
		if h := lm.model.Config().HiddenDim; h != proj.TextHidden() {
			enc.Close()
			return fmt.Errorf("pixtral: the projector emits %d-wide rows and %q's decoder takes %d", proj.TextHidden(), lm.name, h)
		}
		id, ok := lm.tk.TokenID(multimodal.PixtralImageToken)
		if !ok {
			enc.Close()
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.PixtralImageToken)
		}
		for _, t := range []string{multimodal.PixtralImageBreak, multimodal.PixtralImageEnd} {
			if _, ok := lm.tk.TokenID(t); !ok {
				enc.Close()
				return fmt.Errorf("vision: tokenizer has no %q token", t)
			}
		}
		lm.vision = &pixtralTower{Enc: enc, Proj: proj, PP: pp, ImgTok: id}
		fmt.Fprintf(os.Stderr, "loaded Pixtral vision tower for %q (%d layers, image token id %d, tower on CPU, float32) from %s\n", lm.name, enc.Cfg.NumHiddenLayers, id, dir)
	}
	return nil
}
