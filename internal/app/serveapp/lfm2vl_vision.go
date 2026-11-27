package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/multimodal"
)

func (s *server) loadLfm2VLVisionTower(dir string) error {
	enc, proj, pp, err := towers.LoadLfm2VL(dir)
	if err != nil {
		return err
	}
	for _, lm := range s.models {
		if h := lm.model.Config().HiddenDim; h != proj.TextHidden() {
			enc.Close()
			return fmt.Errorf("lfm2-vl: the projector emits %d-wide rows and %q's decoder takes %d", proj.TextHidden(), lm.name, h)
		}
		id, ok := lm.tk.TokenID(multimodal.Lfm2VLImageToken)
		if !ok {
			enc.Close()
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.Lfm2VLImageToken)
		}
		for _, t := range []string{multimodal.Lfm2VLImageStart, multimodal.Lfm2VLImageEnd, multimodal.Lfm2VLThumbnail, multimodal.Lfm2VLRowColToken(0, 0)} {
			if _, ok := lm.tk.TokenID(t); !ok {
				enc.Close()
				return fmt.Errorf("vision: tokenizer has no %q token", t)
			}
		}
		lm.vision = &lfm2vlTower{Enc: enc, Proj: proj, PP: pp, ImgTok: id}
		fmt.Fprintf(os.Stderr, "loaded LFM2-VL vision tower for %q (SigLIP2 NaFlex, %d layers, image token id %d, tower on CPU, float32) from %s\n", lm.name, enc.Cfg.NumHiddenLayers, id, dir)
	}
	return nil
}
