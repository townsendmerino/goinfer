package serveapp

import (
	"fmt"
	"log"

	"github.com/townsendmerino/goinfer/decoder"
)

// attachBlockDrafter loads a pretrained block drafter (--drafter) and attaches it to an
// already-loaded model, so requests can take the block-speculative path.
//
// IT FAILS STARTUP RATHER THAN DEGRADING SILENTLY. An operator who passed --drafter wants block
// drafting; a wrong pairing or a backend that cannot host one should be a startup error they see
// once, not a fleet quietly serving at 1x. The one exception is a sampler the spec path does not
// support, which is a per-REQUEST property and falls back per request by design.
func attachBlockDrafter(lm *loadedModel, dir string) error {
	if lm.model == nil || !lm.model.BlockSpecCapable() {
		return fmt.Errorf("this model has no resident GPU decode path that can host a block " +
			"drafter. Block drafting needs a resident GPU backend (--backend cuda) and a model " +
			"that actually resolved to it — check the startup banner for a residency decline")
	}
	dw, err := decoder.LoadDFlashDrafter(dir)
	if err != nil {
		return fmt.Errorf("load drafter: %w", err)
	}
	spec, err := lm.model.NewBlockSpec(dw, dw.TargetLayerIDs())
	if err != nil {
		if decoder.ErrBlockSpecUnsupported(err) {
			return fmt.Errorf("backend cannot host a block drafter: %w", err)
		}
		return err
	}
	lm.blockSpec = spec
	geo := dw.DrafterGeometry()
	log.Printf("block drafter attached to %q: %d layers, hidden %d, block %d, %d taps %v — "+
		"greedy requests take the speculative path, sampled ones fall back",
		lm.name, geo.Layers, geo.Hidden, dw.BlockSize(), len(dw.TargetLayerIDs()), dw.TargetLayerIDs())
	return nil
}
