package serveapp

import (
	"fmt"
	"log"

	"github.com/townsendmerino/goinfer/decoder"
)

// attachBlockDrafter attaches an already-loaded block drafter (--drafter) to an already-loaded model, so requests can
// take the block-speculative path.
//
// The caller loads dw in loadDecoder before the target model, because drafter-aware sizing needs the drafter's byte
// footprint priced into Options.ExtraResidentBytes ahead of BuildResident (docs/tasks/task-fit-to-hardware.md section 2). Taking
// the loaded value here means that pricing and this attach see the same weights, not two independent reads of one
// file.
//
// It fails startup rather than degrading silently: an operator who passed --drafter wants block drafting, and a wrong
// pairing or a backend that cannot host one should be a startup error seen once, not a fleet quietly serving at 1x.
// The one exception is a sampler the spec path does not support, a per-request property that falls back per request
// by design.
func attachBlockDrafter(lm *loadedModel, dw *decoder.DFlashDrafter) error {
	if lm.model == nil || !lm.model.BlockSpecCapable() {
		return fmt.Errorf("this model has no resident GPU decode path that can host a block " +
			"drafter. Block drafting needs a resident GPU backend (--backend cuda) and a model " +
			"that actually resolved to it — check the startup banner for a residency decline")
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
	warnThinkingTemplate(lm)
	return nil
}

// warnThinkingTemplate warns when the served template will put the target in thinking mode. Block drafting is
// lossless, so a thinking-mode target returns correct responses at reduced speed and nothing in any log says why: the
// pretrained drafters are trained on non-thinking output, so acceptance falls sharply and --drafter can turn from a
// win into a loss. The runtime acceptance guard limits the harm, so this is a warning and not a refusal.
func warnThinkingTemplate(lm *loadedModel) {
	if lm.tk == nil {
		return
	}
	if _, ok := lm.tk.TokenID("<think>"); !ok {
		return // no thinking mode in this vocab
	}
	log.Printf("WARNING: %q has a thinking mode and the served chat template leaves it ON. "+
		"Pretrained block drafters are trained on NON-thinking output, so acceptance roughly "+
		"halves (measured 5.76 -> 3.00 accepted/round on Qwen3-4B) and --drafter turns from a "+
		"~1.5x win into a ~0.8x loss. The runtime acceptance guard limits that to about "+
		"break-even, but you will not see the speedup. Serve a non-thinking template to get it.",
		lm.name)
}
