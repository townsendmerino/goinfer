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
	warnThinkingTemplate(lm)
	return nil
}

// warnThinkingTemplate says so when the served template will put the target in THINKING mode.
//
// This is the one deployment footgun that survives every other safeguard, because it is
// INVISIBLE: block drafting is lossless, so a thinking-mode target returns correct responses at
// reduced speed and nothing in any log says why. Measured on Qwen3-4B, same model, same
// hardware, only the template differing:
//
//	non-thinking   5.76 accepted/round   1.57x
//	THINKING       3.00 accepted/round   0.82x   (0.96x once the acceptance guard trips)
//
// The runtime guard stops the bleeding, so this is a warning and not a refusal — a thinking
// deployment is merely not getting the win, rather than being harmed. But an operator who chose
// --drafter deserves to be told that this configuration is why it is not paying off.
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
