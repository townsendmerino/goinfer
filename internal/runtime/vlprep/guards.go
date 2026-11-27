package vlprep

import (
	"fmt"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// CheckImageBlockFitsWindow refuses a bidirectional image block longer than a model's sliding window.
func CheckImageBlockFitsWindow(slidingWindow, imgLen int) error {
	if slidingWindow > 0 && imgLen > slidingWindow {
		return fmt.Errorf("decoder: an image block of %d tokens is longer than this model's %d-token sliding window, which the bidirectional image mask does not support (docs/multimodal.md)", imgLen, slidingWindow)
	}
	return nil
}

// CheckDeepstack validates Qwen3-VL DeepStack feature sets: at most numLayers sets, each holding imgLen * hiddenDim values.
func CheckDeepstack(numLayers, hiddenDim int, deep [][]float32, imgLen int) error {
	if len(deep) > numLayers {
		return fmt.Errorf("decoder: %d DeepStack sets for %d layers", len(deep), numLayers)
	}
	for l, set := range deep {
		if len(set) != imgLen*hiddenDim {
			return fmt.Errorf("decoder: DeepStack set %d has %d values, want %d x %d", l, len(set), imgLen, hiddenDim)
		}
	}
	return nil
}

// CheckQwen35VLReady refuses an image turn on a non-Qwen3.5 architecture or one lacking a valid 3-axis mrope_section.
func CheckQwen35VLReady(arch *darch.Architecture) error {
	if arch.Qwen35 == nil {
		return fmt.Errorf("decoder: %q is not a Gated-DeltaNet hybrid", arch.Name)
	}
	if len(arch.MRopeSection) != 3 {
		return fmt.Errorf("decoder: %s has no mrope_section in rope_parameters; refusing an image turn rather than rotating image tokens by scalar position", arch.Name)
	}
	if want := len(arch.RoPEInvFreq(0)); arch.MRopeSection[0]+arch.MRopeSection[1]+arch.MRopeSection[2] != want {
		return fmt.Errorf("decoder: %s mrope_section %v sums to %d, want %d (rotary_dim/2)", arch.Name, arch.MRopeSection, arch.MRopeSection[0]+arch.MRopeSection[1]+arch.MRopeSection[2], want)
	}
	return nil
}
