package cpuengine

import "slices"

// ShortConvState is one LFM2 conv layer's rolling window: the last K-1 Bx vectors, oldest
// first, where K is conv_L_cache. The K-th tap is the current token, so it is never stored.
//
// This is a strict subset of Mamba2State — that one carries ConvWin PLUS the SSM state, and
// LFM2 has no SSM. Kept as its own type rather than reusing Mamba2State so a conv layer
// cannot silently acquire an unused ssm field that some future code path starts reading.
type ShortConvState struct {
	ConvWin [][]float32 // up to K-1 prior Bx vectors, oldest first
}

func NewShortConvState() *ShortConvState { return &ShortConvState{} }

// push appends this token's Bx and keeps at most K-1 entries, dropping the oldest.
//
// Storing K-1 rather than K is the invariant the step function depends on: the current
// token is applied directly from its own tap, so a window of K would double-count it.
func (s *ShortConvState) push(bx []float32, K int) {
	if K <= 1 {
		return // kernel of 1 has no history
	}
	s.ConvWin = append(s.ConvWin, slices.Clone(bx))
	if len(s.ConvWin) > K-1 {
		s.ConvWin = s.ConvWin[len(s.ConvWin)-(K-1):]
	}
}
