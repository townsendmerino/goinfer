package decoder

import "slices"

// shortConvState is one LFM2 conv layer's rolling window: the last K-1 Bx vectors, oldest
// first, where K is conv_L_cache. The K-th tap is the current token, so it is never stored.
//
// This is a strict subset of mamba2State — that one carries convWin PLUS the SSM state, and
// LFM2 has no SSM. Kept as its own type rather than reusing mamba2State so a conv layer
// cannot silently acquire an unused ssm field that some future code path starts reading.
type shortConvState struct {
	convWin [][]float32 // up to K-1 prior Bx vectors, oldest first
}

func newShortConvState() *shortConvState { return &shortConvState{} }

// push appends this token's Bx and keeps at most K-1 entries, dropping the oldest.
//
// Storing K-1 rather than K is the invariant the step function depends on: the current
// token is applied directly from its own tap, so a window of K would double-count it.
func (s *shortConvState) push(bx []float32, K int) {
	if K <= 1 {
		return // kernel of 1 has no history
	}
	s.convWin = append(s.convWin, slices.Clone(bx))
	if len(s.convWin) > K-1 {
		s.convWin = s.convWin[len(s.convWin)-(K-1):]
	}
}
