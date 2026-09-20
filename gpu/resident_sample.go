//go:build gpu

package gpu

import (
	"math"

	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentSample = (*residentDecoder)(nil)

// SampleAvailable reports whether the device may draw the next token: only when the device row IS the raw
// LM-head output the host sampler would see (a final-logit softcap is applied on the host after readback).
func (rd *residentDecoder) SampleAvailable() bool { return rd.finalSoftcap == 0 }

// ForwardSample runs one token's forward and draws the next token on-device by Gumbel-max
// (decoder.ResidentSample). It mirrors Forward's preamble exactly (capacity check, sequence reset).
func (rd *residentDecoder) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	if err := rd.checkCap(pos, 1); err != nil {
		return 0, err
	}
	if pos == 0 {
		rd.Reset()
		if rd.resetErr != nil {
			return 0, rd.resetErr
		}
	}
	invT := float32(1 / temperature)
	if math.IsInf(float64(invT), 0) { // absurdly small temperature: greedy, as the host does
		logits, err := rd.runner.Run(embedding, pos, pos)
		if err != nil {
			return 0, err
		}
		best, bi := float32(math.Inf(-1)), 0
		for i, v := range logits {
			if v > best {
				best, bi = v, i
			}
		}
		return bi, nil
	}
	return rd.runner.RunSample(embedding, pos, pos, invT, seed, draw)
}
