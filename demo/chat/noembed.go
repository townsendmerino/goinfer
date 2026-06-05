//go:build !embed && !prequant

package main

import (
	"errors"

	"github.com/townsendmerino/goinfer/decoder"
)

// hasEmbeddedModel is false in the default build: no baked-in model, so --model
// is required. The -tags embed / -tags prequant builds set it true and supply
// loadEmbedded.
const hasEmbeddedModel = false

func loadEmbedded(_ bool, _ decoder.Options) (*session, error) {
	return nil, errors.New("no embedded model in this build (build with -tags embed or -tags prequant)")
}
