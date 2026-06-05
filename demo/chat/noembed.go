//go:build !embed

package main

import "errors"

// hasEmbeddedModel is false in the default build: there is no baked-in model, so
// --model is required. The embed build (-tags embed) sets it true and provides
// the inflate functions below.
const hasEmbeddedModel = false

func embeddedModelBytes() ([]byte, error) {
	return nil, errors.New("no embedded model in this build (build with -tags embed)")
}

func embeddedModelTemp() (path string, cleanup func(), err error) {
	return "", nil, errors.New("no embedded model in this build (build with -tags embed)")
}
