//go:build !embed

package main

// materializeEmbedded reports that there is no baked-in model in the default
// build, so --model is required. The embed build (-tags embed) replaces this
// with a version that inflates the //go:embed-ed zstd model to a temp file.
func materializeEmbedded() (path string, cleanup func(), ok bool) {
	return "", nil, false
}
