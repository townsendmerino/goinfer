//go:build !embed

package embedmodel

// Bytes reports no baked-in model: the default build requires --model.
// Build with -tags embed (see build-embed.sh) to bake the GGUF in.
func Bytes() ([]byte, bool) { return nil, false }
