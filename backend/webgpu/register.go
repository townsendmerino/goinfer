//go:build gpu

package webgpu

import (
	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/goinfer/decoder"
)

// init registers the WebGPU backend with both the goinfer decoder and the aikit encoder,
// along with its hardware info reporter, when this package is blank-imported.
func init() {
	// Return a literal nil interface on failure, not a typed-nil *webgpuBackend: a typed nil converts to
	// a non-nil Backend interface, which defeats the caller's fallback-to-CPU path.
	decoder.RegisterBackend("webgpu", func() (decoder.Backend, error) {
		b, err := newWebGPUBackend("decoder")
		if err != nil {
			return nil, err
		}
		return b, nil
	})
	encoder.RegisterBackend("webgpu", func() (encoder.Backend, error) {
		b, err := newWebGPUBackend("encoder")
		if err != nil {
			return nil, err
		}
		return b, nil
	})
	decoder.RegisterHardwareInfo("webgpu", webgpuHardwareInfo)
}
