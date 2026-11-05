package serveapp

import (
	"fmt"
	"io"
	"slices"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// attachWhisperDevice puts a Whisper-style encoder (Whisper's, and Voxtral's tower) on backend's device when this binary registers one for it, and returns where the encoder
// runs: the backend's name, or "CPU". A decline (no VRAM left, no device) leaves the encoder on the CPU and says why on warn.
func attachWhisperDevice(enc *audio.WhisperEncoder, backend string, warn io.Writer) (audio.WhisperDevice, string) {
	if !slices.Contains(multimodal.WhisperDevices(), backend) {
		return nil, "CPU"
	}
	d, err := multimodal.UseWhisperDevice(backend, enc)
	if err != nil {
		fmt.Fprintf(warn, "audio: the %s Whisper encoder declined (%v); the encoder runs on the CPU\n", backend, err)
		return nil, "CPU"
	}
	if p, ok := d.(interface{ Placement() string }); ok {
		return d, backend + " (" + p.Placement() + ")"
	}
	return d, backend
}
