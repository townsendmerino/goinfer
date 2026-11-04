// Package prequant is the commands' name for the .giw transcode and the sidecar cache. The code lives in decoder
// (decoder/sidecar.go), because decoder.Load takes the sidecar itself by default (Options.DirectLoad); these are the
// same functions under the names cmd/prequant, serve and the other commands already use.
package prequant

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// Phases is decoder.TranscodePhases.
type Phases = decoder.TranscodePhases

// SidecarState is decoder.SidecarState.
type SidecarState = decoder.SidecarState

const (
	SidecarMissing = decoder.SidecarMissing
	SidecarCurrent = decoder.SidecarCurrent
	SidecarLegacy  = decoder.SidecarLegacy
)

// Transcode is decoder.TranscodeGIW.
func Transcode(ctx context.Context, in, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	return decoder.TranscodeGIW(ctx, in, out, quant, embedInt4, target)
}

// TranscodeTimed is decoder.TranscodeGIWTimed.
func TranscodeTimed(ctx context.Context, in, out, quant string, embedInt4 bool, target decoder.GIWTarget) (Phases, error) {
	return decoder.TranscodeGIWTimed(ctx, in, out, quant, embedInt4, target)
}

// TranscodeLoRA is decoder.TranscodeGIWLoRA.
func TranscodeLoRA(ctx context.Context, dir, adapter, out, quant string, embedInt4 bool, target decoder.GIWTarget) error {
	return decoder.TranscodeGIWLoRA(ctx, dir, adapter, out, quant, embedInt4, target)
}

// TranscodeLoRATimed is decoder.TranscodeGIWLoRATimed.
func TranscodeLoRATimed(ctx context.Context, dir, adapter, out, quant string, embedInt4 bool, target decoder.GIWTarget) (Phases, error) {
	return decoder.TranscodeGIWLoRATimed(ctx, dir, adapter, out, quant, embedInt4, target)
}

// AdapterLoRA is decoder.GIWAdapterLoRA.
func AdapterLoRA(adapterDir, modelPath string) (string, error) {
	return decoder.GIWAdapterLoRA(adapterDir, modelPath)
}

// EnsureCachedGIW is decoder.EnsureCachedGIW.
func EnsureCachedGIW(ctx context.Context, ggufPath, quant, backend string, embedInt4 bool) (string, error) {
	return decoder.EnsureCachedGIW(ctx, ggufPath, quant, backend, embedInt4)
}

// SidecarPathIfFresh is decoder.SidecarPathIfFresh.
func SidecarPathIfFresh(ggufPath, quant, backend string, embedInt4 bool) (string, bool) {
	return decoder.SidecarPathIfFresh(ggufPath, quant, backend, embedInt4)
}

// SidecarPathIfCurrent is decoder.SidecarPathIfCurrent.
func SidecarPathIfCurrent(src, quant, backend string, embedInt4 bool) (string, SidecarState) {
	return decoder.SidecarPathIfCurrent(src, quant, backend, embedInt4)
}

// SidecarPath is decoder.SidecarPath.
func SidecarPath(ggufPath, quant, backend string, embedInt4 bool) string {
	return decoder.SidecarPath(ggufPath, quant, backend, embedInt4)
}

// DefaultToSidecar is decoder.DefaultToSidecar.
func DefaultToSidecar(directLoad bool) bool { return decoder.DefaultToSidecar(directLoad) }
