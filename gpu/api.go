//go:build gpu

package gpu

import "github.com/townsendmerino/goinfer/gpu/internal/kernels"

// gpu's public names. The kernel layer lives in gpu/internal/kernels; each type here is an alias of its type there,
// and each function forwards to its function there, so a program written against gpu compiles unchanged.

// AttnWeights is kernels.AttnWeights.
type AttnWeights = kernels.AttnWeights

// Context is kernels.Context.
type Context = kernels.Context

// DecodeRunner is kernels.DecodeRunner.
type DecodeRunner = kernels.DecodeRunner

// DeviceBuffer is kernels.DeviceBuffer.
type DeviceBuffer = kernels.DeviceBuffer

// GEMVRunner is kernels.GEMVRunner.
type GEMVRunner = kernels.GEMVRunner

// GPUEverAvailable is kernels.GPUEverAvailable.
func GPUEverAvailable() bool { return kernels.GPUEverAvailable() }

// LayerW is kernels.LayerW.
type LayerW = kernels.LayerW

// LiveBufferBytes is kernels.LiveBufferBytes.
func LiveBufferBytes() int64 { return kernels.LiveBufferBytes() }

// ModelW is kernels.ModelW.
type ModelW = kernels.ModelW

// New is kernels.New.
func New() (*Context, error) { return kernels.New() }

// ResidentMatrix is kernels.ResidentMatrix.
type ResidentMatrix = kernels.ResidentMatrix

// ResidentStackedW8A8 is kernels.ResidentStackedW8A8.
type ResidentStackedW8A8 = kernels.ResidentStackedW8A8

// ResidentW4A8 is kernels.ResidentW4A8.
type ResidentW4A8 = kernels.ResidentW4A8

// ResidentW8A8 is kernels.ResidentW8A8.
type ResidentW8A8 = kernels.ResidentW8A8

// VisionEncoder is kernels.VisionEncoder.
type VisionEncoder = kernels.VisionEncoder
