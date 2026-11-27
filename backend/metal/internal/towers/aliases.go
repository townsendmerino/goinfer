//go:build darwin

package towers

import gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"

// The device types the towers use, under the names the metal package gives them.
type (
	Device   = gpu.Device
	Buffer   = gpu.Buffer
	Queue    = gpu.Queue
	Pipeline = gpu.Pipeline
	Encoder  = gpu.Encoder
)
