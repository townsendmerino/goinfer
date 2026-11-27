//go:build gpu

package towers

import (
	"io"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
)

type (
	Context      = kernels.Context
	DeviceBuffer = kernels.DeviceBuffer
	ResidentW8A8 = kernels.ResidentW8A8
	towerPipe    = kernels.TowerPipe
)

var _ io.Closer = (*VisionEncoder)(nil)

func maxWG(groups int) int {
	if groups < 1 {
		return 1
	}
	if groups > 65535 {
		return 65535
	}
	return groups
}
