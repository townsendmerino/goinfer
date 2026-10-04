//go:build gpu

package gpu

import (
	"strings"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// softwareAdapterInfo is the pure detection (unit-tested without a GPU).
func softwareAdapterInfo(info wgpu.AdapterInfo) bool {
	if info.AdapterType == wgpu.AdapterTypeCPU {
		return true
	}
	name := strings.ToLower(info.Device + " " + info.Description)
	for _, s := range []string{"llvmpipe", "lavapipe", "softpipe", "swiftshader", "software"} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}
