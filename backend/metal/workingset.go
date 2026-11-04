//go:build darwin

package metal

import (
	"sync"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// metalWorkingSetCap is the system default device's recommendedMaxWorkingSetSize in bytes, or 0 when it cannot be read: how much
// memory Metal can use for this GPU's resources before performance suffers. A variable so a test can stand in a device.
var metalWorkingSetCap = func() int64 { return deviceWorkingSet() }

// deviceWorkingSet asks MTLCreateSystemDefaultDevice once (the same device every resident uses; aikit's Device keeps its handle
// private), then releases it.
var deviceWorkingSet = sync.OnceValue(func() (n int64) {
	defer func() {
		if recover() != nil { // RegisterLibFunc panics on a missing symbol
			n = 0
		}
	}()
	lib, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return 0
	}
	var create func() objc.ID
	purego.RegisterLibFunc(&create, lib, "MTLCreateSystemDefaultDevice")
	dev := create()
	if dev == 0 {
		return 0
	}
	defer dev.Send(objc.RegisterName("release"))
	return int64(objc.Send[uint64](dev, objc.RegisterName("recommendedMaxWorkingSetSize")))
})
