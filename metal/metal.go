//go:build darwin

// Package metal is the Phase-1 (Layer A) proof for the cgo-free native-Metal spike
// (docs/task-metal-cgofree-spike.md): can we reach Metal + compile/run MSL kernels
// through purego-objc with CGO_ENABLED=0 and correct output? This file is the
// binding skeleton — device, the runtime MSL compiler, and the ONE thing that must
// be right before any kernel: MTLCompileOptions.languageVersion.
//
// ⚠️ Risk #7 (the landmine): CGO_ENABLED=0 macOS binaries omit LC_BUILD_VERSION
// (golang/go#77917), so Metal's runtime compiler DEFAULTS languageVersion to MSL 2.4
// and silently strips modern types. We NEVER rely on the default: every library is
// compiled with an explicit MTLCompileOptions at MSL >=3.1, and we assert it took.
package metal

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// MTLLanguageVersion values (NSUInteger). MSL 3.1 = (3<<16)|1. Rig is macOS 26, so
// >=3.1 is safe; bump to match features used.
const MSL3_1 uint = (3 << 16) | 1

var mtlCreateSystemDefaultDevice func() uintptr

func init() {
	h, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal",
		purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic("metal: dlopen Metal.framework: " + err.Error())
	}
	purego.RegisterLibFunc(&mtlCreateSystemDefaultDevice, h, "MTLCreateSystemDefaultDevice")
}

// cached selectors
var (
	selAlloc              = objc.RegisterName("alloc")
	selInit               = objc.RegisterName("init")
	selRelease            = objc.RegisterName("release")
	selSetLanguageVersion = objc.RegisterName("setLanguageVersion:")
	selLanguageVersion    = objc.RegisterName("languageVersion")
	selNewLibrarySource   = objc.RegisterName("newLibraryWithSource:options:error:")
	selStringWithUTF8     = objc.RegisterName("stringWithUTF8String:")
	selUTF8String         = objc.RegisterName("UTF8String")
	selLocalizedDesc      = objc.RegisterName("localizedDescription")
	selName               = objc.RegisterName("name")
)

// nsString wraps a Go string as an autoreleased NSString.
func nsString(s string) objc.ID {
	b := append([]byte(s), 0) // NUL-terminated C string
	id := objc.ID(objc.GetClass("NSString")).Send(selStringWithUTF8, unsafe.Pointer(&b[0]))
	_ = b // keep alive across the call
	return id
}

// goString reads a C `const char *` (id.UTF8String) back into a Go string.
func goString(id objc.ID) string {
	if id == 0 {
		return ""
	}
	p := objc.Send[*byte](id, selUTF8String)
	if p == nil {
		return ""
	}
	var out []byte
	for ptr := uintptr(unsafe.Pointer(p)); ; ptr++ {
		c := *(*byte)(unsafe.Pointer(ptr))
		if c == 0 {
			break
		}
		out = append(out, c)
	}
	return string(out)
}

// Device wraps an MTLDevice.
type Device struct{ id objc.ID }

// CreateSystemDefaultDevice reaches Metal cgo-free (MTLCreateSystemDefaultDevice is a
// plain C export in Metal.framework).
func CreateSystemDefaultDevice() (*Device, error) {
	p := mtlCreateSystemDefaultDevice()
	if p == 0 {
		return nil, fmt.Errorf("metal: MTLCreateSystemDefaultDevice returned nil (no Metal GPU?)")
	}
	return &Device{id: objc.ID(p)}, nil
}

// Name is the device's product name (e.g. "Apple M1 Pro").
func (d *Device) Name() string { return goString(d.id.Send(selName)) }

// CompileLibrary compiles MSL `src` at languageVersion `ver` — with the landmine
// defused: an explicit MTLCompileOptions, plus a read-back assertion that the option
// took (loud, not silent). Returns the MTLLibrary id.
func (d *Device) CompileLibrary(src string, ver uint) (objc.ID, error) {
	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	opts.Send(selSetLanguageVersion, ver)
	if got := uint(objc.Send[uintptr](opts, selLanguageVersion)); got != ver {
		return 0, fmt.Errorf("metal: languageVersion set to %#x but reads %#x — the LC_BUILD_VERSION landmine (golang/go#77917)", ver, got)
	}

	var nsErr objc.ID
	lib := d.id.Send(selNewLibrarySource, nsString(src), opts, unsafe.Pointer(&nsErr))
	if lib == 0 {
		return 0, fmt.Errorf("metal: newLibraryWithSource failed: %s", goString(nsErr.Send(selLocalizedDesc)))
	}
	return lib, nil
}
