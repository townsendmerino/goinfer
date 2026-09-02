//go:build gpu

package gpu

import (
	"sync/atomic"

	"github.com/cogentcore/webgpu/wgpu"
)

// Live device-buffer accounting.
//
// A full ./gpu/ run used to climb to 7,782 MiB of 8,192 and then fail every later test with
// "failed to request device" — an out-of-memory wearing an unrelated message. Context.Close
// releases the DEVICE, but buffers uploaded through UploadF32/UploadW8A8/UploadW4A8Packed are
// caller-owned, and each live one pins that memory.
//
// nvidia-smi can show the total but cannot say WHICH test left it behind. This counter can:
// it is exact, in-process, and read at each test's start, so a full run attributes the growth
// to the test that caused it. See TestNoBufferLeak.
var liveBufferBytes atomic.Int64

// LiveBufferBytes reports device-buffer bytes allocated through goinfer's wrappers and not
// yet released. Behind no build tag because it is also a legitimate runtime gauge, but it
// counts only the wrapper types — raw CreateBufferInit scratch inside a single call is
// released before that call returns and is not tracked.
func LiveBufferBytes() int64 { return liveBufferBytes.Load() }

// GPUEverAvailable reports whether a Context was successfully created at least once in this
// process. A diagnostics gauge like LiveBufferBytes, and the one tests need to tell "this
// machine has no GPU" (a legitimate skip) from "this process ran out of one" (a defect that
// must fail loudly, because skipping there silently turns a correctness gate into a no-op).
func GPUEverAvailable() bool { return gpuEverAvailable.Load() }

func accountAlloc(n int64) {
	if n > 0 {
		liveBufferBytes.Add(n)
	}
}

func accountFree(n int64) {
	if n > 0 {
		liveBufferBytes.Add(-n)
	}
}

// newDeviceBuffer wraps a wgpu buffer and accounts for its size. n is the ELEMENT count; the
// buffer's real byte size is read from the handle so the figure is right for u32-packed int8
// as well as f32.
func newDeviceBuffer(buf *wgpu.Buffer, n int) *DeviceBuffer {
	var sz int64
	if buf != nil {
		sz = int64(buf.GetSize())
	}
	accountAlloc(sz)
	return &DeviceBuffer{buf: buf, n: n, bytes: sz}
}
