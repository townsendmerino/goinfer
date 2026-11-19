//go:build linux

package gpudev

import "unsafe"

// TraceOp names one stream operation TraceForTest sees.
type TraceOp uint8

const (
	TraceLaunch      TraceOp = iota + 1 // Queue.Launch, or Queue.LaunchDirect's own launch
	TraceUpload                         // Upload and every synchronous host-to-device copy built on it (UploadBatch per pair)
	TraceDownload                       // Download and ReadToHost
	TraceUploadAsync                    // UploadAsyncAt, UploadAsyncAtFrom
	TraceZeroAsync                      // ZeroAsync
	TraceCopyDevice                     // CopyDevice, CopyDeviceBatch per pair (before coalescing)
	TraceAlloc                          // a device allocation (MustBufOn, MustBufSplitOn): Dst is its base, Line its plan line
	TraceFree                           // ReleaseBuf of a whole allocation: Dst is its base
)

// TraceEvent is one operation as issued, or an allocation as made. Launch fills Pipe (the pipeline's Ptr), Cfg and Args (a copy:
// a hook site must not let a launch's own parameters escape, or every launch's variadic slice moves to the heap in production);
// every copy fills Dst (the destination's bind address), Bytes and, for a synchronous upload, Src (the host bytes, valid only
// during the call); CopyDevice also fills SrcPtr.
type TraceEvent struct {
	Op     TraceOp
	Pipe   uintptr
	Cfg    LaunchConfig
	Args   []KernelArg
	Dst    uint64
	SrcPtr uint64
	Bytes  int
	Src    []byte
	Line   string
}

// TraceForTest, when set (tests only), is called with every operation above, on the issuing thread, before it is issued: a
// test records a launch-and-copy sequence to hold a refactor to. It is nil in production, where each site pays one nil check.
var TraceForTest func(TraceEvent)

// Raw is the parameter as LaunchDirect passes it: a device pointer (0 for ArgNull) or a scalar's bits in the low bytes, and its
// size in bytes (8 for a pointer; 0 for a nil buffer).
func (k KernelArg) Raw() (bits uint64, size int) { return k.raw, int(k.n) }

// Ptr is the identity a pipeline compares by, for naming it in a trace.
func (p Pipeline) Ptr() uintptr { return uintptr(unsafe.Pointer(p.f)) }

// BindAddr is the device address a kernel argument of b binds (Arg(b)'s pointer), 0 for a nil buffer.
func (b Buffer) BindAddr() uint64 {
	switch {
	case b.raw != 0:
		return uint64(b.raw) + uint64(b.off)
	case b.b != nil:
		return uint64(b.b.DevicePtr()) + uint64(b.off)
	}
	return 0
}
