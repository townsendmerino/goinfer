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
	"runtime"
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

// ---- compute dispatch (Layer A phase-1 completion: queue → buffers → encode → run) ----

var (
	selNewCommandQueue = objc.RegisterName("newCommandQueue")
	selNewFunctionName = objc.RegisterName("newFunctionWithName:")
	selNewPipelineFn   = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selNewBufferBytes  = objc.RegisterName("newBufferWithBytes:length:options:")
	selNewBufferLen    = objc.RegisterName("newBufferWithLength:options:")
	selContents        = objc.RegisterName("contents")
	selCommandBuffer   = objc.RegisterName("commandBuffer")
	selComputeEncoder  = objc.RegisterName("computeCommandEncoder")
	selSetPipeline     = objc.RegisterName("setComputePipelineState:")
	selSetBuffer       = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBuffers      = objc.RegisterName("setBuffers:offsets:withRange:")
	selDispatchThreads = objc.RegisterName("dispatchThreads:threadsPerThreadgroup:")
	selEndEncoding     = objc.RegisterName("endEncoding")
	selCommit          = objc.RegisterName("commit")
	selWaitCompleted   = objc.RegisterName("waitUntilCompleted")
	selDrain           = objc.RegisterName("drain")
	selGPUStartTime    = objc.RegisterName("GPUStartTime")    // CFTimeInterval (double), post-completion
	selGPUEndTime      = objc.RegisterName("GPUEndTime")      // GPU busy window for this command buffer
	selKernelStartTime = objc.RegisterName("kernelStartTime") // incl scheduling; > GPU window
	selKernelEndTime   = objc.RegisterName("kernelEndTime")
)

// mtlSize mirrors MTLSize {NSUInteger width,height,depth}. At 24 bytes (>16) it is
// passed to objc_msgSend BY REFERENCE per AAPCS64 — so the call site passes
// unsafe.Pointer(&sz), which lands the pointer in the arg register exactly as the ABI
// wants. (Getting this wrong is the "MTLSize struct-by-value" hazard the doc flagged.)
type mtlSize struct{ w, h, d uint64 }

// Queue / Pipeline / Buffer are thin id wrappers.
type Queue struct{ id objc.ID }
type Pipeline struct{ id objc.ID }
type Buffer struct {
	id  objc.ID
	n   int     // element count (float32)
	off uintptr // byte offset for binding (a sub-view into a larger buffer; 0 = whole)
}

// At returns a view of the buffer bound at byteOff — for reading a slice of a combined
// buffer (e.g. q/k/v out of a fused QKV output). Zero-copy; only the bind offset changes.
func (b Buffer) At(byteOff int) Buffer { b.off = uintptr(byteOff); return b }

// NewCommandQueue creates the command queue (built once, reused per token in the real backend).
func (d *Device) NewCommandQueue() Queue { return Queue{id: d.id.Send(selNewCommandQueue)} }

// NewComputePipeline looks a kernel up in a compiled library and builds its pipeline state.
func (d *Device) NewComputePipeline(lib objc.ID, fn string) (Pipeline, error) {
	f := lib.Send(selNewFunctionName, nsString(fn))
	if f == 0 {
		return Pipeline{}, fmt.Errorf("metal: no kernel %q in library", fn)
	}
	var nsErr objc.ID
	p := d.id.Send(selNewPipelineFn, f, unsafe.Pointer(&nsErr))
	if p == 0 {
		return Pipeline{}, fmt.Errorf("metal: pipeline %q: %s", fn, goString(nsErr.Send(selLocalizedDesc)))
	}
	return Pipeline{id: p}, nil
}

// NewBufferFloats uploads data into a shared (UMA host-visible) MTLBuffer.
func (d *Device) NewBufferFloats(data []float32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)*4), uintptr(0))
	runtime.KeepAlive(data)
	return Buffer{id: id, n: len(data)}
}

// NewBufferLen allocates an uninitialized shared MTLBuffer of nFloats float32s.
func (d *Device) NewBufferLen(nFloats int) Buffer {
	return Buffer{id: d.id.Send(selNewBufferLen, uintptr(nFloats*4), uintptr(0)), n: nFloats}
}

// NewBufferInt8 uploads int8 data (n counts BYTES for this buffer). NewBufferU32
// uploads a single uint32 (kernel scalar arg). Both are shared/UMA.
func (d *Device) NewBufferInt8(data []int8) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)), uintptr(0))
	runtime.KeepAlive(data)
	return Buffer{id: id, n: len(data)}
}

func (d *Device) NewBufferU32(v uint32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&v), uintptr(4), uintptr(0))
	runtime.KeepAlive(&v)
	return Buffer{id: id, n: 1}
}

// NewBufferUint32s uploads packed u32 data (n counts u32 words). Shared/UMA.
func (d *Device) NewBufferUint32s(data []uint32) Buffer {
	id := d.id.Send(selNewBufferBytes, unsafe.Pointer(&data[0]), uintptr(len(data)*4), uintptr(0))
	runtime.KeepAlive(data)
	return Buffer{id: id, n: len(data)}
}

// Floats / Int8s view the buffer's shared contents as a Go slice (zero-copy on UMA).
func (b Buffer) Floats() []float32 {
	return unsafe.Slice(objc.Send[*float32](b.id, selContents), b.n)
}

func (b Buffer) Int8s() []int8 {
	return unsafe.Slice(objc.Send[*int8](b.id, selContents), b.n)
}

// SetU32 overwrites a 1-word uniform buffer's contents in place (per-token: rope pos,
// nKeys). Zero-copy on UMA.
func (b Buffer) SetU32(v uint32) { *objc.Send[*uint32](b.id, selContents) = v }

// U32 reads a 1-word buffer's first uint32 (per-token: the fused-argmax token id). Zero-copy.
func (b Buffer) U32() uint32 { return *objc.Send[*uint32](b.id, selContents) }

// Run1D encodes and runs a 1-D kernel over n threads (threadgroup width tg), binding
// bufs at indices 0..len-1, and blocks until the GPU finishes. Manual autoreleasepool
// discipline (no ARC): the per-token commandBuffer/encoder are autoreleased into the
// pool and drained here, so the decode loop won't leak (doc risk #2).
func (q Queue) Run1D(p Pipeline, n, tg int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}

// Encoder batches many DIFFERENT kernel dispatches into ONE command buffer — the
// per-token shape the decode loop needs (whole layer stack → one commit/wait, per the
// tax finding). The default serial compute encoder inserts barriers between dependent
// dispatches (like WGSL's storage barriers), so chained kernels see prior writes.
type Encoder struct {
	pool objc.ID
	cb   objc.ID
	enc  objc.ID
	// Encode-tax trims (measured: together only ~0.5ms of the ~2.4ms/token overhead — the
	// rest is GPU-side per-dispatch latency, not Go-side msgSend): skip setComputePipelineState
	// when the pipeline is unchanged, and batch all buffer binds into ONE setBuffers call.
	curPipe objc.ID
	idScr   [16]objc.ID // reusable C-arrays for batched setBuffers:offsets:withRange:
	offScr  [16]uintptr
	// GPU timing of the last committed command buffer (seconds), captured in end() after
	// waitUntilCompleted (valid post-completion) and before the pool drains the cb.
	gpuStart, gpuEnd, kernStart, kernEnd float64
}

func (q Queue) begin() *Encoder {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	cb := q.id.Send(selCommandBuffer)
	return &Encoder{pool: pool, cb: cb, enc: cb.Send(selComputeEncoder)}
}

// dispatch encodes one kernel over n threads (threadgroup width tg, clamped ≤ n), binding
// bufs at indices 0..len-1. dispatchThreads launches EXACTLY n threads (non-uniform), so
// no out-of-range writes.
func (e *Encoder) dispatch(p Pipeline, n, tg int, bufs ...Buffer) {
	if tg > n {
		tg = n
	}
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	// Batch ALL buffer binds into one setBuffers:offsets:withRange: msgSend (vs one per
	// buffer) — the aggressive Go-side-encode-cost probe. NSRange{location,length} lowers to
	// two trailing word args.
	nb := len(bufs)
	for i, b := range bufs {
		e.idScr[i], e.offScr[i] = b.id, b.off
	}
	e.enc.Send(selSetBuffers, unsafe.Pointer(&e.idScr[0]), unsafe.Pointer(&e.offScr[0]), uintptr(0), uintptr(nb))
	runtime.KeepAlive(&e.idScr)
	runtime.KeepAlive(&e.offScr)
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
}

func (e *Encoder) end() {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selCommit)
	e.cb.Send(selWaitCompleted)
	// Timestamps are valid once completed; read before the pool releases the cb. On arm64
	// objc_msgSend returns the double in d0 — objc.Send[float64] uses the fp-return path.
	e.gpuStart = objc.Send[float64](e.cb, selGPUStartTime)
	e.gpuEnd = objc.Send[float64](e.cb, selGPUEndTime)
	e.kernStart = objc.Send[float64](e.cb, selKernelStartTime)
	e.kernEnd = objc.Send[float64](e.cb, selKernelEndTime)
	e.pool.Send(selDrain)
}

// Run1DBatch encodes `reps` dispatches of the same kernel into ONE command buffer and
// submits once — the shape a real token uses (a whole layer stack encoded into one
// command buffer, one commit/wait). Isolates the per-commit round-trip tax (reps=1)
// from the marginal per-encoded-dispatch msgSend cost (the slope over reps).
func (q Queue) Run1DBatch(p Pipeline, n, tg, reps int, bufs ...Buffer) {
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	for r := 0; r < reps; r++ {
		enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
}
