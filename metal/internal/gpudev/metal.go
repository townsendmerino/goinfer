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

package gpudev

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// MSL3_1 and friends are MTLLanguageVersion values (NSUInteger). MSL 3.1 = (3<<16)|1.
// Rig is macOS 26, so
// >=3.1 is safe; bump to match features used.
const MSL3_1 uint = (3 << 16) | 1

var mtlCreateSystemDefaultDevice func() uintptr

// loadOnce guards the Metal.framework dlopen + symbol registration. It is a
// sync.Once rather than a package init so it does not depend on init ordering
// between files in this package (annbackend.go's init registers the backend and
// calls CreateSystemDefaultDevice, and Go runs inits in filename order — a plain
// init here would run after that call). Callers reach it via CreateSystemDefaultDevice.
var loadOnce sync.Once

func loadMetal() {
	loadOnce.Do(func() {
		h, err := purego.Dlopen("/System/Library/Frameworks/Metal.framework/Metal",
			purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			panic("metal: dlopen Metal.framework: " + err.Error())
		}
		purego.RegisterLibFunc(&mtlCreateSystemDefaultDevice, h, "MTLCreateSystemDefaultDevice")
	})
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
	selMaxTgMem           = objc.RegisterName("maxThreadgroupMemoryLength") // device tile-memory limit (bytes)
	selThreadExecWidth    = objc.RegisterName("threadExecutionWidth")       // pipeline SIMD-group width
	selCurrentAllocSize   = objc.RegisterName("currentAllocatedSize")       // actual GPU-side bytes (goinfer audit-metal-2026-09-12 G-06)
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
	// Walk the NUL terminator with unsafe.Add (pointer arithmetic that stays *unsafe.Pointer),
	// then materialize with unsafe.Slice — avoids the uintptr→unsafe.Pointer round-trip that
	// `go vet` flags as "possible misuse" (and that CI's vet rejects). The bytes are C-managed
	// (objc UTF8String), so GC movement is not a concern; this is purely the vet-clean idiom.
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// Device wraps an MTLDevice and OWNS every MTLBuffer allocated through it.
//
// purego means no ARC: an MTLBuffer that is never sent -release leaks by construction. There is
// no Metal equivalent of CUDA's "destroy the context and reclaim everything", so the only way to
// free a resident model is to release each buffer — hence this ledger. Every allocation goes
// through MustBuf, which records the id; ReleaseAll frees them (see Resident.Close).
type Device struct {
	id     objc.ID
	mu     sync.Mutex
	allocs []devAlloc // every MTLBuffer handed out by this Device, for ReleaseAll
	objs   []objc.ID  // non-buffer +1-owned objc objects (command queue, pipelines, libraries) — M24
	leaser Leaser     // every allocation's lease (MM3); nil leases nothing
	// traceMoved is how far CurrentAllocatedSize moved across the last traced allocation, which traceSet says is pending for
	// mustBuf (AllocTraceForTest only).
	traceMoved int64
	traceSet   bool
	act        activity // what its queues ran (SB11)
}

// devAlloc is one MTLBuffer and its lease's release.
type devAlloc struct {
	id      objc.ID
	release func()
}

// Leaser is told of every MTLBuffer the device hands out: whether it aliases caller memory (NewBufferNoCopy, fileMapped), its
// plan line, its length (requested) and what Metal allocated for it (charged). It returns the release, called once when the
// buffer is released. The backend attaches it; this package keeps no account of its own.
type Leaser interface {
	Lease(fileMapped bool, line string, requested, charged int64) (release func())
}

// AllocTrace is one buffer allocation as AllocTraceForTest sees it.
type AllocTrace struct {
	ID         uintptr // the MTLBuffer, which FreeTraceForTest names again when it is released
	FileMapped bool    // a no-copy alias of caller memory
	Line       string  // its plan line
	Length     int64
	Allocated  int64 // Metal's allocatedSize
	Charged    int64 // what was passed to the leaser (0 with none)
	Leased     bool  // the device had a leaser
	Moved      int64 // how far CurrentAllocatedSize moved across the allocation itself; -1 outside the constructors
}

// AllocTraceForTest, when set, sees every buffer mustBuf hands out, leased or not, and FreeTraceForTest every one ReleaseBuf
// releases, with how far CurrentAllocatedSize moved across the release. G-MM3c-4 reconciles the ledger per buffer with them.
// They assume the device allocates and frees one buffer at a time while set. Nil in production, where nothing extra is read.
var (
	AllocTraceForTest func(d *Device, a AllocTrace)
	FreeTraceForTest  func(d *Device, id uintptr, moved int64)
)

// AllocUntracedForTest allocates an n-byte MTLBuffer around the constructors (neither leased nor traced), and returns its
// release: G-MM3c-4's plant for an allocation the ledger never sees.
func (d *Device) AllocUntracedForTest(n int) (release func()) {
	id := d.id.Send(selNewBufferLen, uintptr(n), uintptr(0))
	return func() { id.Send(selRelease) }
}

// allocID runs alloc, the Objective-C call that allocates a buffer, and, while AllocTraceForTest is set, records how far
// CurrentAllocatedSize moved across it for mustBuf to report.
func (d *Device) allocID(alloc func() objc.ID) objc.ID {
	if AllocTraceForTest == nil {
		return alloc()
	}
	before := d.CurrentAllocatedSize()
	id := alloc()
	moved := int64(d.CurrentAllocatedSize()) - int64(before)
	d.mu.Lock()
	d.traceMoved, d.traceSet = moved, true
	d.mu.Unlock()
	return id
}

// NewLeaser, when set, gives every device CreateSystemDefaultDevice creates its leaser: the backend sets it at init.
var NewLeaser func() Leaser

// UnlinedLine is the plan line an allocation through an unlined constructor (MustBuf, NewBufferLen, NewBufferLenOf, …) leases on:
// empty, so unplanned. Production code calls the lined constructors (a source scan in the metal module holds that); a test
// package may set it to its test line for its own buffers.
var UnlinedLine string

// ReleasedForTest, when set, is called by ReleaseObjects after everything the device tracked is released, so a test can read the
// leaser's account then (I1, I2). A test seam, nil in production.
var ReleasedForTest func(*Device)

// Leaser is the attached leaser, or nil.
func (d *Device) Leaser() Leaser {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.leaser
}

// TrackObj records a +1-owned objc object (command queue / pipeline / library) so ReleaseObjects
// frees it at Close. MTLFunctions are NOT tracked — they are released immediately once their
// pipeline state exists (a pipeline does not retain its function). See M24(b).
func (d *Device) TrackObj(id objc.ID) {
	if id == 0 {
		return
	}
	d.mu.Lock()
	d.objs = append(d.objs, id)
	d.mu.Unlock()
}

// ReleaseObjects releases the non-buffer objc objects (command queue, pipelines, libraries) this
// Device tracked, then the MTLDevice itself, and empties the ledgers. purego has no ARC, so these
// leak per load/unload otherwise (M24(b): ReleaseAll freed buffers only). Idempotent: it nils the
// device id, so a second call (or a double Close) is a no-op. Caller MUST ensure no GPU work is in
// flight (Resident.Close stops+waits the executor first). MTLCreateSystemDefaultDevice hands back a
// +1 retain PER CALL on the (shared) system device, so releasing here balances THIS Device's retain
// and leaves any other resident model's device retain intact.
func (d *Device) ReleaseObjects() {
	d.mu.Lock()
	objs := d.objs
	d.objs = nil
	id := d.id
	d.id = 0
	d.mu.Unlock()
	for _, o := range objs {
		o.Send(selRelease)
	}
	if id != 0 {
		id.Send(selRelease)
		if ReleasedForTest != nil {
			ReleasedForTest(d)
		}
	}
}

// CreateSystemDefaultDevice reaches Metal cgo-free (MTLCreateSystemDefaultDevice is a
// plain C export in Metal.framework).
func CreateSystemDefaultDevice() (*Device, error) {
	loadMetal()
	p := mtlCreateSystemDefaultDevice()
	if p == 0 {
		return nil, fmt.Errorf("metal: MTLCreateSystemDefaultDevice returned nil (no Metal GPU?)")
	}
	d := &Device{id: objc.ID(p)}
	if NewLeaser != nil {
		d.leaser = NewLeaser()
	}
	return d, nil
}

// Name is the device's product name (e.g. "Apple M1 Pro").
func (d *Device) Name() string { return goString(d.id.Send(selName)) }

// MaxThreadgroupMemoryLength is the device's maximum threadgroup (tile) memory per dispatch, in
// bytes (~32 KiB on Apple GPUs). A dispatch whose setThreadgroupMemoryLength exceeds it aborts the
// command buffer, so callers that size threadgroup scratch from model dims must check against this
// and decline (goinfer audit M-11). Integer return → objc.Send[uintptr] (arm64 x0 path).
func (d *Device) MaxThreadgroupMemoryLength() int {
	return int(objc.Send[uintptr](d.id, selMaxTgMem))
}

// CurrentAllocatedSize is the device's actual current GPU-side allocation, in bytes — the real
// Metal-reported number, not this package's own bookkeeping (goinfer audit-metal-2026-09-12 G-06).
// LedgerLen counts entries in allocs/objs, which ReleaseAll/ReleaseObjects clear unconditionally
// BEFORE sending release to each id: a leak test asserting LedgerLen()==0 after Close cannot
// distinguish "every buffer was actually released" from "the release loop silently did nothing" —
// both leave the Go-side slice empty. CurrentAllocatedSize reads MTLDevice's own live allocation
// total instead, so it reflects whether the underlying native memory was actually freed,
// independent of this package's tracking and immune to macOS's page compression (unlike RSS —
// see close_leak_test.go's own comment on why RSS is unreliable in both directions here).
func (d *Device) CurrentAllocatedSize() uint64 {
	return uint64(objc.Send[uintptr](d.id, selCurrentAllocSize))
}

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
	d.TrackObj(lib) // +1-owned; released at Close (M24)
	return lib, nil
}

// Fast-math / math-mode selectors and the MTLMathMode raw values (probed empirically
// on macOS 26.5.2: MTLMathModeSafe=0, Relaxed=1, Fast=2, and the default is Fast).
// Metal 3.2 / macOS 15 introduced setMathMode: and DEPRECATED setFastMathEnabled:; we
// prefer the former where it exists and fall back to the latter on older OSes, and
// read the result back either way (see setPreciseMath).
var (
	selRespondsToSel = objc.RegisterName("respondsToSelector:")
	selSetMathMode   = objc.RegisterName("setMathMode:")        // Metal 3.2+
	selMathMode      = objc.RegisterName("mathMode")            // Metal 3.2+
	selSetFastMath   = objc.RegisterName("setFastMathEnabled:") // deprecated in Metal 3.2
	selFastMath      = objc.RegisterName("fastMathEnabled")     // getter is fastMathEnabled, NOT isFastMathEnabled
)

const mtlMathModeSafe uintptr = 0 // no fast-math: true divides, no f32 reassoc/contract

// respondsTo reports whether obj implements sel. BOOL is a signed char returned in the
// low byte of the result register, so mask before testing.
func respondsTo(obj objc.ID, sel objc.SEL) bool {
	return objc.Send[uintptr](obj, selRespondsToSel, uintptr(sel))&0xff != 0
}

// respondsToFn is the seam respondsTo goes through, so a test can force the
// "no verifiable API" path without needing a future OS. Production is respondsTo.
var respondsToFn = respondsTo

// setPreciseMath disables fast-math on opts and VERIFIES the demand took, mirroring the
// languageVersion read-back guard. Without the read-back a future OS could silently
// no-op the setter — setFastMathEnabled: is already deprecated (→ MTLMathMode in Metal
// 3.2 / macOS 15) — and hand back a fast-math library while claiming to be precise,
// breaking the ViT parity gate (its exact maxAbs/127 quant scale needs true divides)
// with no error and no red test. Prefers the non-deprecated MTLMathModeSafe; falls back
// to setFastMathEnabled:NO on older OSes.
//
// It requires that BOTH the setter AND the getter of the chosen path respond before it
// touches either — so a set is never issued through a selector we cannot read back —
// and if NEITHER pair is verifiable it returns an error naming both. "I could not
// verify" must fail loudly, not fall through to an unchecked "precise" library: that
// silent pass is the exact failure this guard exists to prevent.
func setPreciseMath(opts objc.ID) error {
	switch {
	case respondsToFn(opts, selSetMathMode) && respondsToFn(opts, selMathMode):
		opts.Send(selSetMathMode, mtlMathModeSafe)
		if got := objc.Send[uintptr](opts, selMathMode); got != mtlMathModeSafe {
			return fmt.Errorf("metal: setMathMode:MTLMathModeSafe did not take (mathMode reads %d, want %d) — the 'precise' library would be silently fast-math", got, mtlMathModeSafe)
		}
		return nil
	case respondsToFn(opts, selSetFastMath) && respondsToFn(opts, selFastMath):
		opts.Send(selSetFastMath, uintptr(0)) // fastMathEnabled = NO
		if got := objc.Send[uintptr](opts, selFastMath) & 0xff; got != 0 {
			return fmt.Errorf("metal: setFastMathEnabled:NO did not take (fastMathEnabled reads YES) — likely deprecated on this OS (→ MTLMathMode); the 'precise' library is silently fast-math")
		}
		return nil
	default:
		return fmt.Errorf("metal: cannot verify precise math — MTLCompileOptions responds to neither setMathMode:/mathMode (Metal 3.2+) nor setFastMathEnabled:/fastMathEnabled (deprecated); refusing to return an unverified 'precise' library")
	}
}

// CompileLibraryPrecise compiles MSL with fast-math DISABLED: divides are true
// divides (not reciprocal approximations) and the compiler won't reassociate or
// contract f32 — so a computation matches the CPU/CUDA reference where f32 can. The
// ViT kernels use this because their parity gate needs an exact per-row quant scale
// (maxAbs/127, which fast-math turns into maxAbs*rcp(127), off by a ULP). CompileLibrary
// keeps the default (fast-math on), which is what goinfer's tuned decode kernels want.
//
// Both the languageVersion and the fast-math demand are read back and asserted, so a
// silent no-op (the LC_BUILD_VERSION landmine, or a deprecated fast-math setter on a
// future OS) fails loudly here instead of producing a wrong-numerics library.
func (d *Device) CompileLibraryPrecise(src string, ver uint) (objc.ID, error) {
	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	defer opts.Send(selRelease)
	opts.Send(selSetLanguageVersion, ver)
	if got := uint(objc.Send[uintptr](opts, selLanguageVersion)); got != ver {
		return 0, fmt.Errorf("metal: languageVersion set to %#x but reads %#x — the LC_BUILD_VERSION landmine (golang/go#77917)", ver, got)
	}
	if err := setPreciseMath(opts); err != nil {
		return 0, err
	}
	var nsErr objc.ID
	lib := d.id.Send(selNewLibrarySource, nsString(src), opts, unsafe.Pointer(&nsErr))
	if lib == 0 {
		return 0, fmt.Errorf("metal: newLibraryWithSource (precise) failed: %s", goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(lib)
	return lib, nil
}

// ---- compute Dispatch (Layer A phase-1 completion: queue → buffers → encode → run) ----

var (
	selNewCommandQueue = objc.RegisterName("newCommandQueue")
	selNewFunctionName = objc.RegisterName("newFunctionWithName:")
	selNewPipelineFn   = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selNewBufferBytes  = objc.RegisterName("newBufferWithBytes:length:options:")
	selNewBufferNoCopy = objc.RegisterName("newBufferWithBytesNoCopy:length:options:deallocator:")
	selNewBufferLen    = objc.RegisterName("newBufferWithLength:options:")
	selContents        = objc.RegisterName("contents")
	selLength          = objc.RegisterName("length")        // MTLBuffer → NSUInteger, actual allocated bytes
	selAllocatedSize   = objc.RegisterName("allocatedSize") // MTLResource → NSUInteger, what Metal allocated for it (the ledger's charge)
	selCommandBuffer   = objc.RegisterName("commandBuffer")
	selComputeEncoder  = objc.RegisterName("computeCommandEncoder")
	selSetPipeline     = objc.RegisterName("setComputePipelineState:")
	selSetBuffer       = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBuffers      = objc.RegisterName("setBuffers:offsets:withRange:")
	selSetTgMem        = objc.RegisterName("setThreadgroupMemoryLength:atIndex:")
	selDispatchThreads = objc.RegisterName("dispatchThreads:threadsPerThreadgroup:")
	selDispatchTG      = objc.RegisterName("dispatchThreadgroups:threadsPerThreadgroup:")
	selEndEncoding     = objc.RegisterName("endEncoding")
	selCommit          = objc.RegisterName("commit")
	selWaitCompleted   = objc.RegisterName("waitUntilCompleted")
	selDrain           = objc.RegisterName("drain")
	selGPUStartTime    = objc.RegisterName("GPUStartTime")    // CFTimeInterval (double), post-completion
	selGPUEndTime      = objc.RegisterName("GPUEndTime")      // GPU busy window for this command buffer
	selKernelStartTime = objc.RegisterName("kernelStartTime") // incl scheduling; > GPU window
	selKernelEndTime   = objc.RegisterName("kernelEndTime")
	selStatus          = objc.RegisterName("status") // MTLCommandBuffer.status → MTLCommandBufferStatus (Int)
	selError           = objc.RegisterName("error")  // MTLCommandBuffer.error  → NSError* (nil on success)

	selNewSharedEvent     = objc.RegisterName("newSharedEvent")            // MTLDevice → id<MTLSharedEvent>
	selSignaledValue      = objc.RegisterName("signaledValue")             // MTLSharedEvent → uint64 (CPU read)
	selSetSignaledValue   = objc.RegisterName("setSignaledValue:")         // MTLSharedEvent uint64 (CPU signal)
	selEncodeSignalEvent  = objc.RegisterName("encodeSignalEvent:value:")  // MTLCommandBuffer: GPU sets value here
	selEncodeWaitForEvent = objc.RegisterName("encodeWaitForEvent:value:") // MTLCommandBuffer: GPU waits for value
)

// mtlSize mirrors MTLSize {NSUInteger width,height,depth}. At 24 bytes (>16) it is
// passed to objc_msgSend BY REFERENCE per AAPCS64 — so the call site passes
// unsafe.Pointer(&sz), which lands the pointer in the arg register exactly as the ABI
// wants. (Getting this wrong is the "MTLSize struct-by-value" hazard the doc flagged.)
type mtlSize struct{ w, h, d uint64 }

// Queue / Pipeline / Buffer are thin id wrappers.
type Queue struct {
	id  objc.ID
	act *activity // the device's activity count; nil counts nothing
}
type Pipeline struct{ id objc.ID }

// ThreadExecutionWidth is the pipeline's SIMD-group width — the number of threads that
// execute in lockstep and over which simd_sum/simd_shuffle reduce (32 on every Apple GPU,
// but read rather than assumed). A SIMD-group-per-row kernel launches N × this many threads
// and strides each row across the group; the host needs the width for that launch geometry,
// and it must match the kernel's [[threads_per_simdgroup]] since both are the hardware width.
func (p Pipeline) ThreadExecutionWidth() int {
	return int(objc.Send[uintptr](p.id, selThreadExecWidth))
}

type Buffer struct {
	id  objc.ID
	n   int     // element count (float32)
	off uintptr // byte offset for binding (a sub-view into a larger buffer; 0 = whole)
}

// At returns a view of the buffer bound at byteOff — for reading a slice of a combined
// buffer (e.g. q/k/v out of a fused QKV output). Zero-copy; only the bind offset changes.
func (b Buffer) At(byteOff int) Buffer { b.off = uintptr(byteOff); return b }

// Len reports the buffer's element count (float32s, or bytes for an int8 buffer) —
// mirrors cuda.go's Buffer.Len so a consumer can guard on an empty/optional buffer.
func (b Buffer) Len() int { return b.n }

// NewCommandQueue creates the command queue (built once, reused per token in the real backend).
func (d *Device) NewCommandQueue() Queue {
	q := d.id.Send(selNewCommandQueue)
	d.TrackObj(q) // +1-owned; released at Close (M24)
	return Queue{id: q, act: &d.act}
}

// NewComputePipeline looks a kernel up in a compiled library and builds its pipeline state.
func (d *Device) NewComputePipeline(lib objc.ID, fn string) (Pipeline, error) {
	f := lib.Send(selNewFunctionName, nsString(fn))
	if f == 0 {
		return Pipeline{}, fmt.Errorf("metal: no kernel %q in library", fn)
	}
	var nsErr objc.ID
	p := d.id.Send(selNewPipelineFn, f, unsafe.Pointer(&nsErr))
	f.Send(selRelease) // the pipeline state does not retain its MTLFunction — release it now (M24b)
	if p == 0 {
		return Pipeline{}, fmt.Errorf("metal: pipeline %q: %s", fn, goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(p) // +1-owned; released at Close (M24)
	return Pipeline{id: p}, nil
}

// MustBuf turns a FAILED MTLBuffer allocation into a loud panic instead of a silently
// zero-filled Buffer, and records the id so ReleaseAll can free it. objc returns nil on OOM and
// every constructor here used to keep it, so an out-of-memory condition surfaced as garbage
// numerics — an OOM wearing a parity bug's clothes. BuildResident recovers panics and declines
// to the CPU path, which is the right answer for OOM.
func (d *Device) MustBuf(id objc.ID, n int, what string) Buffer {
	return d.MustBufOn(UnlinedLine, id, n, what)
}

// MustBufOn is MustBuf leasing the buffer on plan line line: its length requested, its allocatedSize charged.
func (d *Device) MustBufOn(line string, id objc.ID, n int, what string) Buffer {
	return d.mustBuf(line, false, id, n, what)
}

func (d *Device) mustBuf(line string, fileMapped bool, id objc.ID, n int, what string) Buffer {
	if id == 0 {
		panic(fmt.Sprintf("metal: MTLBuffer allocation failed (%s, %d bytes) — out of memory", what, n))
	}
	release := func() {}
	length, allocated := int64(objc.Send[uintptr](id, selLength)), int64(objc.Send[uintptr](id, selAllocatedSize))
	var charged int64
	l := d.Leaser()
	if l != nil {
		charged = allocated
		release = l.Lease(fileMapped, line, length, charged)
	}
	if trace := AllocTraceForTest; trace != nil {
		d.mu.Lock()
		moved, traced := d.traceMoved, d.traceSet
		d.traceSet = false
		d.mu.Unlock()
		if !traced {
			moved = -1
		}
		trace(d, AllocTrace{ID: uintptr(id), FileMapped: fileMapped, Line: line, Length: length, Allocated: allocated,
			Charged: charged, Leased: l != nil, Moved: moved})
	}
	d.mu.Lock()
	d.allocs = append(d.allocs, devAlloc{id: id, release: release})
	d.mu.Unlock()
	return Buffer{id: id, n: n}
}

// ReleaseAll releases every MTLBuffer this Device handed out and empties the ledger. Callers MUST
// ensure no GPU work is still in flight against them (Resident.Close stops the executor and waits
// first) — releasing a buffer a live command buffer references is a use-after-free.
func (d *Device) ReleaseAll() {
	d.mu.Lock()
	allocs := d.allocs
	d.allocs = nil
	d.mu.Unlock()
	for _, a := range allocs {
		a.id.Send(selRelease)
		a.release()
	}
}

// BufferSizes is each MTLBuffer this Device still owns: its length and Metal's allocatedSize for it (the ledger's requested and
// charged bytes), for a test that reconciles the ledger against the device's own count. Held under the ledger's mutex.
func (d *Device) BufferSizes() (length, allocated []int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, a := range d.allocs {
		length = append(length, int64(objc.Send[uintptr](a.id, selLength)))
		allocated = append(allocated, int64(objc.Send[uintptr](a.id, selAllocatedSize)))
	}
	return length, allocated
}

// LedgerLen reports how many MTLBuffers and non-buffer objc objects this Device still owns —
// the observability hook for leak tests (Close/ReleaseAll must drive both to 0). Held under the
// same mutex the ledger mutations use.
func (d *Device) LedgerLen() (allocs, objs int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.allocs), len(d.objs)
}

// ReleaseBuf releases ONE MTLBuffer and removes it from the ledger, so a later ReleaseAll won't
// double-free it (C5). For per-call scratch that must not accumulate on the ledger until Close —
// the caller MUST ensure no in-flight GPU work references it (PrefillLast commits+waits before its
// deferred releases run). O(n) swap-remove scan of the ledger: fine for coarse per-request scratch,
// never a per-token path. A zero id (empty Buffer) is a no-op.
func (d *Device) ReleaseBuf(b Buffer) {
	if b.id == 0 {
		return
	}
	d.mu.Lock()
	for i, a := range d.allocs {
		if a.id == b.id {
			a.release()
			d.allocs[i] = d.allocs[len(d.allocs)-1]
			d.allocs = d.allocs[:len(d.allocs)-1]
			break
		}
	}
	d.mu.Unlock()
	if trace := FreeTraceForTest; trace != nil {
		before := d.CurrentAllocatedSize()
		b.id.Send(selRelease)
		trace(d, uintptr(b.id), int64(d.CurrentAllocatedSize())-int64(before))
		return
	}
	b.id.Send(selRelease)
}

// NewBufferLen allocates a shared MTLBuffer of nFloats float32s. Metal's newBufferWithLength returns ZERO-FILLED
// contents, so the buffer reads as zeros until written.
//
// Code shared across backends must NOT rely on this: zero-fill is a Metal property only (CUDA's NewBufferLen returns uninitialized memory), so a buffer that must start at zero is zeroed explicitly on every backend, not assumed.
func (d *Device) NewBufferLen(nFloats int) Buffer { return d.NewBufferLenOn(UnlinedLine, nFloats) }

// NewBufferLenOn is NewBufferLen leasing the buffer on plan line line.
func (d *Device) NewBufferLenOn(line string, nFloats int) Buffer {
	return d.MustBufOn(line, d.allocID(func() objc.ID { return d.id.Send(selNewBufferLen, uintptr(nFloats*4), uintptr(0)) }), nFloats, "len")
}

// mtlResourceHazardTrackingModeUntracked is MTLResourceHazardTrackingModeUntracked (1<<8) —
// Metal's own bit, not derived from anything in this binding.
const mtlResourceHazardTrackingModeUntracked = 1 << 8

// newBufferLenUntracked is NewBufferLen with hazard tracking disabled — the M-16 probe's own
// allocator (goinfer audit-metal-2026-09-12.md: every buffer this binding allocates is hazard-
// tracked by default, options=0, which the tree's own record prices as PART of a ~3.8 µs/dispatch
// GPU-side floor across a decode token's ~310 dispatches). MEASURED NEGATIVE
// (hazard_tracking_probe_test.go, interleaved runs): untracked vs tracked GPU-busy time is
// 99.6%/100.4%/100.4% across three runs — noise, not a lever. Kept unexported: it has no
// consumer, and giving a proven-dead-end 0.4%-of-nothing lever permanent exported surface is worse
// than re-adding two lines if this is ever revisited on different hardware.
func (d *Device) newBufferLenUntracked(line string, nFloats int) Buffer {
	return d.MustBufOn(line, d.allocID(func() objc.ID {
		return d.id.Send(selNewBufferLen, uintptr(nFloats*4), uintptr(mtlResourceHazardTrackingModeUntracked))
	}), nFloats, "len-untracked")
}

// NewBufferBytes allocates a shared MTLBuffer of n BYTES (n is the
// element count for the returned Buffer). Zero-filled on Metal, like NewBufferLen; backend-shared code must not rely on that (CUDA's NewBufferBytes
// is uninitialized). For consumers that size a buffer in raw
// bytes rather than float32s — the device-layer primitive goinfer's kernels reach
// for when re-pointed onto this substrate.
func (d *Device) NewBufferBytes(n int) Buffer { return d.NewBufferBytesOn(UnlinedLine, n) }

// NewBufferBytesOn is NewBufferBytes leasing the buffer on plan line line.
func (d *Device) NewBufferBytesOn(line string, n int) Buffer {
	return d.MustBufOn(line, d.allocID(func() objc.ID { return d.id.Send(selNewBufferLen, uintptr(n), uintptr(0)) }), n, "bytes")
}

// NewBufferNoCopy wraps caller-owned memory in a shared MTLBuffer WITHOUT copying
// (newBufferWithBytesNoCopy) — the resident-weights lever: alias an mmap'd .giw straight into
// Metal instead of uploading a second GB-scale copy. ptr MUST be page-aligned and nBytes SHOULD be
// a page multiple (Metal drops a trailing partial page); an mmap base satisfies both. The
// deallocator is nil — Metal never frees this memory; the caller owns the mapping and munmaps it
// (AND must keep it mapped for the buffer's whole life). n counts BYTES, like NewBufferBytes. Bind
// per-tensor sub-views with Buffer.At(byteOffset) over the single whole-mapping buffer.
func (d *Device) NewBufferNoCopy(ptr unsafe.Pointer, nBytes int) Buffer {
	return d.NewBufferNoCopyOn(UnlinedLine, ptr, nBytes)
}

// NewBufferNoCopyOn is NewBufferNoCopy leasing the alias on plan line line, as file-mapped memory: the mapping is the caller's.
func (d *Device) NewBufferNoCopyOn(line string, ptr unsafe.Pointer, nBytes int) Buffer {
	id := d.allocID(func() objc.ID { return d.id.Send(selNewBufferNoCopy, ptr, uintptr(nBytes), uintptr(0), uintptr(0)) })
	return d.mustBuf(line, true, id, nBytes, "nocopy")
}

// ---- generic buffer verbs (any scalar element type) ----
//
// A consumer with a wider mix of element types — a decode path allocating
// float32 activations, int32 indices, uint16 f16 scales and uint32 packed
// weights — wants the same four verbs parameterized instead of a method per
// type. These are those, and they are free functions for the same reason
// NewBufferNoCopy can't be one: Go has no generic methods. This used to be a
// type-suffixed method per element type (NewBufferFloats, NewBufferInt8,
// NewBufferU32, NewBufferUint32s, NewBufferU16s) with every real consumer
// already on the generic twin once cuda.go's collapse landed; the
// type-suffixed set was deleted here too, once nothing called it anymore.

// Scalar is the element type a generic buffer verb (NewBufferOf, Upload,
// Download, ...) can move — fixed-size numeric scalars only. Mirrors cuda.go's
// Scalar exactly: the two files are build-tag mutually exclusive (darwin vs
// linux) so this can't be shared code, but a consumer building against both
// backends needs the same set of allowed element types on each.
type Scalar interface {
	~int8 | ~uint8 |
		~int16 | ~uint16 |
		~int32 | ~uint32 |
		~int64 | ~uint64 |
		~float32 | ~float64
}

// asBytes reinterprets a slice of fixed-size scalars as the []byte the objc
// buffer calls below copy through. The caller must runtime.KeepAlive the
// source across the call. Mirrors cuda.go's asBytes exactly.
func asBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(z)))
}

// NewBufferOf allocates a shared MTLBuffer sized to data and copies it in. n
// for the returned Buffer is len(data), in elements of T. Panics on
// allocation failure (see MustBuf).
func NewBufferOf[T Scalar](d *Device, data []T) Buffer { return NewBufferOfOn(d, UnlinedLine, data) }

// NewBufferOfOn is NewBufferOf leasing the buffer on plan line line.
func NewBufferOfOn[T Scalar](d *Device, line string, data []T) Buffer {
	if len(data) == 0 {
		return d.MustBufOn(line, d.allocID(func() objc.ID { return d.id.Send(selNewBufferLen, uintptr(0), uintptr(0)) }), 0, "typed-empty")
	}
	b := asBytes(data)
	id := d.allocID(func() objc.ID {
		return d.id.Send(selNewBufferBytes, unsafe.Pointer(&b[0]), uintptr(len(b)), uintptr(0))
	})
	runtime.KeepAlive(data)
	return d.MustBufOn(line, id, len(data), "typed")
}

// NewBufferLenOf allocates a shared MTLBuffer of n elements of T. Zero-filled on Metal, like NewBufferLen; backend-shared code must not rely on that
// (CUDA's NewBufferLenOf is uninitialized).
func NewBufferLenOf[T Scalar](d *Device, n int) Buffer { return NewBufferLenOfOn[T](d, UnlinedLine, n) }

// NewBufferLenOfOn is NewBufferLenOf leasing the buffer on plan line line.
func NewBufferLenOfOn[T Scalar](d *Device, line string, n int) Buffer {
	var z T
	nBytes := n * int(unsafe.Sizeof(z))
	return d.MustBufOn(line, d.allocID(func() objc.ID { return d.id.Send(selNewBufferLen, uintptr(nBytes), uintptr(0)) }), n, "typed-len")
}

// capacityBytes queries the MTLBuffer's actual allocated byte length — the
// bounds-check source Upload/Download need. b.n is recorded in the
// constructor's unit (floats, int8s, u32s, …), not bytes, so a generic verb
// crossing that boundary (e.g. Download[float32] from a buffer NewBufferOf[int8]
// built) cannot trust it the way cuda.go's Buffer.n comment already warns.
// Mirrors cuda.go's b.b.Bytes().
func (b Buffer) capacityBytes() uintptr {
	return objc.Send[uintptr](b.id, selLength)
}

// Upload copies src into the buffer at its bind offset. On Metal's UMA this is
// an in-place copy into shared memory, not a real host→device transfer — but
// the verb matches cuda.go's so a consumer building against both backends
// reads the same vocabulary either way.
func Upload[T Scalar](b Buffer, src []T) error {
	if len(src) == 0 {
		return nil
	}
	data := asBytes(src)
	if avail, want := b.capacityBytes(), uint64(b.off)+uint64(len(data)); uint64(avail) < want {
		return fmt.Errorf("metal: upload of %d bytes at offset %d overruns a %d-byte buffer", len(data), b.off, avail)
	}
	dst := unsafe.Slice(objc.Send[*byte](b.id, selContents), int(b.capacityBytes()))
	copy(dst[b.off:], data)
	runtime.KeepAlive(src)
	return nil
}

// Download copies the buffer's contents at its bind offset into dst
// (zero-copy on UMA; dst's length sizes the transfer).
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(dst)].
func Download[T Scalar](b Buffer, dst []T) error {
	if len(dst) == 0 {
		return nil
	}
	var z T
	need := len(dst) * int(unsafe.Sizeof(z))
	if avail, want := b.capacityBytes(), uint64(b.off)+uint64(need); uint64(avail) < want {
		return fmt.Errorf("metal: download of %d bytes at offset %d overruns a %d-byte buffer", need, b.off, avail)
	}
	src := unsafe.Slice(objc.Send[*byte](b.id, selContents), int(b.capacityBytes()))
	copy(asBytes(dst), src[b.off:b.off+uintptr(need)])
	runtime.KeepAlive(dst)
	return nil
}

// Floats / Int8s view the buffer's shared contents as a Go slice (zero-copy on UMA).
func (b Buffer) Floats() []float32 {
	return unsafe.Slice(objc.Send[*float32](b.id, selContents), b.n)
}

func (b Buffer) Int8s() []int8 {
	return unsafe.Slice(objc.Send[*int8](b.id, selContents), b.n)
}

// U16s views the buffer's shared contents as []uint16 (f16 bits) — zero-copy on UMA.
func (b Buffer) U16s() []uint16 {
	return unsafe.Slice(objc.Send[*uint16](b.id, selContents), b.n)
}

// U32s views the buffer's shared contents as a []uint32 (zero-copy on UMA) — e.g. the MoE
// router's idx[k] output.
func (b Buffer) U32s() []uint32 {
	return unsafe.Slice(objc.Send[*uint32](b.id, selContents), b.n)
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
	// G22: pin the OS thread for the pool's whole lifetime. An NSAutoreleasePool is
	// PER-OS-THREAD, and Go may migrate a goroutine between any two calls — draining
	// a pool on a thread other than the one that pushed it is undefined behaviour,
	// and shows up as an intermittent SIGSEGV (fault 0x10) inside objc_msgSend.
	// Consumers that already pin are unaffected: LockOSThread nests.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

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
	mustCmdBufOK(cb, "dispatch aborted")
	q.act.ran(p, 1)
}

// Run2D encodes and runs a 2-D kernel over gx×gy THREADGROUPS of tgx×tgy threads each,
// via dispatchThreadgroups (UNIFORM whole threadgroups) — the shape a tiled GEMM needs.
// Unlike Run1D's dispatchThreads (non-uniform, exactly n threads, no bounds check),
// every thread in the final edge tiles must EXIST so the tile can cooperatively stage
// its A/B sub-tiles into threadgroup memory and all reach the barrier; the kernel then
// bounds-checks its own global (m,n,k) index, exactly like the CUDA tiled kernels.
// Blocks until the GPU finishes. Manual autoreleasepool discipline, like Run1D.
func (q Queue) Run2D(p Pipeline, gx, gy, tgx, tgy int, bufs ...Buffer) {
	// G22: pin the OS thread for the pool's whole lifetime. An NSAutoreleasePool is
	// PER-OS-THREAD, and Go may migrate a goroutine between any two calls — draining
	// a pool on a thread other than the one that pushed it is undefined behaviour,
	// and shows up as an intermittent SIGSEGV (fault 0x10) inside objc_msgSend.
	// Consumers that already pin are unaffected: LockOSThread nests.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)

	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	grid := mtlSize{w: uint64(gx), h: uint64(gy), d: 1}
	perTG := mtlSize{w: uint64(tgx), h: uint64(tgy), d: 1}
	enc.Send(selDispatchTG, unsafe.Pointer(&grid), unsafe.Pointer(&perTG))
	runtime.KeepAlive(&grid)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
	mustCmdBufOK(cb, "dispatch aborted")
	q.act.ran(p, 1)
}

// maxBindBuffers is Metal's limit on the entries of the buffer argument table, per kernel function: 31, the same
// on every GPU family (Apple2 through Apple10). Source: Apple's "Metal feature set tables", section "GPU
// implementation limits by family", row "Maximum number of entries in the buffer argument table, per graphics or
// kernel function" (edition of May 21, 2026). Encoder's scratch is sized to it, so every binding Metal itself
// accepts fits, and bindBuffers refuses one more with a message rather than a bare index-out-of-range.
const maxBindBuffers = 31

// Encoder batches many DIFFERENT kernel dispatches into ONE command buffer — the
// per-token shape the decode loop needs (whole layer stack → one commit/wait, per the
// tax finding). The default serial compute encoder inserts barriers between dependent
// dispatches (like WGSL's storage barriers), so chained kernels see prior writes.
type Encoder struct {
	pool objc.ID
	// pinned records that Begin() locked this goroutine to its OS thread for the
	// pool's lifetime (G22); End() releases it after the drain. False for BeginNP,
	// which has no pool of its own.
	pinned bool
	cb     objc.ID
	enc    objc.ID
	// Encode-tax trims (measured: together only ~0.5ms of the ~2.4ms/token overhead — the
	// rest is GPU-side per-Dispatch latency, not Go-side msgSend): skip setComputePipelineState
	// when the pipeline is unchanged, and batch all buffer binds into ONE setBuffers call.
	curPipe objc.ID
	idScr   [maxBindBuffers]objc.ID // reusable C-arrays for batched setBuffers:offsets:withRange:
	offScr  [maxBindBuffers]uintptr
	// GPU timing of the last committed command buffer (seconds), captured in end() after
	// waitUntilCompleted (valid post-completion) and before the pool drains the cb.
	gpuStart, gpuEnd, kernStart, kernEnd float64
	// err is the command-buffer abort latched at WaitDone/End — read from the cb's status/error
	// BEFORE the autorelease pool drains (and possibly frees) the cb, so Err() is a safe getter.
	err error
	// act is the device's activity count, and hook the dispatch hook it held when this encoder began (SB11).
	act  *activity
	hook func(Pipeline)
}

func (q Queue) Begin() *Encoder {
	// G22: this pool is drained in End(), a DIFFERENT call, so the pin must span the
	// encoder's lifetime rather than one function. Unpinned, Go can migrate the
	// goroutine in between and End() drains on the wrong thread — undefined
	// behaviour, observed as an intermittent SIGSEGV (fault 0x10) in objc_msgSend
	// whose crash site moves between runs.
	//
	// The pairing is the contract: an Encoder from Begin() MUST reach End(), and
	// must not be handed to another goroutine. BeginNP() takes no pool of its own
	// and is the API for callers that own a longer-lived pool (and their own pin).
	runtime.LockOSThread()
	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	cb := q.id.Send(selCommandBuffer)
	return &Encoder{pool: pool, pinned: true, cb: cb, enc: cb.Send(selComputeEncoder), act: q.act, hook: q.act.dispatchHook()}
}

// ARPool is an NSAutoreleasePool handle — the pipelined executor owns one long-lived pool
// (drained periodically) so encode-ahead can hold an un-committed command buffer across the
// loop without per-call pool nesting.
type ARPool struct{ id objc.ID }

func NewARPool() ARPool {
	return ARPool{id: objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)}
}
func (p ARPool) Drain() { p.id.Send(selDrain) }

// BeginNP creates a command buffer + encoder with NO per-call pool — the cb/encoder autorelease
// into whatever pool is active on the calling thread. Used by the pipelined executor, which
// owns one long-lived pool drained periodically (so encode-ahead can hold an un-committed next
// command buffer without the LIFO pool-nesting a per-call pool would cause).
func (q Queue) BeginNP() *Encoder {
	cb := q.id.Send(selCommandBuffer)
	return &Encoder{cb: cb, enc: cb.Send(selComputeEncoder), act: q.act, hook: q.act.dispatchHook()}
}

// FinishEncoding closes the encoder (ready to commit) without committing — the split that lets
// the executor encode token t+1 while token t's command buffer runs.
func (e *Encoder) FinishEncoding() { e.enc.Send(selEndEncoding) }
func (e *Encoder) Commit()         { e.cb.Send(selCommit); e.act.submitted() }

// WaitDone blocks until the committed command buffer completes, then captures any abort and reads
// its GPU timestamps. Both reads are valid only post-completion and are taken here, while the cb is
// still alive — a later pool drain may free it.
func (e *Encoder) WaitDone() {
	e.cb.Send(selWaitCompleted)
	e.act.waited()
	e.captureErr()
	e.ReadTimes()
}

// captureErr latches the command buffer's terminal status/error into e.err. MUST be called after
// waitUntilCompleted and BEFORE the autorelease pool drains the cb (reading a drained cb is a
// use-after-free). waitUntilCompleted returns cleanly even on a GPU fault, so this is the only
// signal that a kernel aborted (goinfer audit C-09).
func (e *Encoder) captureErr() {
	// Integer returns come back in x0 on arm64; objc.Send[uintptr] is the integer-return path this
	// file already uses for enum getters (selLanguageVersion, selMathMode, selRespondsToSel).
	if int(objc.Send[uintptr](e.cb, selStatus)) != mtlCmdBufStatusError {
		return
	}
	e.err = cmdBufError(e.cb.Send(selError)) // NSError* — non-nil once status is Error
}

// MTLCommandBufferStatus values: NotEnqueued 0, Enqueued 1, Committed 2, Scheduled 3,
// Completed 4, Error 5. Only Error means the buffer aborted (a GPU fault, a threadgroup-memory
// over-budget dispatch, etc.).
const mtlCmdBufStatusError = 5

// Err reports whether the last committed command buffer aborted. Valid AFTER WaitDone / End, which
// latch the status while the cb is still alive. waitUntilCompleted returns cleanly even when a
// kernel faults, so without this the host reads stale / previous-token results with no signal
// (goinfer audit C-09): callers MUST consult Err before trusting the outputs of a committed buffer.
func (e *Encoder) Err() error { return e.err }

// cmdBufError formats an aborted command buffer's NSError (or a nil NSError) into a Go error. Split
// out so the NSError→string path is unit-testable with a synthetic NSError: on Apple silicon a
// real GPU abort is nearly impossible to provoke on demand (the hardware silently tolerates OOB
// writes, unmapped-address stores, over-budget threadgroup memory, and over-max dispatches — every
// such command buffer still reports status Completed), which is exactly why this status check must
// exist as the safety net for the strict OS/GPU cases where a fault DOES set status Error.
func cmdBufError(nsErr objc.ID) error {
	if nsErr == 0 {
		return errors.New("metal: command buffer aborted")
	}
	return fmt.Errorf("metal: command buffer aborted: %s", goString(nsErr.Send(selLocalizedDesc)))
}

// mustCmdBufOK panics if the command buffer aborted. The fire-and-forget Queue.Run*
// helpers commit and waitUntilCompleted and return nothing, and waitUntilCompleted
// returns CLEANLY on a GPU fault — so without this a dispatch that aborted (a GPU
// fault, or threadgroup scratch over MaxThreadgroupMemoryLength) left the output
// buffer holding whatever it held before and the caller read it as a result. That
// is the silent-garbage path audit C-02 found in the ViT attention kernel above
// 8192 patches; CUDA's analogue fails the launch with an error.
//
// It panics rather than returning an error because the Run* helpers return nothing
// and giving them errors would break every caller. A loud panic matches MustBuf,
// which is this package's existing answer to an unrecoverable device failure, and
// composes with the callers that already recover it into an error (annmetal's three
// sites, audit C-04). MUST be called after waitUntilCompleted and before the
// autorelease pool drains the cb — reading a drained cb is a use-after-free.
func mustCmdBufOK(cb objc.ID, what string) {
	if int(objc.Send[uintptr](cb, selStatus)) != mtlCmdBufStatusError {
		return
	}
	panic(fmt.Sprintf("metal: %s: %v", what, cmdBufError(cb.Send(selError))))
}

// ReadTimes reads the command buffer's GPU/kernel timestamps (valid post-completion).
func (e *Encoder) ReadTimes() {
	// On arm64 objc_msgSend returns the double in d0 — objc.Send[float64] uses the fp-return path.
	e.gpuStart = objc.Send[float64](e.cb, selGPUStartTime)
	e.gpuEnd = objc.Send[float64](e.cb, selGPUEndTime)
	e.kernStart = objc.Send[float64](e.cb, selKernelStartTime)
	e.kernEnd = objc.Send[float64](e.cb, selKernelEndTime)
}

// bindBuffers binds bufs at indices 0..len-1 of the current compute encoder in ONE setBuffers:offsets:withRange:
// msgSend (vs one per buffer): the aggressive Go-side-encode-cost probe. NSRange{location,length} lowers to two
// trailing word args. Allocation-free: this is the per-dispatch hot path, hundreds of dispatches per token.
//
// More than maxBindBuffers is a programming error in the kernel's binding list, not something to recover from, so
// it panics. Pipeline carries no name, so the stack trace is what identifies the call site.
func (e *Encoder) bindBuffers(bufs []Buffer) {
	nb := len(bufs)
	if nb > maxBindBuffers {
		panic(fmt.Sprintf("gpu: dispatch binds %d buffers, over Metal's limit of %d buffer-argument-table entries per kernel function; the kernel must bind fewer buffers", nb, maxBindBuffers))
	}
	for i, b := range bufs {
		e.idScr[i], e.offScr[i] = b.id, b.off
	}
	e.enc.Send(selSetBuffers, unsafe.Pointer(&e.idScr[0]), unsafe.Pointer(&e.offScr[0]), uintptr(0), uintptr(nb))
	runtime.KeepAlive(&e.idScr)
	runtime.KeepAlive(&e.offScr)
}

// Dispatch encodes one kernel over n threads (threadgroup width tg, clamped ≤ n), binding
// bufs at indices 0..len-1. dispatchThreads launches EXACTLY n threads (non-uniform), so
// no out-of-range writes.
func (e *Encoder) Dispatch(p Pipeline, n, tg int, bufs ...Buffer) {
	if tg > n {
		tg = n
	}
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	e.bindBuffers(bufs)
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	e.counted(p)
	if e.hook != nil && DispatchTwiceForTest.CompareAndSwap(true, false) {
		e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
		e.act.dispatches.Add(1)
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
}

// Dispatch2D encodes a 2-D dispatchThreadgroups into this command buffer — the
// tiled-GEMM shape, which Dispatch's 1-D dispatchThreads cannot express (audit
// M-13). Without it a ViT layer could not be batched: every GEMM had to be its
// own Queue.Run2D, i.e. its own commit and wait.
//
// Note the grid unit differs from Dispatch's. dispatchThreads takes a THREAD
// count and the driver derives the groups; dispatchThreadgroups takes a
// THREADGROUP count, so gx/gy are already divided by tgx/tgy — the same
// convention Queue.Run2D uses, and what gpu.TileDims returns.
func (e *Encoder) Dispatch2D(p Pipeline, gx, gy, tgx, tgy int, bufs ...Buffer) {
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	e.bindBuffers(bufs)
	grid := mtlSize{w: uint64(gx), h: uint64(gy), d: 1}
	perTG := mtlSize{w: uint64(tgx), h: uint64(tgy), d: 1}
	e.enc.Send(selDispatchTG, unsafe.Pointer(&grid), unsafe.Pointer(&perTG))
	e.counted(p)
	runtime.KeepAlive(&grid)
	runtime.KeepAlive(&perTG)
}

// DispatchTG is Dispatch with dynamic threadgroup memory of tgBytes at index 0 — for the Stage A
// GEMV kernels whose activation-staging scratch is sized to K per call (so a small-K model keeps
// full occupancy while a large-K one, e.g. Qwen3 o-proj K=nHhd>1536, gets the room it needs).
func (e *Encoder) DispatchTG(p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	if tg > n {
		tg = n
	}
	if p.id != e.curPipe {
		e.enc.Send(selSetPipeline, p.id)
		e.curPipe = p.id
	}
	e.bindBuffers(bufs)
	e.enc.Send(selSetTgMem, uintptr(tgBytes), uintptr(0))
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	e.enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	e.counted(p)
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
}

func (e *Encoder) End() {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selCommit)
	e.act.submitted()
	e.cb.Send(selWaitCompleted)
	e.act.waited()
	e.captureErr() // BEFORE the drain frees the cb (C-09)
	e.ReadTimes()
	e.pool.Send(selDrain)
	if e.pinned { // G22: released only AFTER the drain, never before
		e.pinned = false
		runtime.UnlockOSThread()
	}
}

// SharedEvent is an MTLSharedEvent — a monotonic counter both the GPU (encodeSignalEvent /
// encodeWaitForEvent inside a command buffer) and the CPU (Value / SetValue) can read and advance.
// It exists to let the host observe per-layer progress WITHOUT tearing a token into one command
// buffer per layer: encode the whole trunk once, have the GPU signal after each layer's router and
// wait for a CPU ack before the expert dispatches, and the CPU stage the routed experts in between —
// one submit per token, with per-layer host intervention. This is the Metal-idiomatic alternative to
// per-layer submit+wait for the Gemma-4 MoE expert-paging path; the Step-6 Step-0 probe prices it.
type SharedEvent struct{ id objc.ID }

// NewSharedEvent creates a shared event (initial signaledValue 0).
func (d *Device) NewSharedEvent() SharedEvent { return SharedEvent{id: d.id.Send(selNewSharedEvent)} }

// Value reads the event's current signaledValue (CPU side). SetValue advances it (CPU signal to the
// GPU's encodeWaitForEvent). Both are the documented CPU⇄GPU shared-event handshake ops on UMA.
func (ev SharedEvent) Value() uint64     { return objc.Send[uint64](ev.id, selSignaledValue) }
func (ev SharedEvent) SetValue(v uint64) { ev.id.Send(selSetSignaledValue, uintptr(v)) }

// EventBoundary closes the current compute segment, has the GPU signal `sig` (so a spinning CPU sees
// this layer finished), then has the GPU wait for `wait` (blocking the next segment until the CPU
// acks), then opens a fresh compute segment. curPipe is reset so the next Dispatch re-binds its
// pipeline. Called between encodeLayer calls to build the one-command-buffer handshake trunk.
func (e *Encoder) EventBoundary(ev SharedEvent, sig, wait uint64) {
	e.enc.Send(selEndEncoding)
	e.cb.Send(selEncodeSignalEvent, ev.id, uintptr(sig))
	e.cb.Send(selEncodeWaitForEvent, ev.id, uintptr(wait))
	e.enc = e.cb.Send(selComputeEncoder)
	e.curPipe = 0
}

// DrainPool drains the Encoder's autorelease pool — the tail of a hand-built command buffer (e.g.
// the shared-event handshake) that uses FinishEncoding + Commit + WaitDone directly instead of End()
// because the caller runs CPU handshake work between Commit and WaitDone.
func (e *Encoder) DrainPool() { e.pool.Send(selDrain) }

// Run1DBatch encodes `reps` dispatches of the same kernel into ONE command buffer and
// submits once — the shape a real token uses (a whole layer stack encoded into one
// command buffer, one commit/wait). Isolates the per-commit round-trip tax (reps=1)
// from the marginal per-encoded-Dispatch msgSend cost (the slope over reps).
func (q Queue) Run1DBatch(p Pipeline, n, tg, reps int, bufs ...Buffer) {
	// G22: pin the OS thread for the pool's whole lifetime. An NSAutoreleasePool is
	// PER-OS-THREAD, and Go may migrate a goroutine between any two calls — draining
	// a pool on a thread other than the one that pushed it is undefined behaviour,
	// and shows up as an intermittent SIGSEGV (fault 0x10) inside objc_msgSend.
	// Consumers that already pin are unaffected: LockOSThread nests.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

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
	for range reps {
		enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
	mustCmdBufOK(cb, "dispatch aborted")
	q.act.ran(p, reps)
}

// Run1DBatchTG is Run1DBatch with dynamic threadgroup memory of tgBytes at index 0 (for kernels
// with a `threadgroup T* x [[threadgroup(0)]]` param sized per-Dispatch — batch-k stages only
// what k needs, preserving occupancy).
func (q Queue) Run1DBatchTG(p Pipeline, n, tg, reps, tgBytes int, bufs ...Buffer) {
	// G22: pin the OS thread for the pool's whole lifetime. An NSAutoreleasePool is
	// PER-OS-THREAD, and Go may migrate a goroutine between any two calls — draining
	// a pool on a thread other than the one that pushed it is undefined behaviour,
	// and shows up as an intermittent SIGSEGV (fault 0x10) inside objc_msgSend.
	// Consumers that already pin are unaffected: LockOSThread nests. Missed here
	// originally — every sibling helper (Run1D, Run2D, Run1DBatch) pins; this one
	// didn't, and Run1DTG delegates straight to it (goinfer audit-metal-2026-09-12.md
	// C-05; qwenmetal.ForwardViT's own caller-side pin masked it in production).
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
	defer pool.Send(selDrain)
	cb := q.id.Send(selCommandBuffer)
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	for i, b := range bufs {
		enc.Send(selSetBuffer, b.id, b.off, uintptr(i))
	}
	enc.Send(selSetTgMem, uintptr(tgBytes), uintptr(0))
	total := mtlSize{w: uint64(n), h: 1, d: 1}
	perTG := mtlSize{w: uint64(tg), h: 1, d: 1}
	for range reps {
		enc.Send(selDispatchThreads, unsafe.Pointer(&total), unsafe.Pointer(&perTG))
	}
	runtime.KeepAlive(&total)
	runtime.KeepAlive(&perTG)
	enc.Send(selEndEncoding)
	cb.Send(selCommit)
	cb.Send(selWaitCompleted)
	mustCmdBufOK(cb, "dispatch aborted")
	q.act.ran(p, reps)
}

// Run1DTG runs ONE 1-D dispatch over n threads (threadgroup width tg) with tgBytes of
// dynamic threadgroup memory bound at index 0 — Run1D plus a `threadgroup T*
// [[threadgroup(0)]]` param (the ViT attention kernel stages its per-query score row
// there). It is Run1DBatchTG with reps=1, the shape the encoder kernels want.
func (q Queue) Run1DTG(p Pipeline, n, tg, tgBytes int, bufs ...Buffer) {
	q.Run1DBatchTG(p, n, tg, 1, tgBytes, bufs...)
}

// GPUStart returns the last committed command buffer's GPU start timestamp; GPUEnd,
// KernStart and KernEnd are the siblings covering the rest of the GPU/kernel
// timestamps (seconds), valid after WaitDone/End. Accessors, since the fields are unexported.
func (e *Encoder) GPUStart() float64  { return e.gpuStart }
func (e *Encoder) GPUEnd() float64    { return e.gpuEnd }
func (e *Encoder) KernStart() float64 { return e.kernStart }
func (e *Encoder) KernEnd() float64   { return e.kernEnd }
