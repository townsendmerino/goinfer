//go:build linux

// cuda.go is the CUDA half of aikit's cgo-free GPU device substrate — the Linux
// mirror of metal.go (docs/task-native-gpu.md, Phase 1b). It presents the SAME
// vocabulary as the Metal impl — Device / Buffer / Queue / Pipeline / Encoder,
// CreateSystemDefaultDevice, NewBuffer*, Run1D, Begin/Dispatch/End, and the
// allocation ledger (ReleaseAll / ReleaseBuf / ReleaseObjects / LedgerLen) — so a
// consumer reads the same on both platforms. The two files are build-tag mutually
// exclusive (darwin vs linux), so the shared type names never collide.
//
// It is a thin wrapper over github.com/eitamring/gocudrv, which dlopens libcuda
// and is cgo-free by construction (CGO_ENABLED=0 throughout). No aikit imports —
// the device layer stays ann-free, exactly like metal.go.
//
// # Three places the CUDA surface deliberately diverges from Metal
//
//  1. **Host memory is not device memory.** Apple's UMA lets an MTLBuffer's
//     contents be a zero-copy Go slice, so metal.go's Floats/Int8s/SetU32 both
//     read AND write device memory in place. A discrete NVIDIA GPU has no such
//     mapping, so faking those views would silently drop writes. Transfers are
//     therefore EXPLICIT here: Upload/SetU32 upload, Download reads back into
//     caller-owned storage. Same buffer constructors, explicit traffic.
//
//  2. **Dispatch launches whole blocks.** Metal's dispatchThreads launches
//     EXACTLY n threads, so its kernels need no bounds check. CUDA launches
//     ceil(n/tg) blocks of tg threads, so the tail block overruns n. Every kernel
//     run through this layer MUST guard its global index against its own count
//     parameter — see gpu/anncuda's kernels for the shape.
//
//  3. **Failures are returned, not swallowed.** Metal's Run1D/Dispatch/End are
//     infallible (objc msgSend has nothing to report); cuLaunchKernel does. These
//     return error rather than let a failed launch read back as a buffer of zeros
//     — an OOM or a bad grid wearing a parity bug's clothes.
//
// # Thread affinity
//
// A CUDA context is bound to an OS thread, so a per-call dispatch from a
// migrating goroutine is an intermittent-crash generator — the same class of bug
// NSAutoreleasePool caused on the Metal side (see gpu/annmetal's runLocked).
// Nothing here needs runtime.LockOSThread: gocudrv's Context OWNS a dedicated
// runtime.LockOSThread'd executor goroutine and funnels every driver call
// (alloc, memcpy, launch, sync) through it. The affinity guarantee is structural,
// which is why this file has no pinning of its own.

package gpudev

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudasys"
)

// bg is the context.Context every gocudrv call takes. The driver calls this layer
// makes are short and already serialized on the context's pinned executor, so
// there is nothing to cancel; a real deadline belongs to the caller's request, not
// to a memcpy.
var bg = context.Background()

// Device wraps a CUDA primary context and OWNS every allocation made through it.
//
// The ledger mirrors metal.go's: gocudrv requires every Buffer be closed before
// its Context (a buffer outliving the context cannot reach the executor to free
// itself), so tracking allocations centrally is what makes "release a resident
// model" a single call. Modules and streams are tracked alongside as objs.
type Device struct {
	cx  *gc.Context
	dev *gc.Device

	mu     sync.Mutex
	allocs []devAlloc // every device allocation handed out, for ReleaseAll
	objs   []closer   // modules + streams, for ReleaseObjects
	hosts  map[closer]func()
	leaser Leaser // every allocation's lease (MM3); nil leases nothing
	// launches and waits count every kernel launch and every host wait on the device (Activity), always on: the device-level
	// count SB11's per-kernel counts are checked against.
	launches, waits atomic.Int64
}

// noteWait counts one host wait on d (nil: a Buffer built without a device, which counts nowhere).
func (d *Device) noteWait() {
	if d != nil {
		d.waits.Add(1)
	}
}

// PCIBusID is the device's PCI bus id ("0000:01:00.0"), which names the same GPU to nvidia-smi whatever CUDA's ordinal order and
// CUDA_VISIBLE_DEVICES make of it.
func (d *Device) PCIBusID() (string, error) {
	if d.dev == nil {
		return "", fmt.Errorf("cuda: PCIBusID on a released device")
	}
	return d.dev.PCIBusID()
}

// Activity is how many kernel launches and host waits (a stream or event sync, a blocking copy to or from the host) the device
// has made since it was created.
func (d *Device) Activity() (launches, waits int64) { return d.launches.Load(), d.waits.Load() }

// devAlloc is one device allocation and its lease's release.
type devAlloc struct {
	b       *gc.Buffer[uint8]
	release func()
}

// Leaser is told of every allocation the device makes (a device buffer, or pinned host memory when onDevice is false): its plan
// line, the bytes asked for and the bytes the allocator takes. It returns the release, called once when the memory is freed. The
// backend attaches it (SetLeaser) when it creates the device; this package keeps no account of its own.
type Leaser interface {
	Lease(onDevice bool, line string, requested, charged int64) (release func())
}

// NewLeaser, when set, gives every device CreateSystemDefaultDevice creates its leaser: the backend sets it at init, so a device
// any package creates is leased.
var NewLeaser func() Leaser

// SetLeaser attaches l; allocations after it are leased to it.
func (d *Device) SetLeaser(l Leaser) {
	d.mu.Lock()
	d.leaser = l
	d.mu.Unlock()
}

// Leaser is the attached leaser, or nil.
func (d *Device) Leaser() Leaser {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.leaser
}

func (d *Device) lease(onDevice bool, line string, requested, charged int64) func() {
	d.mu.Lock()
	l := d.leaser
	d.mu.Unlock()
	if l == nil {
		return func() {}
	}
	return l.Lease(onDevice, line, requested, charged)
}

// AllocQuantumBytes is the driver's allocation granularity, MEASURED (cuda/allocgran_test.go: 5 MiB -> 6, 6 -> 6, 9 -> 10, so
// 2 MiB granular and not next-power-of-two).
const AllocQuantumBytes = 2 << 20

// ChargedBytes is what cuMemAlloc takes for a buffer of n bytes: one of a quantum or more rounds up to the next quantum; a smaller
// one is charged at its size (the driver packs it into a shared page; docs/code-notes/cuda.md#allocRoundSlack). The plan's
// rounding slack is this rule's excess, so the plan and the ledger price one figure.
func ChargedBytes(n int64) int64 {
	if n < AllocQuantumBytes {
		return n
	}
	return (n + AllocQuantumBytes - 1) / AllocQuantumBytes * AllocQuantumBytes
}

// UnlinedLine is the plan line an allocation through an unlined constructor (MustBuf, NewBufferLenOf, NewHostBuffer, …) leases
// on: empty, so unplanned. Production code calls the lined constructors (a source scan in the cuda module holds that); a test
// package may set it to its test line for its own buffers.
var UnlinedLine string

// ReleasedForTest, when set, is called by ReleaseObjects after everything the device tracked is freed, so a test can read the
// leaser's account then (I1, I2). A test seam, nil in production.
var ReleasedForTest func(*Device)

// closer is the shared shape of the non-buffer resources the Device tracks
// (gc.Module, gc.Stream): both free driver-owned handles at Close.
type closer interface{ Close() error }

// CreateSystemDefaultDevice initializes the driver and retains device 0's primary
// context — the CUDA analogue of MTLCreateSystemDefaultDevice, and named the same
// so consumer code is platform-agnostic. Returns an error (not a panic) when there
// is no libcuda, no device, or no context: callers degrade to the CPU path.
func CreateSystemDefaultDevice() (*Device, error) {
	if err := gc.Init(); err != nil {
		return nil, fmt.Errorf("cuda: driver init: %w", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		return nil, fmt.Errorf("cuda: GetDevice(0): %w", err)
	}
	cx, err := dev.Primary()
	if err != nil {
		return nil, fmt.Errorf("cuda: retain primary context: %w", err)
	}
	d := &Device{cx: cx, dev: dev}
	if NewLeaser != nil {
		d.leaser = NewLeaser()
	}
	return d, nil
}

// Name is the device's product name (e.g. "NVIDIA GeForce RTX 2070 SUPER"), or ""
// if the driver cannot report it. Mirrors metal.go's Name — diagnostics only, so a
// query failure degrades to the empty string rather than complicating the signature.
func (d *Device) Name() string {
	n, err := d.dev.Name()
	if err != nil {
		return ""
	}
	return n
}

// SMCount reports the device's multiprocessor count, or 0 if the driver will not say.
//
// It exists so a launch can size its grid against THIS device rather than against a
// constant measured on one. A kernel that puts one block on each unit of work is
// starved whenever that work count is below the SM count — gpu/anncuda's top-k is
// exactly that shape, and splits each query across several blocks below this threshold.
// Hardcoding 40 (an RTX 2070 SUPER) would be the same mistake as the GEMM benchmark
// that divided by another machine's peak for months, or gemmTileMinM, which was correct
// when written and silently wrong once the kernel it was compared against got faster.
//
// Callers must tolerate 0: it means "unknown", not "no SMs".
func (d *Device) SMCount() int {
	n, err := d.dev.Attribute(gc.DeviceAttributeMultiprocessorCount)
	if err != nil || n <= 0 {
		return 0
	}
	return int(n)
}

// MaxSharedBytesPerBlockOptIn is the most shared memory, static plus dynamic, one block may use once its kernel opts in
// (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN): 64 KiB on Turing, more on later parts. 0 means unknown. A kernel's
// dynamic opt-in is this less the kernel's own static shared memory (Pipeline.MaxDynamicSharedBytes).
func (d *Device) MaxSharedBytesPerBlockOptIn() int {
	n, err := d.dev.Attribute(gc.DeviceAttribute(97)) // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
	if err != nil || n <= 0 {
		return 0
	}
	return int(n)
}

// Context exposes the underlying gocudrv context, for consumers that need driver
// surface this layer does not wrap (events, graphs, cooperative launch). goinfer's
// tuned decode kernels are the intended caller when they re-point onto this layer.
func (d *Device) Context() *gc.Context { return d.cx }

// TrackObj records a driver-owned handle (module, stream) so ReleaseObjects frees
// it at Close. Mirrors metal.go's TrackObj.
func (d *Device) TrackObj(c closer) {
	if c == nil {
		return
	}
	d.mu.Lock()
	d.objs = append(d.objs, c)
	d.mu.Unlock()
}

// untrackObj drops a tracked handle from the ledger, for a resource the caller
// closes itself (HostBuffer.Close) — so ReleaseObjects neither double-frees it nor
// leaves LedgerLen reporting a resource that is already gone.
func (d *Device) untrackObj(c closer) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if rel := d.hosts[c]; rel != nil {
		rel()
		delete(d.hosts, c)
	}
	for i, o := range d.objs {
		if o == c {
			d.objs[i] = d.objs[len(d.objs)-1]
			d.objs = d.objs[:len(d.objs)-1]
			return
		}
	}
}

// ReleaseAll frees every device allocation this Device handed out and empties the
// ledger. Callers MUST ensure no launch is still in flight against them (releasing
// memory a running kernel reads is a use-after-free). Idempotent.
func (d *Device) ReleaseAll() {
	d.mu.Lock()
	bufs := d.allocs
	d.allocs = nil
	d.mu.Unlock()
	for _, a := range bufs {
		_ = a.b.Close()
		a.release()
	}
}

// ReleaseObjects frees the modules and streams this Device tracked, then releases
// the primary context — the CUDA counterpart of metal.go's ReleaseObjects.
//
// It drains the buffer ledger FIRST (via ReleaseAll) because gocudrv cannot free a
// Buffer once its Context is closed: the free has to reach the context's executor.
// Metal has no such ordering constraint, so the two-defer idiom that is merely
// tidy there (`defer d.ReleaseObjects(); defer d.ReleaseAll()`) is load-bearing
// here — calling ReleaseAll internally makes the correct order unconditional
// rather than something every caller has to remember. Idempotent: it nils the
// context, so a second call (or a double Close) is a no-op.
func (d *Device) ReleaseObjects() {
	d.ReleaseAll()
	d.mu.Lock()
	objs, hosts := d.objs, d.hosts
	d.objs, d.hosts = nil, nil
	cx := d.cx
	d.cx = nil
	d.mu.Unlock()
	for _, o := range objs {
		_ = o.Close()
		if rel := hosts[o]; rel != nil {
			rel()
		}
	}
	if cx != nil {
		_ = cx.Close()
	}
	if ReleasedForTest != nil && cx != nil {
		ReleasedForTest(d)
	}
}

// ReleaseBuf frees ONE allocation and removes it from the ledger, so a later
// ReleaseAll won't double-free it. For per-call scratch that must not accumulate
// until Close — the caller MUST ensure no in-flight launch references it. O(n)
// swap-remove scan: fine for coarse per-request scratch, never a per-token path.
// A zero Buffer is a no-op. Mirrors metal.go's ReleaseBuf.
func (d *Device) ReleaseBuf(b Buffer) {
	if b.b == nil {
		return
	}
	d.mu.Lock()
	for i, x := range d.allocs {
		if x.b == b.b {
			x.release()
			d.allocs[i] = d.allocs[len(d.allocs)-1]
			d.allocs = d.allocs[:len(d.allocs)-1]
			break
		}
	}
	d.mu.Unlock()
	_ = b.b.Close()
}

// LedgerLen reports how many allocations and non-buffer objects this Device still
// owns — the observability hook for leak tests (ReleaseObjects must drive both to
// 0). Mirrors metal.go's LedgerLen.
func (d *Device) LedgerLen() (allocs, objs int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.allocs), len(d.objs)
}

// ---- modules + pipelines ----

// Library is a loaded PTX module — the CUDA counterpart of an MTLLibrary.
type Library struct{ m *gc.Module }

// CompileLibrary hands `ptx` to the driver's JIT (cuModuleLoadData) and returns the
// module. It is the CUDA counterpart of metal.go's CompileLibrary, with the one
// unavoidable signature change: Metal compiles MSL *source* at run time, whereas
// the cgo-free CUDA path ships PTX built ahead of time (build_ptx.sh → NVRTC →
// embedded with go:embed) so the runtime needs no CUDA toolkit — only libcuda.
// That also means
// there is no languageVersion landmine to defuse on this side.
func (d *Device) CompileLibrary(ptx []byte) (Library, error) {
	if d.cx == nil {
		return Library{}, fmt.Errorf("cuda: CompileLibrary on a released device")
	}
	m, err := d.cx.LoadModule(ptx)
	if err != nil {
		return Library{}, fmt.Errorf("cuda: LoadModule: %w", err)
	}
	d.TrackObj(m)
	return Library{m: m}, nil
}

// Pipeline is a kernel entry point — the CUDA counterpart of an
// MTLComputePipelineState. Value type, like Metal's. raw is f's driver handle,
// for LaunchDirect; it is 0 when it could not be read, and LaunchDirect then
// takes Launch's path.
type Pipeline struct {
	f   *gc.Function
	raw cudasys.CUfunction
}

// SetMaxDynamicSharedBytes opts the kernel into launches that request up to n bytes of dynamic shared memory
// (cuFuncSetAttribute, MAX_DYNAMIC_SHARED_SIZE_BYTES). Without it a launch is limited to the device's 48 KiB default; n above
// MaxDynamicSharedBytes is refused by the driver. It applies to the kernel, so to every launch of it.
func (p Pipeline) SetMaxDynamicSharedBytes(n int) error {
	if p.f == nil {
		return fmt.Errorf("cuda: SetMaxDynamicSharedBytes on a nil pipeline")
	}
	return p.f.SetMaxDynamicSharedMemory(n)
}

// MaxDynamicSharedBytes is the most dynamic shared memory this kernel can opt into on d: the device's per-block opt-in limit,
// which counts static and dynamic shared memory together, less the kernel's static shared memory
// (CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES). 0 means unknown.
func (p Pipeline) MaxDynamicSharedBytes(d *Device) int {
	if p.f == nil {
		return 0
	}
	opt := d.MaxSharedBytesPerBlockOptIn()
	static, err := p.f.Attribute(gc.FuncAttrSharedSizeBytes)
	if opt <= 0 || err != nil || static > opt {
		return 0
	}
	return opt - static
}

// NewComputePipeline looks a kernel up in a loaded module. The kernel must be
// declared `extern "C"` so its name is unmangled.
func (d *Device) NewComputePipeline(lib Library, fn string) (Pipeline, error) {
	if lib.m == nil {
		return Pipeline{}, fmt.Errorf("cuda: pipeline %q: nil library", fn)
	}
	f, err := lib.m.Function(fn)
	if err != nil {
		return Pipeline{}, fmt.Errorf("cuda: no kernel %q in module: %w", fn, err)
	}
	return Pipeline{f: f, raw: rawFunction(f)}, nil
}

// ---- buffers ----

// Buffer is a device allocation, addressed as raw bytes exactly like an MTLBuffer.
// A single untyped Buffer (rather than gocudrv's generic Buffer[T]) is what lets
// the variadic `bufs ...Buffer` dispatch signature mirror Metal's; the element type
// lives in the kernel signature, where it already had to agree.
type Buffer struct {
	b   *gc.Buffer[uint8]
	cx  *gc.Context // owning context, so upload can order its copy against launches
	d   *Device     // owning device, whose Activity counts this buffer's host waits
	n   int         // element count in the constructor's unit (floats, int8s, u32s, …)
	off uintptr     // byte offset for binding (a sub-view into a larger buffer; 0 = whole)
	// raw != 0 ⇒ this Buffer is a device-usable pointer into MAPPED PINNED HOST memory (see
	// NewMappedHostBuffer): b is nil, the bytes live in host RAM, and a kernel reads them
	// directly over PCIe (zero-copy). arg() binds raw instead of b.b.DevicePtr().
	raw cudasys.CUdeviceptr
}

// At returns a view of the buffer bound at byteOff — for reading a slice of a
// combined buffer. Zero-copy; only the bind offset changes. Mirrors metal.go's At.
func (b Buffer) At(byteOff int) Buffer { b.off = uintptr(byteOff); return b }

// Len is the buffer's element count in its constructor's unit.
func (b Buffer) Len() int { return b.n }

// MustBuf turns a FAILED allocation into a loud panic instead of a silently unusable
// Buffer, and records it so ReleaseAll can free it — metal.go's discipline, kept
// verbatim: an OOM must not surface downstream as garbage numerics. Consumers that
// can degrade (a backend's init, an EnableGPU) recover the panic and fall back to CPU.
func (d *Device) MustBuf(nBytes, n int, what string) Buffer {
	return d.MustBufOn(UnlinedLine, nBytes, n, what)
}

// MustBufOn is MustBuf leasing the bytes on plan line line.
func (d *Device) MustBufOn(line string, nBytes, n int, what string) Buffer {
	if d.cx == nil {
		panic(fmt.Sprintf("cuda: allocation on a released device (%s, %d bytes)", what, nBytes))
	}
	buf, err := gc.Alloc[uint8](d.cx, nBytes)
	if err != nil {
		panic(fmt.Sprintf("cuda: device allocation failed (%s, %d bytes): %v", what, nBytes, err))
	}
	release := d.lease(true, line, int64(nBytes), ChargedBytes(int64(nBytes)))
	d.mu.Lock()
	d.allocs = append(d.allocs, devAlloc{b: buf, release: release})
	d.mu.Unlock()
	return Buffer{b: buf, cx: d.cx, d: d, n: n}
}

// asBytes reinterprets a slice of fixed-size scalars as the []byte gocudrv copies.
// The caller must runtime.KeepAlive the source across the copy.
func asBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var z T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(z)))
}

// upload copies src into the buffer at its bind offset, and does NOT return until the
// bytes are actually visible to a kernel.
//
// That last part is load-bearing and was a real data race. gocudrv creates every
// stream with CU_STREAM_NON_BLOCKING, which by definition does not order against the
// legacy null stream — and the copy below runs on the context, i.e. the null stream,
// while Queue.Launch runs on our own stream. Worse, cuMemcpyHtoD from PAGEABLE host
// memory (a Go slice) is documented to return once the source has been staged for DMA,
// with the transfer to the device still in flight. So "upload then launch" had no
// ordering guarantee at all.
//
// It surfaced as an intermittently wrong GEMM: whichever kernel launched FIRST after a
// large upload read a partially-landed buffer, with the errors clustered in the tail of
// the last bytes copied. It reproduced 15/15 at a 129x65x1152 f32 GEMM and vanished
// when a second kernel was launched ahead of it — which is exactly why it hid for so
// long behind small fixtures.
//
// The synchronize makes the copy complete before we return. It is a full device sync,
// which is heavier than ideal, but uploads are per-request (a corpus, a tower, a batch
// of patches) rather than per-op, and a correct upload is not negotiable. A cheaper
// fix — an async copy on the queue's own stream — needs pinned host memory and a
// queue-aware Buffer; worth doing if profiling ever shows this on a hot path.
func (b Buffer) upload(src []byte) error {
	if b.b == nil {
		return fmt.Errorf("cuda: upload to a nil buffer")
	}
	if len(src) == 0 {
		return nil
	}
	if got, want := b.b.Bytes(), uint64(b.off)+uint64(len(src)); got < want {
		return fmt.Errorf("cuda: upload of %d bytes at offset %d overruns a %d-byte buffer", len(src), b.off, got)
	}
	// SYNCHRONIZE BEFORE THE COPY TOO, not just after (audit C-01).
	//
	// The trailing Synchronize below orders this copy before LATER launches —
	// the read-after-write race the comment above describes. It does nothing
	// about the mirror image: a launch ALREADY QUEUED on the (non-blocking)
	// stream that READS this buffer, while the null-stream DMA lands new bytes
	// underneath it. That write-after-read is how qwencuda's mid-forward segment
	// upload could feed full-image bounds to layers still running with shared
	// memory sized for the windowed ones.
	//
	// Same reasoning as the trailing sync: a full device sync is heavier than
	// ideal, uploads are per-request rather than per-op, and a correct upload is
	// not negotiable. A caller on a hot path wants an async copy on the queue's
	// own stream, which needs pinned host memory and a queue-aware Buffer.
	b.d.noteWait() // the copy waits on the device before and after it: one host wait
	if b.cx != nil {
		if err := b.cx.Synchronize(bg); err != nil {
			return err
		}
	}
	if err := b.b.CopyFromAt(bg, int(b.off), src); err != nil {
		return err
	}
	if b.cx == nil {
		return nil
	}
	return b.cx.Synchronize(bg)
}

// download copies the buffer's contents at its bind offset into dst.
func (b Buffer) download(dst []byte) error {
	if b.b == nil {
		return fmt.Errorf("cuda: download from a nil buffer")
	}
	if len(dst) == 0 {
		return nil
	}
	if got, want := b.b.Bytes(), uint64(b.off)+uint64(len(dst)); got < want {
		return fmt.Errorf("cuda: download of %d bytes at offset %d overruns a %d-byte buffer", len(dst), b.off, got)
	}
	b.d.noteWait() // a blocking copy to the host
	return b.b.CopyToAt(bg, dst, int(b.off))
}

// NewBufferLen allocates a device buffer of nFloats float32s. The memory is UNINITIALIZED (cuMemAlloc does not clear it): it holds whatever the
// allocation last held until the first write. Metal's NewBufferLen is zero-filled, so the two backends differ here by design; code shared across
// backends must not rely on zeros from either. Zero a buffer that needs it explicitly (Queue.ZeroAsync).
func (d *Device) NewBufferLen(nFloats int) Buffer { return d.MustBuf(nFloats*4, nFloats, "len") }

// NewBufferLenOn is NewBufferLen leasing the bytes on plan line line.
func (d *Device) NewBufferLenOn(line string, nFloats int) Buffer {
	return d.MustBufOn(line, nFloats*4, nFloats, "len")
}

// NewBufferBytes allocates an UNINITIALIZED device buffer of n BYTES (n is also
// the element count for the returned Buffer; Metal's NewBufferBytes is zero-filled, so shared code must not rely on zeros) — the primitive for consumers that
// size in raw bytes rather than float32s.
func (d *Device) NewBufferBytes(n int) Buffer { return d.MustBuf(n, n, "bytes") }

// NewBufferBytesOn is NewBufferBytes leasing the bytes on plan line line.
func (d *Device) NewBufferBytesOn(line string, n int) Buffer {
	return d.MustBufOn(line, n, n, "bytes")
}

// SetU32 overwrites a 1-word uniform buffer (per-call: rope pos, nKeys, a row count).
func (b Buffer) SetU32(v uint32) error { return b.upload(asBytes([]uint32{v})) }

// ---- generic buffer verbs (any scalar element type) ----
//
// A consumer with a wider mix of element types — a decode path allocating
// float32 activations, int32 indices, uint16 f16 scales and uint32 packed
// weights — wants the same four verbs parameterized instead of a method per
// type. These are those, and they are free functions for the same reason
// NewHostBuffer is: Go has no generic methods. This used to be two parallel
// APIs (a type-suffixed one mirroring metal.go's constructors one for one,
// plus these) with every real consumer already on the generic one; the
// type-suffixed twin was deleted once nothing called it anymore.

// NewBufferOf allocates a device buffer sized to data and uploads it. n for the
// returned Buffer is len(data), in elements of T. Panics on allocation failure
// (see MustBuf).
func NewBufferOf[T Scalar](d *Device, data []T) Buffer { return NewBufferOfOn(d, UnlinedLine, data) }

// NewBufferOfOn is NewBufferOf leasing the bytes on plan line line.
func NewBufferOfOn[T Scalar](d *Device, line string, data []T) Buffer {
	var z T
	b := d.MustBufOn(line, len(data)*int(unsafe.Sizeof(z)), len(data), "typed")
	if err := b.upload(asBytes(data)); err != nil {
		panic(fmt.Sprintf("cuda: NewBufferOf upload: %v", err))
	}
	runtime.KeepAlive(data)
	return b
}

// NewBufferLenOf allocates an UNINITIALIZED device buffer of n elements of T (Metal's NewBufferLenOf is zero-filled; code shared across backends must not
// rely on zeros).
func NewBufferLenOf[T Scalar](d *Device, n int) Buffer { return NewBufferLenOfOn[T](d, UnlinedLine, n) }

// NewBufferLenOfOn is NewBufferLenOf leasing the bytes on plan line line.
func NewBufferLenOfOn[T Scalar](d *Device, line string, n int) Buffer {
	var z T
	return d.MustBufOn(line, n*int(unsafe.Sizeof(z)), n, "typed-len")
}

// Upload copies src into the buffer at its bind offset (host → device).
func Upload[T Scalar](b Buffer, src []T) error {
	err := b.upload(asBytes(src))
	runtime.KeepAlive(src)
	return err
}

// Download copies the buffer's contents at its bind offset into dst (device →
// host). dst's length sizes the transfer.
//
// Output contract: overwrites dst; do not pre-zero. Covers dst[:len(dst)].
func Download[T Scalar](b Buffer, dst []T) error {
	err := b.download(asBytes(dst))
	runtime.KeepAlive(dst)
	return err
}

// ---- pinned host memory ----

// Scalar is the element type of a by-value kernel argument, a typed buffer verb,
// or a pinned host buffer. It mirrors gocudrv's Supported exactly — fixed-size numeric scalars
// only, with `int`/`uint` deliberately excluded because their width is not part of
// the ABI the kernel was compiled against. Declaring it here is what lets
// consumers write gpu.ArgValue(int32(x)) without ever naming gocudrv.
type Scalar interface {
	~int8 | ~uint8 |
		~int16 | ~uint16 |
		~int32 | ~uint32 |
		~int64 | ~uint64 |
		~float32 | ~float64
}

// HostBuffer is page-locked (pinned) host memory. A DtoH copy into pinned memory
// can be done by the DMA engine without the driver staging through a bounce
// buffer, which is why a hot readback path — goinfer's per-token logits vector, a
// vocab-sized float32 every token — allocates one of these once and reuses it.
//
// Slice() hands back a []T view of that memory, so the readback costs one copy
// rather than a copy plus an allocation. The Device tracks it, so ReleaseObjects
// frees it; a HostBuffer MUST NOT outlive its Device (the free has to reach the
// context's executor).
type HostBuffer[T Scalar] struct {
	d *Device
	h *gc.HostBuffer[T]
}

// NewHostBuffer allocates n elements of pinned host memory on d.
//
// It is a package-level function rather than a method because Go has no generic
// methods — Device.NewHostBuffer[T](n) is not expressible. Call it as
// gpu.NewHostBuffer[float32](dev, vocab).
func NewHostBuffer[T Scalar](d *Device, n int) (*HostBuffer[T], error) {
	return NewHostBufferOn[T](d, UnlinedLine, n)
}

// NewHostBufferOn is NewHostBuffer leasing the bytes on plan line line, as host memory.
func NewHostBufferOn[T Scalar](d *Device, line string, n int) (*HostBuffer[T], error) {
	if d == nil || d.cx == nil {
		return nil, fmt.Errorf("cuda: NewHostBuffer on a released device")
	}
	h, err := gc.AllocHost[T](d.cx, n)
	if err != nil {
		return nil, fmt.Errorf("cuda: AllocHost(%d): %w", n, err)
	}
	d.TrackObj(h)
	var z T
	bytes := int64(n) * int64(unsafe.Sizeof(z))
	release := d.lease(false, line, bytes, bytes)
	d.mu.Lock()
	if d.hosts == nil {
		d.hosts = map[closer]func(){}
	}
	d.hosts[h] = release
	d.mu.Unlock()
	return &HostBuffer[T]{d: d, h: h}, nil
}

// Slice is a []T view of the pinned memory, valid until Close. Read and write it
// directly; it is ordinary host memory that happens to be page-locked.
func (h *HostBuffer[T]) Slice() []T {
	if h == nil {
		return nil
	}
	return h.h.Slice()
}

// Len is the element count.
func (h *HostBuffer[T]) Len() int {
	if h == nil {
		return 0
	}
	return h.h.Len()
}

// Close frees the pinned memory and drops it from the Device's ledger, so a later
// ReleaseObjects neither double-frees it nor leaves the leak counters stale.
// Idempotent.
func (h *HostBuffer[T]) Close() error {
	if h == nil || h.h == nil {
		return nil
	}
	h.d.untrackObj(h.h)
	return h.h.Close()
}

// MappedHostBuffer is page-locked host memory a kernel reads DIRECTLY over PCIe (zero-copy): no
// device allocation, no H2D copy. cuMemAllocHost memory is device-accessible, and on a UVA system
// (asserted at construction — every 64-bit CUDA GPU has it) the host pointer IS the device pointer,
// so Buffer().arg() binds it straight into a launch. This is how an int4 model whose weights exceed
// VRAM runs resident: keep the always-resident core in device memory and the large streamed tensors
// (MoE experts) host-mapped, and the kernel streams them on demand. Slower per byte than a
// DMA-to-VRAM cache (uncoalesced PCIe reads, no DMA engine), but architecturally trivial — the
// kernels and dispatch are unchanged; only where the tensor is allocated differs. Close before the
// owning Device.
type MappedHostBuffer struct {
	hb      *HostBuffer[uint8]        // origin: NewMappedHostBuffer (allocates, caller fills after)
	rh      *gc.RegisteredHost[uint8] // origin: RegisterMappedHostBuffer (caller already filled, this pins in place)
	buf     Buffer
	release func() // RegisterMappedHostBuffer's pin's lease (NewMappedHostBuffer's is its HostBuffer's)
}

// NewMappedHostBuffer allocates nBytes of pinned, device-mapped host memory. Bytes() is the
// host-writable backing (fill it with the tensor's bytes); Buffer() is the device-usable handle to
// pass to a kernel. Errors (not panics) so a caller can fall back to a device upload.
//
// KNOWN CONDITION — validate before relying on mapped reads in a NEW kernel. The zero-copy read is
// verified bit-identical to a device buffer for a straightforward GEMV at K up to 1 MB rows, byte
// offsets up to 64 MB, and 4096 concurrent warps. BUT one indexed MoE GEMV (goinfer's
// gemv_w4a8_moe) was observed to mis-read this memory at large widths — deterministically, cause
// unknown (not offset/K/occupancy/stream-ordering; a device buffer of the same bytes is correct).
// The primitive is correct in isolation and the failure was in that kernel's composition, but the
// safe rule for a new consumer is: check any kernel's mapped reads against a device-buffer control
// (same bytes, same launch) before trusting them; a passing isolation test does not prove the
// composed use.
func (d *Device) NewMappedHostBuffer(nBytes int) (*MappedHostBuffer, error) {
	return d.NewMappedHostBufferOn(UnlinedLine, nBytes)
}

// NewMappedHostBufferOn is NewMappedHostBuffer leasing the bytes on plan line line.
func (d *Device) NewMappedHostBufferOn(line string, nBytes int) (*MappedHostBuffer, error) {
	if d.cx == nil {
		return nil, fmt.Errorf("cuda: NewMappedHostBuffer on a released device")
	}
	if nBytes <= 0 {
		return nil, fmt.Errorf("cuda: NewMappedHostBuffer nBytes=%d", nBytes)
	}
	// UVA is required: without unified addressing the host pointer is not a valid device pointer and
	// the zero-copy read would fault. Every 64-bit CUDA GPU has it, but assert rather than assume.
	if d.dev != nil {
		if uva, err := d.dev.Attribute(gc.DeviceAttributeUnifiedAddressing); err == nil && uva == 0 {
			return nil, fmt.Errorf("cuda: NewMappedHostBuffer needs UNIFIED_ADDRESSING (zero-copy host reads)")
		}
	}
	hb, err := NewHostBufferOn[uint8](d, line, nBytes)
	if err != nil {
		return nil, err
	}
	s := hb.Slice()
	if len(s) == 0 {
		_ = hb.Close()
		return nil, fmt.Errorf("cuda: NewMappedHostBuffer got an empty pinned slice")
	}
	dptr := cudasys.CUdeviceptr(uintptr(unsafe.Pointer(&s[0])))
	return &MappedHostBuffer{hb: hb, buf: Buffer{cx: d.cx, d: d, n: nBytes, raw: dptr}}, nil
}

// RegisterMappedHostBuffer pins data IN PLACE — no allocation, no copy — and returns the same
// zero-copy device-usable handle NewMappedHostBuffer does. data must already hold its final
// content; the returned MappedHostBuffer keeps data reachable (preventing the GC from collecting
// or moving it) for as long as the buffer is open, so the caller does not need to separately.
//
// WHY THIS EXISTS, and why it is not simply "NewMappedHostBuffer but you supply the bytes":
// cuMemAllocHost (what NewMappedHostBuffer calls) allocates FRESH pinned memory — the driver has
// to find that many bytes of LOCKABLE physical RAM right then, which on a box with a large page
// cache means reclaiming cache pages first, a real and highly variable cost. cuMemHostRegister
// (this function) pins pages that are already resident — data was already touched by the caller's
// own fill, so the memory is already committed; no reclaim is forced by this call. Measured on an
// RTX 2070 SUPER at the 11.4 GB scale goinfer's C′ expert-stack staging runs at: populate-then-
// register beat allocate-then-copy in every trial, 1.33x-4.46x (docs/measurements/
// lead3-pin-order-2026-09-22.md in goinfer — cross-repo citation, not a path this module can
// resolve, described rather than linked).
//
// Errors (not panics) so a caller can fall back to NewMappedHostBuffer, same convention as it.
func (d *Device) RegisterMappedHostBuffer(data []byte) (*MappedHostBuffer, error) {
	return d.RegisterMappedHostBufferOn(UnlinedLine, data)
}

// RegisterMappedHostBufferOn is RegisterMappedHostBuffer leasing the pinned bytes on plan line line, as host memory. The pin is not
// tracked by the device, so its lease stays open until Close: a pin left open past ReleaseObjects shows as an open lease there.
func (d *Device) RegisterMappedHostBufferOn(line string, data []byte) (*MappedHostBuffer, error) {
	if d.cx == nil {
		return nil, fmt.Errorf("cuda: RegisterMappedHostBuffer on a released device")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("cuda: RegisterMappedHostBuffer got empty data")
	}
	if d.dev != nil {
		if uva, err := d.dev.Attribute(gc.DeviceAttributeUnifiedAddressing); err == nil && uva == 0 {
			return nil, fmt.Errorf("cuda: RegisterMappedHostBuffer needs UNIFIED_ADDRESSING (zero-copy host reads)")
		}
	}
	rh, err := gc.RegisterHost[uint8](d.cx, data)
	if err != nil {
		return nil, fmt.Errorf("cuda: RegisterHost(%d bytes): %w", len(data), err)
	}
	dptr := cudasys.CUdeviceptr(uintptr(unsafe.Pointer(&data[0])))
	release := d.lease(false, line, int64(len(data)), int64(len(data)))
	return &MappedHostBuffer{rh: rh, buf: Buffer{cx: d.cx, d: d, n: len(data), raw: dptr}, release: release}, nil
}

// Bytes is the host-writable backing memory — fill it with the tensor's bytes before launch.
// (A RegisterMappedHostBuffer's data was already filled before this call; Bytes() still returns
// it, for a caller that wants to read back or verify what it pinned.)
func (m *MappedHostBuffer) Bytes() []byte {
	if m == nil {
		return nil
	}
	if m.rh != nil {
		return m.rh.Slice()
	}
	return m.hb.Slice()
}

// Buffer is the device-usable, zero-copy handle to pass to a kernel via Arg.
func (m *MappedHostBuffer) Buffer() Buffer {
	if m == nil {
		return Buffer{}
	}
	return m.buf
}

// Close unregisters (and, for NewMappedHostBuffer's origin, frees) the pinned host memory. A
// RegisterMappedHostBuffer's underlying data is the caller's own slice and is not freed here —
// only unpinned; the caller's ordinary Go GC ownership of it resumes. Close before the owning
// Device.
func (m *MappedHostBuffer) Close() error {
	if m == nil {
		return nil
	}
	if m.rh != nil {
		if m.release != nil {
			m.release()
			m.release = nil
		}
		return m.rh.Close()
	}
	return m.hb.Close()
}

// ReadToHost copies the device buffer's contents (from its bind offset) into
// pinned host memory — the readback goinfer's logits path uses. The transfer size
// is dst's byte length, so dst sizes the copy.
//
// It is a free function for the same reason NewHostBuffer is: Buffer is not
// generic (it is untyped bytes, like an MTLBuffer), and the element type lives
// only on the host side here.
//
// Output contract: overwrites dst's whole slice; do not pre-zero.
func ReadToHost[T Scalar](b Buffer, dst *HostBuffer[T]) error {
	if dst == nil {
		return fmt.Errorf("cuda: ReadToHost into a nil host buffer")
	}
	s := dst.Slice()
	if s == nil {
		return fmt.Errorf("cuda: ReadToHost into a closed host buffer")
	}
	err := b.download(asBytes(s))
	runtime.KeepAlive(dst)
	return err
}

// ---- explicit launch geometry, scalar args, async dispatch ----
//
// Everything below is the surface a TUNED kernel set needs and the ANN proving
// path does not: kernels that take scalars positionally between buffers, geometry
// the author picks by hand (including dynamic shared memory), and a launch-many-
// then-sync-at-a-boundary model rather than a sync per call. Run1D / Run1DBatch /
// Encoder above stay as they are — they are the right ergonomics for aikit's own
// consumers, and they are built on this same machinery.
//
// The design constraint is that a consumer can express its whole decode loop
// importing ONLY this package: no gocudrv type appears in any signature here.

// KernelArg is one positional kernel parameter — a device buffer (Arg) or a scalar
// passed by value (ArgValue). A kernel's parameter list is these in order.
//
// raw holds the parameter's bytes as LaunchDirect passes them (a device pointer, or
// the scalar's bits in its low bytes); n is its size, 0 for a nil buffer.
type KernelArg struct {
	a   gc.KernelArg
	raw uint64
	n   uint8
}

// Arg passes a device buffer as a kernel pointer parameter. A buffer carrying a
// bind offset (Buffer.At) passes base+offset.
func Arg(b Buffer) KernelArg {
	k := KernelArg{a: b.arg()}
	switch {
	case b.raw != 0:
		k.raw, k.n = uint64(b.raw)+uint64(b.off), 8
	case b.b != nil:
		k.raw, k.n = uint64(b.b.DevicePtr())+uint64(b.off), 8
	}
	return k
}

// ArgValue passes a scalar BY VALUE — the calling convention the tuned kernels
// use for their dimensions, epsilons and flags (`int32 n`, `float32 eps`), as
// opposed to the 1-element-buffer convention the ANN kernels inherited from MSL's
// `constant uint&` binds. Both work; a kernel's signature decides which it needs.
func ArgValue[T Scalar](v T) KernelArg {
	k := KernelArg{a: gc.ArgValue(v), n: uint8(unsafe.Sizeof(v))}
	*(*T)(unsafe.Pointer(&k.raw)) = v
	return k
}

// ArgNull passes a NULL device pointer (CUdeviceptr 0) as a kernel pointer
// parameter — the "optional buffer, absent this call" arg a tuned kernel reads as
// `if (bias) …`. It is distinct from Arg(Buffer{}): a zero Buffer marshals through
// gocudrv's Arg, which rejects a nil buffer handle (ErrNilBuffer), whereas a kernel
// that guards on a null pointer needs the pointer itself to be 0. goinfer's decode
// path binds an absent bias this way (it was gc.ArgDevicePtr(0) before the
// re-point); there is no host allocation, so it costs nothing.
func ArgNull() KernelArg { return KernelArg{a: gc.ArgDevicePtr(0), n: 8} }

// LaunchConfig is explicit launch geometry: grid and block dimensions plus dynamic
// shared memory. Mirrors the driver's own shape, so a kernel author sizes blocks
// and shared memory exactly as the kernel was written to expect — which
// Run1D's derived 1-D geometry cannot express.
type LaunchConfig struct {
	GridX, GridY, GridZ    uint32
	BlockX, BlockY, BlockZ uint32
	SharedMemBytes         uint32
}

// Grid1D covers n elements with ceil(n/block) blocks of `block` threads — the
// ordinary elementwise shape. As with every launch through this layer, the grid
// overhangs n whenever block ∤ n, so the kernel must bounds-check.
func Grid1D(n, block int) LaunchConfig {
	if n <= 0 || block <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32((n + block - 1) / block), GridY: 1, GridZ: 1,
		BlockX: uint32(block), BlockY: 1, BlockZ: 1,
	}
}

// GridOne is a SINGLE block of `block` threads with sharedBytes of dynamic shared
// memory — the shape a whole-vector reduction uses (an RMS norm over the hidden
// state staging its partials in shared memory, say), where one cooperating block
// is the point rather than a limitation.
func GridOne(block, sharedBytes int) LaunchConfig {
	if block <= 0 || sharedBytes < 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: 1, GridY: 1, GridZ: 1,
		BlockX: uint32(block), BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32(sharedBytes),
	}
}

// valid reports whether every dimension is non-zero (the driver rejects a zero).
func (c LaunchConfig) valid() bool {
	return c.GridX > 0 && c.GridY > 0 && c.GridZ > 0 &&
		c.BlockX > 0 && c.BlockY > 0 && c.BlockZ > 0
}

func (c LaunchConfig) toGC() gc.LaunchConfig {
	return gc.LaunchConfig{
		GridX: c.GridX, GridY: c.GridY, GridZ: c.GridZ,
		BlockX: c.BlockX, BlockY: c.BlockY, BlockZ: c.BlockZ,
		SharedMemBytes: c.SharedMemBytes,
	}
}

// Launch ENQUEUES one kernel on this queue's stream and returns without waiting.
// Nothing has necessarily executed when it returns — call Sync to wait.
//
// This is the shape a tuned decode loop needs: a whole layer stack is enqueued
// back-to-back and synchronized once at a readback boundary, so the GPU is never
// idling on a per-launch round trip. Launches on one stream still run in issue
// order and each sees the prior one's writes, so chained kernels need no barrier.
//
// Errors are returned, not latched — a caller driving a hot chain typically keeps
// its own sticky first-error latch so it can discard returns in the inner loop
// without losing a bad config.
func (q Queue) Launch(p Pipeline, cfg LaunchConfig, args ...KernelArg) error {
	if p.f == nil {
		return fmt.Errorf("cuda: launch of a nil pipeline")
	}
	if !cfg.valid() {
		return fmt.Errorf("cuda: invalid launch geometry (grid %dx%dx%d block %dx%dx%d)",
			cfg.GridX, cfg.GridY, cfg.GridZ, cfg.BlockX, cfg.BlockY, cfg.BlockZ)
	}
	ga := make([]gc.KernelArg, len(args))
	for i, a := range args {
		if a.a == nil {
			return fmt.Errorf("cuda: kernel arg %d is uninitialized", i)
		}
		ga[i] = a.a
	}
	if q.d != nil {
		q.d.launches.Add(1)
	}
	if q.s == nil {
		return p.f.Launch(bg, cfg.toGC(), ga...)
	}
	return p.f.LaunchOn(bg, q.s, cfg.toGC(), ga...)
}

// Sync blocks until every kernel enqueued on this queue has completed — the
// explicit boundary the async Launch model is built around.
func (q Queue) Sync() error { return q.sync() }

// Graph is a replayable capture of a FIXED sequence of stream work: record it once with Capture,
// then Replay it per iteration as a SINGLE driver call instead of re-issuing every kernel launch —
// which is the whole win on the cgo-free (purego) path, where each launch is an FFI crossing. The
// captured work must be STATIC: fixed kernel arguments, with only buffer CONTENTS varying between
// replays (a per-call arg VALUE is frozen at capture). Nothing may synchronize the stream during
// capture (no Sync/Upload/Download inside the issue closure). Capture and Replay must run on the same
// OS thread that owns the queue's context. Close before the owning Device.
type Graph struct {
	exec *gc.GraphExec
	q    Queue
}

// Capture records the launches that `issue` enqueues on this queue's stream into a replayable Graph.
// It brackets issue with BeginCapture/EndCapture (thread-local: safety checks scoped to this thread,
// matching a single pinned-executor model) and instantiates the result.
func (q Queue) Capture(issue func() error) (*Graph, error) {
	if q.s == nil {
		return nil, fmt.Errorf("cuda: Capture on a queue with no stream")
	}
	if err := q.s.BeginCapture(gc.CaptureModeThreadLocal); err != nil {
		return nil, err
	}
	issueErr := issue()
	g, endErr := q.s.EndCapture() // always end capture, even if issue failed, to leave the stream usable
	if issueErr != nil {
		if g != nil {
			_ = g.Close()
		}
		return nil, fmt.Errorf("cuda: Capture issue: %w", issueErr)
	}
	if endErr != nil {
		return nil, endErr
	}
	defer func() { _ = g.Close() }()
	exec, err := g.Instantiate()
	if err != nil {
		return nil, err
	}
	return &Graph{exec: exec, q: q}, nil
}

// Replay launches the captured graph on the queue's stream — one driver call for all captured
// kernels, stream-ordered like any launch. Run it on the queue's owning thread.
func (g *Graph) Replay() error {
	if g == nil || g.exec == nil {
		return fmt.Errorf("cuda: Replay of a nil graph")
	}
	if g.q.d != nil {
		g.q.d.launches.Add(1) // one submission for the captured kernels
	}
	return g.exec.Launch(bg, g.q.s)
}

// Close destroys the instantiated graph. Call before the owning Device.
func (g *Graph) Close() error {
	if g == nil || g.exec == nil {
		return nil
	}
	return g.exec.Close()
}

// UploadAsync enqueues an H2D copy of the pinned src into dst on THIS queue's stream and returns
// WITHOUT synchronizing. Because the copy is stream-ordered, any kernel later Launch'd on the same
// queue observes the bytes with no host-side wait — unlike Upload, which does a blocking synchronous
// copy through a private path. Use it for a producer whose only consumer is a kernel on this queue
// (e.g. a per-token index buffer): same-stream order is the ONLY guarantee, so a consumer on a
// DIFFERENT queue still needs an explicit event, and the host must not overwrite src until this queue
// drains (Sync, or a later drain such as a per-token logits readback). src MUST be pinned (a
// HostBuffer); dst is written from its base (any bind offset on dst is ignored).
func (q Queue) UploadAsync(dst Buffer, src *HostBuffer[uint8]) error {
	if dst.b == nil {
		return fmt.Errorf("cuda: UploadAsync into a nil or mapped-host buffer")
	}
	if q.s == nil {
		return fmt.Errorf("cuda: UploadAsync on a queue with no stream")
	}
	if src == nil || src.h == nil {
		return fmt.Errorf("cuda: UploadAsync from a nil pinned buffer")
	}
	if src.h.Bytes() > dst.b.Bytes() {
		return fmt.Errorf("cuda: UploadAsync src %d bytes exceeds dst %d bytes", src.h.Bytes(), dst.b.Bytes())
	}
	return dst.b.CopyFromHostAsync(bg, q.s, src.h)
}

// arg builds the kernel argument for this buffer. At offset 0 it goes through
// gocudrv's Arg, which holds the buffer's lock for the launch (so a concurrent
// Close cannot free memory out from under a running kernel); an offset view has to
// pass a raw device pointer and forgoes that guard, which is why At is documented
// as a sub-view of a buffer the caller keeps alive.
func (b Buffer) arg() gc.KernelArg {
	if b.raw != 0 { // mapped pinned host memory: the host pointer IS the device pointer under UVA
		return gc.ArgDevicePtr(b.raw + cudasys.CUdeviceptr(b.off))
	}
	if b.off == 0 {
		return gc.Arg(b.b)
	}
	return gc.ArgDevicePtr(b.b.DevicePtr() + cudasys.CUdeviceptr(b.off))
}

// ---- queue + dispatch ----

// Queue is a CUDA stream — the counterpart of an MTLCommandQueue. Value type.
type Queue struct {
	d *Device
	s *gc.Stream // nil ⇒ the default (null) stream
}

// NewCommandQueue creates the stream work is submitted on (built once, reused).
// Total, like Metal's: if the driver declines a new stream, the Queue falls back to
// the default (null) stream, which is a valid submission target — a degraded
// stream is not worth failing a whole backend over.
func (d *Device) NewCommandQueue() Queue {
	s, err := d.cx.NewStream()
	if err != nil {
		return Queue{d: d}
	}
	d.TrackObj(s)
	return Queue{d: d, s: s}
}

// grid derives the launch geometry for n threads at threadgroup width tg — the
// Metal-shaped convenience the ANN path uses, where the caller says "n threads"
// and the block size is a tuning detail. tg is clamped to n and to CUDA's hard
// 1024-threads-per-block ceiling. As always, the grid overhangs n whenever the
// block size does not divide it, so the kernel must bounds-check (divergence 2 in
// the file header).
func grid(n, tg int) (LaunchConfig, error) {
	if n <= 0 {
		return LaunchConfig{}, fmt.Errorf("cuda: dispatch of %d threads", n)
	}
	if tg <= 0 {
		tg = 256
	}
	if tg > n {
		tg = n
	}
	if tg > 1024 {
		tg = 1024 // CUDA's hard max threads-per-block
	}
	cfg := Grid1D(n, tg)
	if !cfg.valid() {
		return cfg, fmt.Errorf("cuda: invalid launch geometry (n=%d tg=%d)", n, tg)
	}
	return cfg, nil
}

// launch enqueues one kernel on this queue's stream, binding bufs positionally —
// the all-buffers convention the ANN kernels use. It is Launch with the geometry
// derived and every argument a buffer.
func (q Queue) launch(p Pipeline, n, tg int, bufs []Buffer) error {
	cfg, err := grid(n, tg)
	if err != nil {
		return err
	}
	args := make([]KernelArg, len(bufs))
	for i, b := range bufs {
		if b.b == nil {
			return fmt.Errorf("cuda: kernel arg %d is a nil buffer", i)
		}
		args[i] = Arg(b)
	}
	return q.Launch(p, cfg, args...)
}

// sync blocks until this queue's stream drains.
func (q Queue) sync() error {
	if q.d != nil {
		q.d.waits.Add(1)
	}
	if q.s == nil {
		return q.d.cx.Synchronize(bg)
	}
	return q.s.Synchronize(bg)
}

// Run1D runs a 1-D kernel over n threads (threadgroup width tg), binding bufs as
// the kernel's positional parameters, and blocks until the GPU finishes.
// Mirrors metal.go's Run1D, plus an error return (divergence 3).
func (q Queue) Run1D(p Pipeline, n, tg int, bufs ...Buffer) error {
	if err := q.launch(p, n, tg, bufs); err != nil {
		return fmt.Errorf("cuda: launch: %w", err)
	}
	if err := q.sync(); err != nil {
		return fmt.Errorf("cuda: synchronize: %w", err)
	}
	return nil
}

// Run1DBatch enqueues `reps` dispatches of the same kernel and synchronizes once —
// the shape that isolates the per-submit round trip from the marginal per-launch
// cost. Mirrors metal.go's Run1DBatch.
func (q Queue) Run1DBatch(p Pipeline, n, tg, reps int, bufs ...Buffer) error {
	for range reps {
		if err := q.launch(p, n, tg, bufs); err != nil {
			return fmt.Errorf("cuda: launch: %w", err)
		}
	}
	if err := q.sync(); err != nil {
		return fmt.Errorf("cuda: synchronize: %w", err)
	}
	return nil
}

// Encoder batches many DIFFERENT kernel dispatches before one synchronize — the
// per-token shape the decode loop needs (a whole layer stack → one round trip),
// and the counterpart of metal.go's one-command-buffer Encoder. Launches on a
// single CUDA stream run in issue order and each sees the prior one's writes, so
// chained kernels need no explicit barrier, exactly as Metal's serial compute
// encoder provides.
//
// The first error is latched and short-circuits the rest; End returns it. That
// keeps the call sites free of per-Dispatch error checks (matching Metal's
// signature) without ever losing a failure.
type Encoder struct {
	q   Queue
	err error
	n   int
}

// Begin starts a batch of dispatches on this queue.
func (q Queue) Begin() *Encoder { return &Encoder{q: q} }

// Dispatch encodes one kernel over n threads (threadgroup width tg, clamped ≤ n),
// binding bufs as the kernel's positional parameters.
func (e *Encoder) Dispatch(p Pipeline, n, tg int, bufs ...Buffer) {
	if e.err != nil {
		return
	}
	if err := e.q.launch(p, n, tg, bufs); err != nil {
		e.err = fmt.Errorf("cuda: dispatch %d: %w", e.n, err)
		return
	}
	e.n++
}

// End waits for every dispatch in the batch and reports the first failure.
func (e *Encoder) End() error {
	if e.err != nil {
		return e.err
	}
	if err := e.q.sync(); err != nil {
		return fmt.Errorf("cuda: synchronize: %w", err)
	}
	return nil
}

// Err reports the first dispatch failure so far, without waiting.
func (e *Encoder) Err() error { return e.err }
