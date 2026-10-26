//go:build darwin

package gpudev

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// Fence probe — goinfer Metal audit finding C-B03, for goinfer's reopened M-11.
//
// Question: per command-buffer boundary, does a GPU-written, CPU-spun fence word wake the host
// sooner than commit + waitUntilCompleted? goinfer's paged-MoE decode on Metal ends 61–81 command
// buffers per token (the host must read the router's choice before encoding the next phase), and
// measured ~0.2–0.3 ms of GPU idle per boundary. The earlier event probe (goinfer
// metal/pagecost_sharedevent_test.go; this package's metal_sharedevent_test.go) priced the
// MTLSharedEvent handshake at ~0.26 ms/boundary vs ~0.23 ms for a plain per-layer submit — no gain.
// That tested the EVENT. MLX's fast fence (mlx/backend/metal/fence.cpp + kernels/fence.metal,
// gated by MLX_METAL_FAST_SYNCH) is a different wake path: a kernel stores a counter through a
// `volatile coherent(system) device uint*` followed by a seq_cst system-scope
// atomic_thread_fence, and the CPU spins on that word in user space instead of sleeping in
// waitUntilCompleted. Needs MSL 3.2 (coherent(system)) on macOS 15 / a Metal-3 GPU.
//
// Arms, each a one-thread kernel writing one word, interleaved in rotating blocks:
//
//	A (today)   one command buffer per boundary: commit, waitUntilCompleted.
//	B (fence)   the same, but the kernel also stores the sequence number to a coherent(system)
//	            word; commit, spin until the CPU sees it (wake), THEN waitUntilCompleted (done),
//	            so the report shows whether the spin returns before the driver's completion. The
//	            payload is stored through coherent(system) too — see B0 for why.
//	B0 (diag)   B with the payload through a plain `device uint*`, as an unmodified router kernel
//	            writes it. The first smoke saw that payload STALE on 47/50 boundaries when the
//	            fence word was already visible: a system-scope fence does not publish
//	            non-coherent stores, which is why MLX runs input_coherent before a cross-device
//	            fence. B0 is kept to measure that stale rate; its wake time is not usable.
//	C (event)   the earlier probe's shape for comparability: one command buffer per block, each
//	            boundary = GPU encodeWaitForEvent(ack) → kernel → encodeSignalEvent; the CPU acks,
//	            then spins on signaledValue. Per boundary = CPU ack → CPU observes the signal.
//
// "wake" is the round trip the decode loop pays: A/B from commit, C from the ack, to the moment
// the CPU knows the GPU work is done. GPUStartTime/GPUEndTime (host clock, = CLOCK_UPTIME_RAW)
// split A/B into launch (commit → GPU start), GPU span, and tail (GPU end → CPU knows).
//
// Not covered: the real router writes its output from many threads in an EARLIER dispatch, so a
// production fence also needs the encoder barrier MLX inserts before fence_update; here one thread
// writes payload and fence in one kernel.
//
// DECISION LINE (pre-registered in goinfer's audit; do not move): p50(A.wake) − p50(B.wake)
// under 25 µs kills the idea; at or above it, the number goes to goinfer. B, not B0, because B
// is the arm whose payload is visible when its fence is. Applied only at n ≥ 1000 per arm —
// smaller runs print EXPLORATORY and no verdict, and a stale payload in B voids the verdict.
//
// Timing probe, so opt-in (CI and default runs skip it). The owner's Mac runs timed work on the
// night queue only; by day, smoke at a tiny n:
//
//	AIKIT_FENCE_PROBE=1 AIKIT_FENCE_N=50 GOWORK=off go test ./ -run 'TestFenceProbe' -v -count=1 -timeout 5m
//
// Real run (night queue): AIKIT_FENCE_N=5000 (≥ 1000). AIKIT_FENCE_BLOCK sets the block size
// (default 50).

// msl3_2 is MTLLanguageVersion3_2 — the first MSL with the coherent(system) address qualifier.
const msl3_2 uint = (3 << 16) | 2

// fenceProbeSrc follows MLX's kernels/fence.metal (v0.32.2) exactly in its preamble: the
// system thread scope is not public MSL, so MLX enables internals and builds it from the
// compiler's __METAL_MEMORY_SCOPE_SYSTEM__ (3). The fence kernels add a release fence BEFORE the
// fence store (MLX gets that ordering from an encoder barrier + its input_coherent pass): a fence
// word is only useful if the payload written before it is visible when it is seen, and arms B/B0
// check exactly that (stale-payload count).
const fenceProbeSrc = `
#pragma METAL internals : enable

#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>

// Arms A and C: one thread writes one word (the payload a router would leave behind).
[[kernel]] void tick(device uint* out [[buffer(0)]], constant uint& v [[buffer(1)]]) {
  out[0] = v;
}

// Arm B: the payload through a system-coherent pointer, then the sequence number into a
// system-coherent word.
[[kernel]] void tick_fence(
    volatile coherent(system) device uint* out [[buffer(0)]],
    constant uint& v [[buffer(1)]],
    volatile coherent(system) device uint* fence [[buffer(2)]]) {
  out[0] = v;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
  fence[0] = v;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}

// Arm B0 (diagnostic): as tick_fence, but the payload through a plain device pointer.
[[kernel]] void tick_fence_plain(
    device uint* out [[buffer(0)]],
    constant uint& v [[buffer(1)]],
    volatile coherent(system) device uint* fence [[buffer(2)]]) {
  out[0] = v;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
  fence[0] = v;
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device,
      metal::memory_order_seq_cst,
      metal::thread_scope_system);
}
`

const (
	clockUptimeRaw     = 8  // CLOCK_UPTIME_RAW: mach_absolute_time in ns — the host clock GPUStartTime reports in
	clockThreadCPUTime = 16 // CLOCK_THREAD_CPUTIME_ID: this OS thread's CPU time (the test is LockOSThread'd)
	mtlGPUFamilyMetal3 = 5001

	fenceSpinCap     = 1 << 31 // spin iteration cap — well past the deadline at ~ns/iteration
	fenceSpinTimeout = 2e9     // ns: a fence that never lands fails in 2 s instead of hanging
)

var (
	fenceClockOnce     sync.Once
	clockGettimeNsecNP func(clockID uint32) uint64

	fenceSelSetBytes       = objc.RegisterName("setBytes:length:atIndex:")
	fenceSelSupportsFamily = objc.RegisterName("supportsFamily:")
)

func loadFenceClocks() {
	fenceClockOnce.Do(func() {
		h, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			panic("fence probe: dlopen libSystem: " + err.Error())
		}
		purego.RegisterLibFunc(&clockGettimeNsecNP, h, "clock_gettime_nsec_np")
	})
}

func hostNs() uint64      { return clockGettimeNsecNP(clockUptimeRaw) }
func threadCPUNs() uint64 { return clockGettimeNsecNP(clockThreadCPUTime) }

// fenceProbeGate skips unless AIKIT_FENCE_PROBE is set, and returns a device that can run the
// probe (Metal-3 family, as MLX requires before it enables the fast fence).
func fenceProbeGate(t *testing.T) *Device {
	t.Helper()
	if os.Getenv("AIKIT_FENCE_PROBE") == "" {
		t.Skip("set AIKIT_FENCE_PROBE=1 to run the coherent(system) fence probe (perf timing)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	if objc.Send[uintptr](d.id, fenceSelSupportsFamily, uintptr(mtlGPUFamilyMetal3))&0xff == 0 {
		t.Fatalf("%s does not support MTLGPUFamilyMetal3 — MLX would not enable the fast fence here", d.Name())
	}
	return d
}

// TestFenceProbe_compileMSL32 is the feasibility gate: does the runtime compiler build an MSL 3.2
// kernel that stores through coherent(system) on this OS/GPU? If not, the fence is unavailable on
// this machine and the compiler error is the answer. MSL 3.1 is tried too, for the record of
// whether 3.2 is really the floor.
func TestFenceProbe_compileMSL32(t *testing.T) {
	d := fenceProbeGate(t)
	defer d.ReleaseObjects()
	t.Logf("device %s", d.Name())
	if _, err := d.CompileLibrary(fenceProbeSrc, MSL3_1); err != nil {
		t.Logf("MSL 3.1: rejected (expected) — %s", strings.SplitN(err.Error(), "\n", 2)[0])
	} else {
		t.Logf("MSL 3.1: also compiles")
	}
	lib, err := d.CompileLibrary(fenceProbeSrc, msl3_2)
	if err != nil {
		t.Fatalf("MSL 3.2 coherent(system) does NOT compile on this machine: %v", err)
	}
	for _, fn := range []string{"tick", "tick_fence", "tick_fence_plain"} {
		if _, err := d.NewComputePipeline(lib, fn); err != nil {
			t.Fatalf("pipeline %s: %v", fn, err)
		}
	}
	t.Logf("MSL 3.2: compiles; tick, tick_fence and tick_fence_plain pipelines built")
}

// encodeTick appends one compute pass: pipeline p over one thread, out at buffer(0), v at
// buffer(1) via setBytes (so each pass in a command buffer carries its own value), and the fence
// word at buffer(2) when fence != 0.
func encodeTick(cb objc.ID, p Pipeline, out Buffer, v uint32, fence objc.ID) {
	enc := cb.Send(selComputeEncoder)
	enc.Send(selSetPipeline, p.id)
	enc.Send(selSetBuffer, out.id, uintptr(0), uintptr(0))
	enc.Send(fenceSelSetBytes, unsafe.Pointer(&v), uintptr(4), uintptr(1))
	if fence != 0 {
		enc.Send(selSetBuffer, fence, uintptr(0), uintptr(2))
	}
	one := mtlSize{w: 1, h: 1, d: 1}
	enc.Send(selDispatchThreads, unsafe.Pointer(&one), unsafe.Pointer(&one))
	runtime.KeepAlive(&v)
	runtime.KeepAlive(&one)
	enc.Send(selEndEncoding)
}

// spinUntil busy-waits until done() reports true: no sleep, no yield — the user-space wake path is
// the thing under test. Bounded twice (iteration cap, and a host-clock deadline checked every 1024
// spins) so a fence that never lands fails loudly in seconds.
func spinUntil(done func() bool) (iters uint64, ok bool) {
	deadline := hostNs() + fenceSpinTimeout
	for iters = 0; iters < fenceSpinCap; iters++ {
		if done() {
			return iters, true
		}
		if iters&1023 == 1023 && hostNs() > deadline {
			return iters, false
		}
	}
	return iters, false
}

// fenceArm accumulates one arm's per-boundary samples, in µs.
type fenceArm struct {
	name                     string
	wake, done, lead         []float64 // CPU knows; driver reports completion (B); done − wake (B)
	gpu, launch, tail        []float64 // GPUEnd−GPUStart; GPUStart−commit; CPU knows − GPUEnd (A, B)
	cpu, enc                 []float64 // thread CPU time inside the wake window; encode time
	spins                    []float64 // spin iterations (B, C)
	stalePayload, misaligned int
}

func usBetween(a, b uint64) float64 { return float64(int64(b-a)) / 1e3 }

func pctl(xs []float64, p float64) float64 {
	s := slices.Clone(xs)
	slices.Sort(s)
	i := int(math.Ceil(p/100*float64(len(s)))) - 1
	return s[max(0, min(i, len(s)-1))]
}

func mean(xs []float64) float64 {
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

func summary(xs []float64) string {
	if len(xs) == 0 {
		return "n/a"
	}
	return fmt.Sprintf("p50 %8.1f  p90 %8.1f  p99 %8.1f  mean %8.1f", pctl(xs, 50), pctl(xs, 90), pctl(xs, 99), mean(xs))
}

func envInt(t *testing.T, key string, def int) int {
	t.Helper()
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil || v <= 0 {
		t.Fatalf("%s=%q: want a positive integer", key, s)
	}
	return v
}

// TestFenceProbe_wakeLatency is the microbenchmark. See the file header for the arms, the
// decision line, and the run commands.
func TestFenceProbe_wakeLatency(t *testing.T) {
	d := fenceProbeGate(t)
	runtime.LockOSThread() // one OS thread throughout: pools are per-thread, and CPU time is per-thread
	defer runtime.UnlockOSThread()
	loadFenceClocks()

	n := envInt(t, "AIKIT_FENCE_N", 1000)
	block := min(envInt(t, "AIKIT_FENCE_BLOCK", 50), n)
	rounds := (n + block - 1) / block
	n = rounds * block

	lib, err := d.CompileLibrary(fenceProbeSrc, msl3_2)
	if err != nil {
		t.Fatalf("MSL 3.2 coherent(system) does not compile here (see TestFenceProbe_compileMSL32): %v", err)
	}
	pTick, err := d.NewComputePipeline(lib, "tick")
	if err != nil {
		t.Fatal(err)
	}
	pFence, err := d.NewComputePipeline(lib, "tick_fence")
	if err != nil {
		t.Fatal(err)
	}
	pFencePlain, err := d.NewComputePipeline(lib, "tick_fence_plain")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	out := NewBufferLenOf[uint32](d, 1)
	fenceBuf := NewBufferLenOf[uint32](d, 1)
	outWord, fenceWord := &out.U32s()[0], &fenceBuf.U32s()[0]
	atomic.StoreUint32(outWord, 0)
	atomic.StoreUint32(fenceWord, 0)
	ev := d.NewSharedEvent()
	defer func() {
		ev.id.Send(selRelease)
		d.ReleaseAll()
		d.ReleaseObjects()
	}()

	// Clock cost, so the reader can tell what the instrumentation itself adds to each sample.
	const calN = 10000
	c0 := hostNs()
	for range calN {
		_ = hostNs()
	}
	clockNs := float64(hostNs()-c0) / calN

	var seq uint32 // payload / fence sequence, monotonic across arms
	var evv uint64 // shared-event value base, monotonic (an MTLSharedEvent must not go backwards)
	a, b, b0, c := &fenceArm{name: "A wait"}, &fenceArm{name: "B fence"}, &fenceArm{name: "B0 diag"}, &fenceArm{name: "C event"}

	// gpuTimes records GPU span / launch / tail for one completed command buffer and counts
	// samples whose host-clock ordering is impossible (GPU start before commit, or GPU end after
	// the driver reported completion) — non-zero means GPUStartTime is not on CLOCK_UPTIME_RAW and
	// the launch/tail columns are meaningless.
	gpuTimes := func(arm *fenceArm, cb objc.ID, tc, tKnow, tDone uint64) {
		gs := uint64(objc.Send[float64](cb, selGPUStartTime) * 1e9)
		ge := uint64(objc.Send[float64](cb, selGPUEndTime) * 1e9)
		arm.gpu = append(arm.gpu, usBetween(gs, ge))
		arm.launch = append(arm.launch, usBetween(tc, gs))
		arm.tail = append(arm.tail, usBetween(ge, tKnow))
		const tolNs = 2000 // two clock ticks plus float rounding of a ~1e5 s CFTimeInterval
		if int64(gs-tc) < -tolNs || int64(tDone-ge) < -tolNs {
			arm.misaligned++
		}
	}

	runA := func(record bool) {
		pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
		defer pool.Send(selDrain)
		for range block {
			seq++
			t0 := hostNs()
			cb := q.id.Send(selCommandBuffer)
			encodeTick(cb, pTick, out, seq, 0)
			te := hostNs()
			cpu0 := threadCPUNs()
			tc := hostNs()
			cb.Send(selCommit)
			cb.Send(selWaitCompleted)
			tw := hostNs()
			cpu1 := threadCPUNs()
			mustCmdBufOK(cb, "fence probe arm A")
			if got := atomic.LoadUint32(outWord); got != seq {
				t.Fatalf("arm A: payload %d after completion, want %d", got, seq)
			}
			if record {
				a.wake = append(a.wake, usBetween(tc, tw))
				a.done = append(a.done, usBetween(tc, tw))
				a.cpu = append(a.cpu, usBetween(cpu0, cpu1))
				a.enc = append(a.enc, usBetween(t0, te))
				gpuTimes(a, cb, tc, tw, tw)
			}
		}
	}

	// runB runs arm B (pipeline pFence) or B0 (pFencePlain) into arm.
	runB := func(arm *fenceArm, p Pipeline, record bool) {
		pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
		defer pool.Send(selDrain)
		for range block {
			seq++
			want := seq
			t0 := hostNs()
			cb := q.id.Send(selCommandBuffer)
			encodeTick(cb, p, out, seq, fenceBuf.id)
			te := hostNs()
			cpu0 := threadCPUNs()
			tc := hostNs()
			cb.Send(selCommit)
			iters, ok := spinUntil(func() bool { return atomic.LoadUint32(fenceWord) == want })
			tw := hostNs()
			cpu1 := threadCPUNs()
			payload := atomic.LoadUint32(outWord)
			cb.Send(selWaitCompleted)
			td := hostNs()
			mustCmdBufOK(cb, "fence probe arm "+arm.name)
			if !ok {
				t.Fatalf("arm %s: fence word never reached %d (%d spins, %.0f µs); after completion it reads %d — "+
					"the coherent(system) store is not reaching the CPU before completion",
					arm.name, want, iters, usBetween(tc, tw), atomic.LoadUint32(fenceWord))
			}
			if record {
				if payload != want {
					arm.stalePayload++
				}
				arm.wake = append(arm.wake, usBetween(tc, tw))
				arm.done = append(arm.done, usBetween(tc, td))
				arm.lead = append(arm.lead, usBetween(tw, td))
				arm.cpu = append(arm.cpu, usBetween(cpu0, cpu1))
				arm.enc = append(arm.enc, usBetween(t0, te))
				arm.spins = append(arm.spins, float64(iters))
				gpuTimes(arm, cb, tc, tw, td)
			}
		}
	}

	// runC: one command buffer per block. A primer pass signals evv+1 so the CPU knows the buffer
	// is live on the GPU before the first timed boundary; boundary i then waits for ack evv+2i+2,
	// runs tick, and signals evv+2i+3.
	runC := func(record bool) {
		pool := objc.ID(objc.GetClass("NSAutoreleasePool")).Send(selAlloc).Send(selInit)
		defer pool.Send(selDrain)
		cb := q.id.Send(selCommandBuffer)
		seq++
		encodeTick(cb, pTick, out, seq, 0)
		cb.Send(selEncodeSignalEvent, ev.id, uintptr(evv+1))
		for i := range block {
			cb.Send(selEncodeWaitForEvent, ev.id, uintptr(evv+uint64(2*i+2)))
			encodeTick(cb, pTick, out, seq+1+uint32(i), 0)
			cb.Send(selEncodeSignalEvent, ev.id, uintptr(evv+uint64(2*i+3)))
		}
		cb.Send(selCommit)
		fail := func(what string, iters uint64) {
			ev.SetValue(evv + uint64(2*block+2)) // release every remaining GPU wait so the buffer can finish
			cb.Send(selWaitCompleted)
			t.Fatalf("arm C: %s never signalled (%d spins); event reads %d", what, iters, ev.Value())
		}
		if iters, ok := spinUntil(func() bool { return ev.Value() >= evv+1 }); !ok {
			fail("primer", iters)
		}
		for i := range block {
			sig := evv + uint64(2*i+3)
			cpu0 := threadCPUNs()
			ta := hostNs()
			ev.SetValue(evv + uint64(2*i+2))
			iters, ok := spinUntil(func() bool { return ev.Value() >= sig })
			tw := hostNs()
			cpu1 := threadCPUNs()
			if !ok {
				fail(fmt.Sprintf("boundary %d", i), iters)
			}
			if record {
				if atomic.LoadUint32(outWord) != seq+1+uint32(i) {
					c.stalePayload++
				}
				c.wake = append(c.wake, usBetween(ta, tw))
				c.cpu = append(c.cpu, usBetween(cpu0, cpu1))
				c.spins = append(c.spins, float64(iters))
			}
		}
		cb.Send(selWaitCompleted)
		mustCmdBufOK(cb, "fence probe arm C")
		seq += uint32(block)
		evv += uint64(2*block + 2)
	}

	arms := []func(bool){
		runA,
		func(record bool) { runB(b, pFence, record) },
		func(record bool) { runB(b0, pFencePlain, record) },
		runC,
	}
	for _, run := range arms { // warm-up block per arm, discarded
		run(false)
	}
	t0 := hostNs()
	for r := range rounds {
		for k := range arms { // rotate the order each round so no arm always follows the same one
			arms[(r+k)%len(arms)](true)
		}
	}
	wall := usBetween(t0, hostNs()) / 1e6

	label := "RESULT"
	if n < 1000 {
		label = "EXPLORATORY, NOT A RESULT"
	}
	t.Logf("[%s] %s, n=%d boundaries/arm, %d interleaved rounds of %d, %.1f s; clock read %.0f ns",
		label, d.Name(), n, rounds, block, wall, clockNs)
	t.Logf("all columns µs per boundary")
	for _, arm := range []*fenceArm{a, b, b0, c} {
		t.Logf("%-8s wake    %s", arm.name, summary(arm.wake))
		if arm == b || arm == b0 {
			t.Logf("%-8s done    %s", arm.name, summary(arm.done))
			t.Logf("%-8s done−wake %s  (>0: the spin returns before the driver reports completion)", arm.name, summary(arm.lead))
		}
		if arm != c {
			t.Logf("%-8s GPU     %s", arm.name, summary(arm.gpu))
			t.Logf("%-8s launch  %s  (commit → GPUStartTime)", arm.name, summary(arm.launch))
			t.Logf("%-8s tail    %s  (GPUEndTime → CPU knows)", arm.name, summary(arm.tail))
			t.Logf("%-8s encode  %s", arm.name, summary(arm.enc))
		} else {
			t.Logf("%-8s GPU     n/a — one command buffer per block; wake is ack → signal observed", arm.name)
		}
		t.Logf("%-8s CPU     %s  (thread CPU time inside the wake window; %.0f%% of wall)",
			arm.name, summary(arm.cpu), 100*mean(arm.cpu)/mean(arm.wake))
		if arm.spins != nil {
			t.Logf("%-8s spins   %s  (iterations)", arm.name, summary(arm.spins))
		}
		if arm.stalePayload > 0 {
			t.Logf("%-8s STALE PAYLOAD on %d/%d boundaries: the word written before the signal was not yet visible when the signal was",
				arm.name, arm.stalePayload, len(arm.wake))
		}
		if arm.misaligned > 0 {
			t.Logf("%-8s CLOCK MISALIGNED on %d/%d samples: GPUStartTime is not on CLOCK_UPTIME_RAW here — ignore launch/tail",
				arm.name, arm.misaligned, len(arm.wake))
		}
	}
	saving := pctl(a.wake, 50) - pctl(b.wake, 50)
	t.Logf("fence saving vs A: p50 %.1f µs (A %.1f − B %.1f); B0 vs A %+.1f µs (diagnostic); event C vs A: p50 %+.1f µs; earlier event probe recorded ~260 µs/boundary in a real forward",
		saving, pctl(a.wake, 50), pctl(b.wake, 50), pctl(b0.wake, 50)-pctl(a.wake, 50), pctl(c.wake, 50)-pctl(a.wake, 50))
	switch {
	case b.stalePayload > 0:
		t.Errorf("DECISION: not applied — arm B saw a stale coherent(system) payload on %d/%d boundaries, so its wake time is not a usable fence", b.stalePayload, len(b.wake))
	case n < 1000:
		t.Logf("DECISION: not applied — n=%d < 1000 is a smoke; run the night command (AIKIT_FENCE_N=5000)", n)
	case saving < 25:
		t.Logf("DECISION: KILL — p50 saving %.1f µs < 25 µs pre-registered line", saving)
	default:
		t.Logf("DECISION: REPORT — p50 saving %.1f µs ≥ 25 µs; goinfer decides whether to build it into paged MoE", saving)
	}
}
