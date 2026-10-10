//go:build darwin

package metal

import (
	"sync/atomic"
	"time"
)

// pagedFenceOn: a paged MoE layer's phase 1 ends with paged_fence and the host spins on its word instead of sleeping in
// waitUntilCompleted (docs/audit-metal-2026-09-30.md, C-B03). OFF: aikit's probe saw the spin wake sooner on a one-word
// payload, but on M26 the fenced token measured slower than the sleep on both a busy spin and a 10 us nap. The likely cost
// is the system-scope fence publishing a whole decode layer's writes, which the probe's one word never priced; re-measure on
// M26 before turning it on (docs/tasks/task-m26-mac-2026-10.md, "M-11"; docs/code-notes/metal.md#pagedFenceOn). Kept, gated
// bit-exact and not stale (TestPagedFence_bitExactAndNotStale).
var pagedFenceOn = false

// pagedFenceCheckForTest makes every fenced boundary also wait for its command buffer and compare the mirrored ids with
// the router's own buffer, counting mismatches (pagedFence.stale): the stale-payload check a test asserts is zero.
var pagedFenceCheckForTest = false

// pagedFenceNap is how long the host sleeps between looks at the fence word (0: a busy spin).
var pagedFenceNap = 10 * time.Microsecond

// msl3_2 is MTLLanguageVersion3_2, the first MSL with the coherent(system) address qualifier.
const msl3_2 uint = (3 << 16) | 2

// pagedFenceSrc follows aikit's C-B03 probe (gpu/metal_fence_probe_test.go, after MLX's kernels/fence.metal): the system
// thread scope is not public MSL, so it is built from the compiler's __METAL_MEMORY_SCOPE_SYSTEM__. The router wrote its ids
// with plain stores in an earlier dispatch of the same serial encoder, and a system-scope fence does not publish non-coherent
// stores (the probe saw stale payloads), so this kernel copies the ids through coherent(system) stores into a mirror the host
// reads, fences, then stores the sequence number the host spins on.
// History: docs/code-notes/metal.md#pagedFenceSrc.
const pagedFenceSrc = `
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
#include <metal_atomic>
[[kernel]] void paged_fence(device const uint* idx [[buffer(0)]], device const uint* idx2 [[buffer(1)]],
    volatile coherent(system) device uint* mirror [[buffer(2)]], volatile coherent(system) device uint* word [[buffer(3)]],
    constant uint& k [[buffer(4)]], constant uint& seq [[buffer(5)]], constant uint& has2 [[buffer(6)]]) {
  for (uint j = 0; j < k; j++) mirror[j] = idx[j];
  if (has2 != 0u) for (uint j = 0; j < k; j++) mirror[k + j] = idx2[j];
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
  word[0] = seq;
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
}
`

// pagedFence is one resident's fence: its pipeline, the id mirror (2k words: the route, and lever 3's guess), the word
// the host spins on, one sequence uniform per layer (rewritten only after that layer's previous fence was seen), and
// counters.
type pagedFence struct {
	p                    Pipeline
	mirror, word, uK     Buffer
	seq, has             []Buffer
	n                    uint32
	waits, fallbacks     int
	stale                int // pagedFenceCheckForTest: boundaries whose mirrored ids differed from the router's buffer
	spinNanos, waitNanos int64
}

// newPagedFence compiles paged_fence; nil when the fence is off or this machine's compiler rejects MSL 3.2.
func newPagedFence(d *Device, nL, k int) *pagedFence {
	if !pagedFenceOn {
		return nil
	}
	lib, err := d.CompileLibrary(pagedFenceSrc, msl3_2)
	if err != nil {
		return nil
	}
	p, err := d.NewComputePipeline(lib, "paged_fence")
	if err != nil {
		return nil
	}
	f := &pagedFence{p: p, mirror: NewBufferUint32s(d, make([]uint32, 2*k)), word: NewBufferUint32s(d, make([]uint32, 1)),
		uK: NewBufferU32(d, uint32(k))}
	for range nL {
		f.seq = append(f.seq, NewBufferU32(d, 0))
	}
	f.has = []Buffer{NewBufferU32(d, 0), NewBufferU32(d, 1)}
	return f
}

// encode appends the fence for layer l's phase 1: the route idx and, when idx2 is set, lever 3's guess, mirrored; it
// returns the sequence number the host waits for.
func (f *pagedFence) encode(e *Encoder, l int, idx, idx2 Buffer) uint32 {
	f.n++
	f.seq[l].SetU32(f.n)
	has := f.has[1]
	if idx2 == (Buffer{}) {
		idx2, has = idx, f.has[0]
	}
	e.Dispatch(f.p, 1, 1, idx, idx2, f.mirror, f.word, f.uK, f.seq[l], has)
	return f.n
}

// wait spins until the fence word reads seq. If it has not after 50 ms, it waits for the command buffer instead (a
// fallback, counted) and reports that the buffer is complete.
func (f *pagedFence) wait(e *Encoder, seq uint32) (completed bool) {
	w := f.word.U32s()
	st := time.Now()
	for spins := 0; atomic.LoadUint32(&w[0]) != seq; spins++ {
		if pagedFenceNap > 0 {
			time.Sleep(pagedFenceNap)
		}
		if (pagedFenceNap > 0 || spins&1023 == 1023) && time.Since(st) > 50*time.Millisecond {
			f.fallbacks++
			e.WaitDone()
			f.waitNanos += time.Since(st).Nanoseconds()
			return true
		}
	}
	f.waits++
	f.spinNanos += time.Since(st).Nanoseconds()
	return false
}

// ids reads k mirrored ids, the route (second false) or the guess (true).
func (f *pagedFence) ids(k int, second bool) []int {
	m := f.mirror.U32s()
	off := 0
	if second {
		off = k
	}
	out := make([]int, k)
	for j := range out {
		out[j] = int(atomic.LoadUint32(&m[off+j]))
	}
	return out
}

func u32sToInts(u []uint32) []int {
	out := make([]int, len(u))
	for i, v := range u {
		out[i] = int(v)
	}
	return out
}
