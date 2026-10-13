package kernels

// PagedFenceSrc follows aikit's C-B03 probe (gpu/metal_fence_probe_test.go, after MLX's kernels/fence.metal): the system
// thread scope is not public MSL, so it is built from the compiler's __METAL_MEMORY_SCOPE_SYSTEM__. The router wrote its ids
// with plain stores in an earlier dispatch of the same serial encoder, and a system-scope fence does not publish non-coherent
// stores (the probe saw stale payloads), so this kernel copies the ids through coherent(system) stores into a mirror the host
// reads, fences, then stores the sequence number the host spins on.
// History: docs/code-notes/metal.md#pagedFenceSrc.
const PagedFenceSrc = `
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
