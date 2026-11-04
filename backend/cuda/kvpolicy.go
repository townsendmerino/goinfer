//go:build cuda

package cuda

import (
	"sync/atomic"

	"github.com/townsendmerino/goinfer/decoder"
)

// CUDA's f16 KV cache policy (docs/tasks/task-cuda-f16-kv-2026-10.md, docs/measurements/kv4-cuda-2026-10-21.md §2 G5). Every kernel that reads or writes the
// cache has an f16 twin (kvtype.cuh, the *_kv16.ptx modules); the drafter's own cache does not, and resolveCUDAKV keeps a drafter load at f32. The decision is made
// from the architecture's facts and the options BEFORE any pricing: the fit guard, Plan, the context cap and the slot count all depend on the KV width. The tables
// are decoder.CUDAKVF16Decline (what f16 is delivered for) and decoder.CUDAKVF16AutoDecline (what auto keeps at f32); this registers them with CUDA's auto switch.
func init() {
	decoder.RegisterKVF16Policy("cuda", decoder.KVF16Policy{
		Decline:     decoder.CUDAKVF16Decline,
		Auto:        func() bool { return kvF16Auto.Load() },
		AutoDecline: decoder.CUDAKVF16AutoDecline,
	})
	kvF16Auto.Store(true)
}

// kvF16Auto is whether "auto" means f16 on CUDA for a covered family. True by OWNER DECISION of 2026-10-13, over two registered gates that did not pass (the task doc,
// "Owner override"); the cuda tests pin it false in init so every exact-cache contract test keeps its f32 cache.
var kvF16Auto atomic.Bool
