//go:build cuda

package cuda

import (
	"sync/atomic"

	"github.com/townsendmerino/goinfer/decoder"
)

// CUDA's f16 KV cache policy (docs/tasks/task-cuda-f16-kv-2026-10.md). The attention and KV-store kernels have f16 twins (kvtype.cuh, the *_kv16.ptx modules); everything
// else that touches the cache (the windowed ring, the MLA latent, the drafter's own cache, the VL image prefill, the pairwise-RoPE store) does not, so a build that needs
// any of those stays f32. The decision is made from the architecture's facts and the options BEFORE any pricing: the fit guard, Plan, the context cap and the slot count
// all depend on the KV width. The table itself is decoder.CUDAKVF16Decline, where decoder.ResolveKV reads it; this registers it with CUDA's auto switch.
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
