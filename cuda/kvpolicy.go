//go:build cuda

package cuda

import (
	"sync/atomic"

	"github.com/townsendmerino/goinfer/decoder"
)

// CUDA's f16 KV cache policy (docs/tasks/task-cuda-f16-kv-2026-10.md). The attention and KV-store kernels have f16 twins (kvtype.cuh, the *_kv16.ptx modules); everything
// else that touches the cache (the windowed ring, the MLA latent, the drafter's own cache, the VL image prefill, the pairwise-RoPE store) does not, so a build that needs
// any of those stays f32. The decision is made from the architecture's facts and the options BEFORE any pricing: the fit guard, Plan, the context cap and the slot count
// all depend on the KV width. It lives here so a change to it is a CUDA change, not a decoder one.
func init() {
	decoder.RegisterKVF16Policy("cuda", decoder.KVF16Policy{
		Decline: cudaKVF16Decline,
		Auto:    func() bool { return kvF16Auto.Load() },
	})
	kvF16Auto.Store(true)
}

// kvF16Auto is whether "auto" means f16 on CUDA for a covered family. True by OWNER DECISION of 2026-10-13, over two registered gates that did not pass (the task doc,
// "Owner override"); the cuda tests pin it false in init so every exact-cache contract test keeps its f32 cache.
var kvF16Auto atomic.Bool

// cudaKVF16Cleared lists the families `--kv auto` puts on an f16 KV cache: the dense decoders the R6 gates measured the attention kernels on. `--kv f16` asks for f16 on
// any family cudaKVF16Decline lets through, which today is the same set; a family joins by passing the fidelity and overflow gates, not by compiling.
func cudaKVF16Cleared(f decoder.KVFacts) bool {
	switch f.Family {
	case "llama", "mistral", "qwen2", "qwen3", "phi3":
		return true
	}
	return false
}

// cudaKVF16Decline returns why a CUDA resident cannot hold this model's KV cache as f16, or "" when it can. Each reason names a path whose kernels read or write the
// cache and have no f16 twin yet.
func cudaKVF16Decline(f decoder.KVFacts, o decoder.Options) string {
	switch {
	case !cudaKVF16Cleared(f):
		return "f16 KV on CUDA covers llama, mistral, qwen2, qwen3 and phi3 so far; " + f.Family + " keeps f32"
	case f.MLA:
		return "an MLA latent cache has no f16 form"
	case f.MoE:
		return "MoE models keep f32 KV so far"
	case o.MoECacheExperts:
		return "the paged-experts build keeps f32 KV"
	case f.SlidingWindow:
		return "sliding-window attention (the windowed KV ring) has no f16 form"
	case f.PairwiseRoPE:
		return "pairwise RoPE has no f16 store kernel"
	case f.MRoPE:
		return "m-RoPE image prefill has no f16 kernels"
	case f.Hybrid:
		return "this family's attention is not covered"
	}
	return ""
}
