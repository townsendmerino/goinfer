package decoder

import "sync/atomic"

// f16 KV on CUDA (docs/tasks/task-cuda-f16-kv-2026-10.md). CUDA's resident cache is f32 unless this file says otherwise: the attention and
// KV-store kernels have f16 twins (cuda/kvtype.cuh, the *_kv16.ptx modules), and everything else that touches the cache (the windowed ring,
// the MLA latent, the drafter's own cache, the VL image prefill, the pairwise-RoPE store) does not, so a build that needs any of those stays f32.
//
// The decision is made here, from the architecture and the options, BEFORE any pricing: the fit guard, Plan, the context cap and the slot count
// all depend on the KV width, and a resident that learned at build time that it must be f32 after all would have been sized for f16.

// cudaKVF16Cleared lists the families `--kv auto` (the default) puts on an f16 KV cache on CUDA: the dense decoders the R6 gates measured the
// attention kernels on. `--kv f16` asks for f16 on any family cudaKVF16Decline lets through, which today is the same set; a family joins the
// default by passing the fidelity and overflow gates, not by compiling.
func cudaKVF16Cleared(a *Architecture) bool {
	if a == nil {
		return false
	}
	switch a.Name {
	case "llama", "mistral", "qwen2", "qwen3", "phi3":
		return true
	}
	return false
}

// cudaKVF16Decline returns why a CUDA resident cannot hold this model's KV cache as f16, or "" when it can. Each reason names a path whose
// kernels read or write the cache and have no f16 twin yet.
func cudaKVF16Decline(a *Architecture, o Options) string {
	switch {
	case a == nil:
		return "no architecture"
	case !cudaKVF16Cleared(a):
		return "f16 KV on CUDA covers llama, mistral, qwen2, qwen3 and phi3 so far; " + a.Name + " keeps f32"
	case a.mla != nil:
		return "an MLA latent cache has no f16 form"
	case a.MoE != nil:
		return "MoE models keep f32 KV so far"
	case o.MoECacheExperts:
		return "the paged-experts build keeps f32 KV"
	case a.SlidingWindow > 0:
		return "sliding-window attention (the windowed KV ring) has no f16 form"
	case a.pairwiseRoPE():
		return "pairwise RoPE has no f16 store kernel"
	case len(a.MRopeSection) > 0:
		return "m-RoPE image prefill has no f16 kernels"
	case a.qwen35 != nil || a.gemma4 != nil || a.laguna != nil || a.granite != nil || a.lfm2 != nil:
		return "this family's attention is not covered"
	}
	return ""
}

// cudaKVBackend reports whether o's load will run on the CUDA backend: asked for by name and compiled into this binary.
func cudaKVBackend(o Options) bool {
	if o.Backend != "cuda" {
		return false
	}
	for _, b := range CompiledBackends() {
		if b == "cuda" {
			return true
		}
	}
	return false
}

// kvF16For is whether this load's resident KV cache is f16 for architecture a under these options. "f16" asks for it, "auto" (and the zero
// value) takes the backend's default, and "f32" refuses it. Only the CUDA branch is new: WebGPU honours "f16" as before and its default stays
// f32, Metal's f16 is unconditional (ResidentKVPrecision), and the CPU has no f16 cache.
func (o Options) kvF16For(a *Architecture) bool {
	if cudaKVBackend(o) {
		switch o.KVPrecision {
		case "f16":
			return cudaKVF16Decline(a, o) == ""
		case "", "auto":
			return cudaKVF16Default.Load() && cudaKVF16Decline(a, o) == ""
		}
		return false
	}
	return o.KVPrecision == "f16"
}

// cudaKVF16Default is whether "auto" means f16 on CUDA for a cleared family. It is true by OWNER DECISION of 2026-10-13, over two registered gates that
// did not pass (docs/tasks/task-cuda-f16-kv-2026-10.md, "Owner override"): the fidelity decision cell read PARKED, and the paired speed check was stopped
// before its deep cells ran. Tests that need the bit-exact cache pin KVPrecision "f32".
var cudaKVF16Default atomic.Bool

func init() { cudaKVF16Default.Store(true) }

// KVF16DeclineReason is the reason a CUDA load asked for f16 KV (or left it on auto) is held at f32, "" when it is not held back or the load
// is not on CUDA. serve's banner prints it, so a model that quietly stays f32 under the f16 default says why.
func (m *Model) KVF16DeclineReason() string {
	if m == nil || m.be == nil || m.be.Name() != "cuda" || m.kvF16 {
		return ""
	}
	return m.kvF16Decline
}
