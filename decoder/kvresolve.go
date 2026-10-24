package decoder

import "fmt"

// The KV precision a load delivers (docs/tasks/task-kv-precision-everywhere-2026-10.md, phase 0). ResolveKV is the one answer every
// backend's build, the fit guard's pricing, the banner and serve's request log read; an explicit precision it cannot deliver refuses the
// load (KVRefusal), so no path runs a precision the caller did not ask for without saying so.

// Precisions a KV cache can hold. kvPrecNone is a model that keeps no attention KV (Mamba-only), which accepts any request.
const (
	kvPrecF32  = "f32"
	kvPrecF16  = "f16"
	kvPrecI8   = "i8"
	kvPrecNone = "n/a"
	kvPrecAuto = "auto"
)

// kvExplicit reports whether req names a precision rather than leaving the pick to the backend.
func kvExplicit(req string) bool { return req == kvPrecF32 || req == kvPrecF16 || req == kvPrecI8 }

// gpuKVRequested is the request for a resident (GPU) cache: Options.KVPrecision, "auto" when unset.
func gpuKVRequested(o Options) string {
	if o.KVPrecision == "" {
		return kvPrecAuto
	}
	return o.KVPrecision
}

// cpuKVRequested is the request for the CPU (host) cache: Options.KVQuant, which serve and goinfer-chat set from --kv (loadflags.CPUKV),
// "auto" when unset.
func cpuKVRequested(o Options) string {
	if o.KVQuant == "" {
		return kvPrecAuto
	}
	return o.KVQuant
}

// keepsNoKV reports whether no layer of the model keeps attention K/V (a Mamba-only family).
func (a *Architecture) keepsNoKV() bool {
	for i := 0; i < a.NumLayers; i++ {
		if !a.hasNoAttentionKVAt(i) {
			return false
		}
	}
	return a.NumLayers > 0
}

// ResolveKV is the precision of the KV cache backend builds for arch under opts: "f32", "f16", "i8", "n/a" for a model that keeps no
// KV, or "" when nothing can be built as asked. reason says why whenever delivered is not what was requested: the backend's pick
// under auto, or why an explicit request cannot be met (which Load refuses, KVRefusal). backend "cpu" is the CPU (host) cache, which a
// load uses when no resident runner is active, whatever Options.Backend names.
func ResolveKV(backend string, arch *Architecture, opts Options) (delivered, reason string) {
	if resolveKVHook != nil {
		if got, why, ok := resolveKVHook(backend, opts); ok {
			return got, why
		}
	}
	if arch != nil && arch.keepsNoKV() {
		return kvPrecNone, "this model keeps no attention KV"
	}
	if backend == "" || backend == "cpu" {
		return resolveCPUKV(arch, opts)
	}
	req := gpuKVRequested(opts)
	o := opts
	o.Backend = backend
	switch backend {
	case "cuda":
		return resolveCUDAKV(arch, o, req)
	case "metal":
		if arch != nil && arch.mla != nil {
			return kvPrecF32, "an MLA model keeps its latent cache in f32 on Metal"
		}
		switch req {
		case kvPrecF32:
			return kvPrecF32, ""
		case kvPrecI8:
			return kvPrecI8, ""
		case kvPrecAuto:
			return kvPrecF16, "auto is f16 on Metal"
		}
		return kvPrecF16, ""
	}
	// WebGPU and any other resident backend build what is asked (its family and prefill coverage is phase 5's survey); auto is f32.
	switch req {
	case kvPrecF16, kvPrecI8, kvPrecF32:
		return req, ""
	}
	return kvPrecF32, "auto is f32 here"
}

func resolveCUDAKV(arch *Architecture, o Options, req string) (string, string) {
	if arch == nil {
		return kvPrecF32, "no architecture"
	}
	p := kvPolicies["cuda"]
	switch req {
	case kvPrecF32:
		return kvPrecF32, ""
	case kvPrecI8:
		return kvPrecF32, "CUDA has no int8 KV cache until phase 2"
	case kvPrecF16:
		if o.Drafter {
			return kvPrecF32, "the drafter needs f32 KV until phase 6"
		}
		if why := p.Decline(arch.kvFacts(), o); why != "" {
			return kvPrecF32, why
		}
		return kvPrecF16, ""
	}
	// auto
	switch {
	case o.Drafter:
		return kvPrecF32, "the block drafter (--drafter) keeps f32 K/V beside the target's attention kernels"
	case !p.Auto():
		return kvPrecF32, "auto is f32 on CUDA in this build"
	}
	if why := p.Decline(arch.kvFacts(), o); why != "" {
		return kvPrecF32, why
	}
	if p.AutoDecline != nil {
		if why := p.AutoDecline(arch.kvFacts(), o); why != "" {
			return kvPrecF32, why
		}
	}
	if p.AutoReason != nil {
		if why := p.AutoReason(arch.kvFacts(), o); why != "" {
			return kvPrecF16, why
		}
	}
	return kvPrecF16, "auto is f16 on CUDA for this family"
}

func resolveCPUKV(arch *Architecture, o Options) (string, string) {
	req := cpuKVRequested(o)
	if (req == kvPrecF16 || req == kvPrecI8) && o.CPUBatchDecode == CPUBatchOn {
		return "", "CPU batched decode (CPUBatchDecode on) reads f32 KV only until phase 4"
	}
	switch req {
	case kvPrecF16:
		if arch != nil && arch.kvF16OK() {
			return kvPrecF16, ""
		}
		return kvPrecF32, cpuKVDecline(arch)
	case kvPrecI8:
		if arch != nil && arch.kvInt8OK() {
			return kvPrecI8, ""
		}
		return kvPrecF32, cpuKVDecline(arch)
	case kvPrecF32:
		return kvPrecF32, ""
	}
	return kvPrecF32, "auto is f32 on the CPU"
}

// cpuKVDecline is why the CPU cache holds this model at f32 whatever is asked: the families whose attention is not the batched kernel.
func cpuKVDecline(arch *Architecture) string {
	switch {
	case arch == nil:
		return "no architecture"
	case arch.MoE != nil:
		return "MoE families keep an f32 CPU cache until phase 4"
	case arch.ownLoopFamily():
		return "this family's own forward loop keeps an f32 CPU cache until phase 4"
	}
	return "this family's attention keeps an f32 CPU cache until phase 4"
}

// KVRefusal is Load's refusal of an explicit KV precision the path the load resolved to cannot deliver (D1 of the task doc).
type KVRefusal struct {
	Path      string // "cuda", "metal", "webgpu" (the resident cache) or "cpu"
	Requested string
	Delivered string // what would have run ("" when nothing could)
	Reason    string
}

func (e *KVRefusal) Error() string {
	return fmt.Sprintf("--kv %s cannot be delivered on the %s KV cache: %s", e.Requested, e.Path, e.Reason)
}

// kvRefusal is the refusal of opts's explicit request on path, or nil when it is delivered or not explicit.
func kvRefusal(path string, arch *Architecture, opts Options) *KVRefusal {
	req := gpuKVRequested(opts)
	if path == "" || path == "cpu" {
		path, req = "cpu", cpuKVRequested(opts)
	}
	if !kvExplicit(req) {
		return nil
	}
	got, why := ResolveKV(path, arch, opts)
	if got == req || got == kvPrecNone {
		return nil
	}
	return &KVRefusal{Path: path, Requested: req, Delivered: got, Reason: why}
}

// cudaKVF16Delivered lists the families whose every KV-touching CUDA path has an f16 kernel: `--kv f16` is delivered for them whatever
// the quality reads (the precision doc's contract, item 4). A family group joins with its exact-input identity and delivered-bytes gates
// (docs/measurements/kv4-cuda-2026-10-21.md §2, G5), never by compiling.
func cudaKVF16Delivered(f KVFacts) bool {
	switch f.Family {
	case "llama", "mistral", "qwen2", "qwen3", "phi3", "qwen2_moe", "qwen3_moe", "mixtral", "gemma2", "gemma3", "gpt-oss", "cohere",
		"cohere2", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe", "qwen3_next", "gemma4", "deepseek_v2", "deepseek_v3", "kimi_k2":
		return true
	}
	return false
}

// cudaKVF16AutoCleared reports whether `--kv auto` puts a delivered model on f16 KV on CUDA: the families whose quality gate passed (the
// owner's D7: f16 becomes the default for each family group as it clears its G4, docs/measurements/kv4-cuda-2026-10-21.md §2, G5). A
// model carrying any trait whose group has not cleared stays f32 under auto, whatever else it carries; f16 is delivered on request.
func cudaKVF16AutoCleared(f KVFacts, o Options) bool {
	if f.PairwiseRoPE || f.Hybrid || f.MLA {
		return false
	}
	switch f.Family {
	case "llama", "mistral", "qwen2", "qwen3", "phi3", "gemma2", "gemma3", "qwen2_5_vl", "qwen3_vl", "qwen2_moe", "qwen3_moe", "mixtral",
		"qwen3_vl_moe", "gpt-oss":
		return true
	}
	return false
}

// CUDAKVF16Decline returns why a CUDA resident cannot hold this model's KV cache as f16, or "" when it can: a family outside
// cudaKVF16Delivered. Every kernel that reads or writes the cache, MLA's latent included, has an f16 twin; a family joins the list
// with its group's gates, so one whose paths were never checked stays f32. It lives here so ResolveKV answers for CUDA without the
// cuda package; the cuda package registers it as its policy.
func CUDAKVF16Decline(f KVFacts, _ Options) string {
	if !cudaKVF16Delivered(f) {
		return "f16 KV on CUDA covers llama, mistral, qwen2, qwen3, phi3, qwen2_moe, qwen3_moe, mixtral, gemma2, gemma3, gpt-oss, cohere, cohere2, " +
			"qwen2_5_vl, qwen3_vl, qwen3_vl_moe, qwen3_5, qwen3_5_moe, qwen3_next, gemma4, deepseek_v2, deepseek_v3 and kimi_k2 so far; " +
			f.Family + " keeps f32"
	}
	return ""
}

// CUDAKVF16AutoDecline returns why `--kv auto` keeps a model CUDAKVF16Decline lets through at f32 on CUDA, or "" when auto picks f16.
// A group whose G4 read DOES NOT PASS says so, with the reading (docs/measurements/kv4-cuda-2026-10-21.md §3, "KV4b (G5), G4").
func CUDAKVF16AutoDecline(f KVFacts, o Options) string {
	switch {
	case cudaKVF16AutoCleared(f, o):
		return ""
	case f.Hybrid:
		return "f16 KV is delivered on request (--kv f16); auto keeps f32: the hybrids' quality gate did not pass (mean KL 1.21x f32's at 3,900 keys)"
	}
	return "f16 KV is delivered on request (--kv f16) for this build; auto keeps f32 until its quality gate passes"
}

// CUDAKVF16AutoReason is the reason `--kv auto` gives when it picks f16 on CUDA, "" for the plain one. The sliding-window group's says that
// auto moved by the owner's decision over an AMBIGUOUS G4 cell (docs/measurements/kv4-cuda-2026-10-21.md §5, D-KV4-6).
func CUDAKVF16AutoReason(f KVFacts, _ Options) string {
	if f.SlidingWindow {
		return "auto is f16 on CUDA for this family (sliding window: by the owner's decision over an AMBIGUOUS quality reading at 8,000 keys)"
	}
	return ""
}

// The CUDA policy ResolveKV reads when the cuda package is not linked in, as in the decoder's own tests and the generated KV matrix:
// auto is f16 for a cleared family (the owner's decision of 2026-10-13). The cuda package registers the same table with its own auto
// switch, which its tests pin to f32.
func init() {
	RegisterKVF16Policy("cuda", KVF16Policy{Decline: CUDAKVF16Decline, Auto: func() bool { return true }, AutoDecline: CUDAKVF16AutoDecline,
		AutoReason: CUDAKVF16AutoReason})
}

// kvPath is the cache a generation of this model writes: the resident backend's when a runner is active, else the CPU's.
func (m *Model) kvPath() string {
	if m.resident != nil && m.be != nil {
		return m.be.Name()
	}
	return "cpu"
}

// kvDeliveredOrRefuse is Load's check, once residency is decided, that every explicit precision the load asked for is the one its
// caches hold (D1). The CPU cache is checked as well under an adapter, whose session path takes it beside a resident.
func (m *Model) kvDeliveredOrRefuse() error {
	if m.w == nil {
		return nil
	}
	if r := kvRefusal(m.kvPath(), m.w.arch, m.kvOpts); r != nil {
		return r
	}
	if m.kvPath() != "cpu" && m.kvOpts.LoRA != "" {
		if r := kvRefusal("cpu", m.w.arch, m.kvOpts); r != nil {
			return r
		}
	}
	return nil
}

// KVPrecisionReport is one cache's requested and delivered KV precision, and why they differ ("" when they do not).
type KVPrecisionReport struct {
	Cache     string // "resident" or "cpu"
	Requested string
	Delivered string
	Reason    string
}

// KVPrecisionReports is what serve's banner prints for the load: the resident cache when a runner is active, and the CPU cache when
// no runner is, or when an adapter's session path can take it.
func (m *Model) KVPrecisionReports() []KVPrecisionReport {
	if m.w == nil {
		return nil
	}
	var out []KVPrecisionReport
	if path := m.kvPath(); path != "cpu" {
		got, why := ResolveKV(path, m.w.arch, m.kvOpts)
		out = append(out, KVPrecisionReport{Cache: "resident", Requested: gpuKVRequested(m.kvOpts), Delivered: got, Reason: why})
		if m.kvOpts.LoRA == "" {
			return out
		}
	}
	got, why := ResolveKV("cpu", m.w.arch, m.kvOpts)
	return append(out, KVPrecisionReport{Cache: "cpu", Requested: cpuKVRequested(m.kvOpts), Delivered: got, Reason: why})
}

// resolveKVHook, set only by a test (goinfer_testhooks), answers ResolveKV for the cases it claims (ok), so a test can make the
// resolver and the raw request disagree and see which one a backend's build follows.
var resolveKVHook func(backend string, opts Options) (delivered, reason string, ok bool)

// KVDeliveredFor is the precision ResolveKV gives this model's KV cache on backend: what a backend's resident build allocates.
func (m *Model) KVDeliveredFor(backend string) string {
	if m.w == nil {
		return ""
	}
	got, _ := ResolveKV(backend, m.w.arch, m.kvOpts)
	return got
}

// KVDelivered is the precision of the cache a generation of this model writes (the request log's kv field).
func (m *Model) KVDelivered() string {
	if m.w == nil {
		return ""
	}
	got, _ := ResolveKV(m.kvPath(), m.w.arch, m.kvOpts)
	return got
}
