package decoder

import (
	"fmt"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// The KV precision a load delivers. ResolveKV is the one answer every
// backend's build, the fit guard's pricing, the banner and serve's request log read; an explicit precision it cannot deliver refuses the
// load (KVRefusal), so no path runs a precision the caller did not ask for without saying so.

// Precisions a KV cache can hold. kvPrecNone is a model that keeps no attention KV (Mamba-only), which accepts any request.
const (
	kvPrecF32  = mem.KVF32
	kvPrecF16  = mem.KVF16
	kvPrecI8   = mem.KVI8
	kvPrecNone = mem.KVNone
	kvPrecAuto = mem.KVAuto
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

// ResolveKV is the precision of the KV cache backend builds for arch under opts: "f32", "f16", "i8", "n/a" for a model that keeps no
// KV, or "" when nothing can be built as asked. reason says why whenever delivered is not what was requested: the backend's pick
// under auto, or why an explicit request cannot be met (which Load refuses, KVRefusal). backend "cpu" is the CPU (host) cache, which a
// load uses when no resident runner is active, whatever Options.Backend names.
func ResolveKV(backend string, arch *darch.Architecture, opts Options) (delivered, reason string) {
	if resolveKVHook != nil {
		if got, why, ok := resolveKVHook(backend, opts); ok {
			return got, why
		}
	}
	o := opts
	if backend != "" && backend != "cpu" {
		o.Backend = backend
	}
	q := mem.KVQuery{Backend: backend, Request: gpuKVRequested(opts), CPURequest: cpuKVRequested(opts),
		Drafter: o.Drafter, HasArch: arch != nil}
	if arch != nil {
		q.NoKV, q.MLA, q.OwnLoop = arch.KeepsNoKV(), arch.MLA != nil, arch.OwnLoopFamily()
		q.CPUF16OK, q.CPUI8OK = arch.KVF16OK(), arch.KVInt8OK()
		p := kvPolicies["cuda"]
		q.CUDAPolicy = mem.KVPolicy{Decline: func() string { return p.Decline(arch.KVFacts(), o) }, Auto: p.Auto}
		if p.AutoDecline != nil {
			q.CUDAPolicy.AutoDecline = func() string { return p.AutoDecline(arch.KVFacts(), o) }
		}
		if p.DeclineI8 != nil {
			q.CUDAPolicy.DeclineI8 = func() string { return p.DeclineI8(arch.KVFacts(), o) }
		}
	}
	return mem.ResolveKV(q)
}

// KVRefusal is Load's refusal of an explicit KV precision the path the load resolved to cannot deliver.
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
func kvRefusal(path string, arch *darch.Architecture, opts Options) *KVRefusal {
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

// CUDAKVF16Decline returns why a CUDA resident cannot hold this model's KV cache as f16, or "" when it can.
func CUDAKVF16Decline(f darch.KVFacts, _ Options) string { return memprice.CUDAF16Decline(f) }

// CUDAKVF16AutoDecline returns why --kv auto keeps a model CUDAKVF16Decline lets through at f32 on CUDA, or "" when auto picks f16.
func CUDAKVF16AutoDecline(f darch.KVFacts, _ Options) string { return memprice.CUDAF16AutoDecline(f) }

// CUDAKVI8Decline returns why a CUDA resident cannot hold this model's KV cache as int8 under o, or "" when it can.
func CUDAKVI8Decline(f darch.KVFacts, _ Options) string { return memprice.CUDAI8Decline(f) }

// kvPath is the cache a generation of this model writes: the resident backend's when a runner is active, else the CPU's.
func (m *Model) kvPath() string {
	if m.resident != nil && m.be != nil {
		return m.be.Name()
	}
	return "cpu"
}

// kvDeliveredOrRefuse is Load's check, once residency is decided, that every explicit precision the load asked for is the one its
// caches hold. The CPU cache is checked as well under an adapter, whose session path takes it beside a resident.
func (m *Model) kvDeliveredOrRefuse() error {
	if m.w == nil {
		return nil
	}
	if r := kvRefusal(m.kvPath(), m.w.Arch, m.opts); r != nil {
		return r
	}
	if m.kvPath() != "cpu" && m.opts.LoRA != "" {
		if r := kvRefusal("cpu", m.w.Arch, m.opts); r != nil {
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
		got, why := ResolveKV(path, m.w.Arch, m.opts)
		out = append(out, KVPrecisionReport{Cache: "resident", Requested: gpuKVRequested(m.opts), Delivered: got, Reason: why})
		if m.opts.LoRA == "" {
			return out
		}
	}
	got, why := ResolveKV("cpu", m.w.Arch, m.opts)
	return append(out, KVPrecisionReport{Cache: "cpu", Requested: cpuKVRequested(m.opts), Delivered: got, Reason: why})
}

// resolveKVHook, set only by a test (goinfer_testhooks), answers ResolveKV for the cases it claims (ok), so a test can make the
// resolver and the raw request disagree and see which one a backend's build follows.
var resolveKVHook func(backend string, opts Options) (delivered, reason string, ok bool)

// KVDeliveredFor is the precision ResolveKV gives this model's KV cache on backend: what a backend's resident build allocates.
func (m *Model) KVDeliveredFor(backend string) string {
	if m.w == nil {
		return ""
	}
	got, _ := ResolveKV(backend, m.w.Arch, m.opts)
	return got
}

// KVDelivered is the precision of the cache a generation of this model writes (the request log's kv field).
func (m *Model) KVDelivered() string {
	if m.w == nil {
		return ""
	}
	got, _ := ResolveKV(m.kvPath(), m.w.Arch, m.opts)
	return got
}
