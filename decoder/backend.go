package decoder

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Backend abstracts the one hot primitive the decoder forward pass is
// bound by — the big weight matmuls. Norms, RoPE, softmax and elementwise
// ops are cheap and stay on the CPU even when a GPU matmul backend is in
// use, which avoids a host↔device round-trip per layer.
//
// The default "cpu" backend is always registered. A WebGPU backend lives in
// the opt-in github.com/townsendmerino/goinfer/gpu module: importing it under
// `-tags gpu` calls RegisterBackend("webgpu", …) on init, so the decoder gains
// GPU acceleration WITHOUT pulling github.com/oliverbestmann/webgpu (cgo) into
// goinfer's core dependency graph.
type Backend interface {
	// Name identifies the backend ("cpu", "webgpu").
	Name() string
	// MatmulBT computes dst[M,N] = a[M,K] · b[N,K]ᵀ, the PyTorch [out,in]
	// weight layout the safetensors checkpoints already store (so no
	// transpose copy), matching encoder/'s matmulBT convention.
	MatmulBT(a, b, dst []float32, M, K, N int)
	// Close releases backend resources (GPU buffers, etc.). No-op on CPU.
	Close() error
}

// QuantBackend is an optional Backend extension: a backend that can run the int8xint8 (W8A8) weight matmul on-device.
// linalg.WeightMat type-asserts for it and routes the W8A8 path through it, keeping the weight resident keyed by the q8
// slice's backing pointer, and falls back to the CPU kernel when the backend does not implement it or a call declines
// (false on any GPU error, so results stay correct). The shared-activation batches go through QuantBatchBackend.
type QuantBackend interface {
	MatmulW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) bool
}

// QuantBackend4 is QuantBackend's int4 (W4A8) counterpart for the staged (non-resident) path, keyed by the
// packed-nibble slice's backing pointer. A decline (false) falls back to linalg.WeightMat's CPU W4A8 kernel.
type QuantBackend4 interface {
	MatmulW4A8(a []float32, bQ4 []byte, bScales16 []uint16, group int, dst []float32, M, K, N int) bool
}

// QuantBatchBackend additionally runs a SET of W8A8 matmuls that share one
// activation a[M,K] — the fused qkv and gate/up projections — as a single
// submit: quantize once, dispatch all ops, sync once. This is what makes the GPU
// worthwhile at decode (a token issues hundreds of tiny matmuls, each otherwise a
// separate submit+poll; batching the independent ones cuts the sync count). The
// backend keeps each op's weight resident keyed by &op.BQ[0]. Returns false to
// decline (the caller uses the CPU batch kernel).
type QuantBatchBackend interface {
	MatmulW8A8Batch(a []float32, M, K int, ops []linalg.W8A8Op) bool
}

// matmulW8A8Batch routes a shared-activation W8A8 batch through a
// QuantBatchBackend (one GPU submit) when available, else the CPU batch kernel.
// actGroup is the ops' weights' activation group (they share one activation); a per-group batch
// skips the staged GPU consult, which reads per-vector activation scales only.
func matmulW8A8Batch(be Backend, ws *linalg.Workspace, a []float32, M, K int, ops []linalg.W8A8Op, actGroup int) {
	if qb, ok := be.(QuantBatchBackend); ok && actGroup == 0 && qb.MatmulW8A8Batch(a, M, K, ops) {
		return
	}
	ws.SetActQuantGroup(actGroup)
	linalg.MatmulBTW8A8Batch(ws, a, M, K, ops)
}

// QuantBatchBackend4 is QuantBatchBackend's int4 (W4A8) counterpart, with the same decline contract: false falls back
// to the CPU batch kernel.
type QuantBatchBackend4 interface {
	MatmulW4A8Batch(a []float32, M, K, group int, ops []linalg.W4A8Op) bool
}

// matmulW4A8Batch routes a shared-activation W4A8 batch through a QuantBatchBackend4 (one GPU
// submit) when available, else the CPU batch kernel — the int4 twin of matmulW8A8Batch above.
func matmulW4A8Batch(be Backend, ws *linalg.Workspace, a []float32, M, K, group int, ops []linalg.W4A8Op, actGroup int) {
	if qb, ok := be.(QuantBatchBackend4); ok && actGroup == 0 && qb.MatmulW4A8Batch(a, M, K, group, ops) {
		return
	}
	ws.SetActQuantGroup(actGroup)
	linalg.MatmulBTW4A8Batch(ws, a, M, K, group, ops)
}

var (
	backendMu       sync.RWMutex
	backendRegistry = map[string]func() (Backend, error){}
)

// RegisterBackend registers a named Backend factory; the goinfer/gpu, cuda and metal modules call it from init() so
// the decoder does not import them. Safe for concurrent use; a later registration of a name replaces the earlier.
func RegisterBackend(name string, factory func() (Backend, error)) {
	backendMu.Lock()
	defer backendMu.Unlock()
	backendRegistry[name] = factory
}

// autoOrder is the order "auto" tries GPU backends in. webgpu is left out on purpose: it is cgo, and slower than cuda or
// metal where those exist, so it stays an explicit choice.
var autoOrder = []string{"cuda", "metal"}

// autoFound is the reason auto gives for picking each backend in autoOrder. Metal's probe reads the RAM size rather
// than opening a device, so what auto actually checks for it is the architecture.
var autoFound = map[string]string{
	"cuda":  "auto: a CUDA device answered",
	"metal": "auto: Apple silicon; metal runs int4 models, other precisions stay on the CPU", // autoMetalPrecision
}

// AutoChoice is what a backend of "auto" resolved to on this binary and machine.
type AutoChoice struct {
	Backend string // "cuda", "metal" or "cpu"
	Reason  string // why, worded for the one line the CLIs print
	// Skipped is set when this binary links a GPU backend that auto passed over: no device answered, the backend is
	// untested on this machine, or it is webgpu. goinfer-serve's -require-backend refuses to start then, rather than
	// run on the CPU.
	Skipped bool
}

// AutoBackend resolves a backend of "auto": the first GPU backend this binary links whose device probe answers, cuda
// then metal, else "cpu". A backend it picks that then cannot build a model resident still declines to the CPU path
// with its own message, as an explicit -backend cuda or metal does.
func AutoBackend() AutoChoice { return autoBackend(CompiledBackends(), FreeBytesFor, runtime.GOARCH) }

// autoBackend is AutoBackend over its inputs: the linked backends, the memory probe and the architecture.
func autoBackend(linked []string, probe func(string) (int64, bool), goarch string) AutoChoice {
	has := map[string]bool{}
	for _, b := range linked {
		has[b] = true
	}
	var passed []string
	for _, b := range autoOrder {
		if !has[b] {
			continue
		}
		if b == "metal" && goarch != "arm64" {
			// Intel Macs were scoped out (docs/completed/task-metal-cgofree-spike.md) and no record has run Metal on one, though the
			// darwin/amd64 release asset links it.
			passed = append(passed, "metal is built in, but has only been run on Apple silicon; -backend metal asks for it")
			continue
		}
		if _, ok := probe(b); ok {
			return AutoChoice{Backend: b, Reason: autoFound[b]}
		}
		passed = append(passed, b+" is built in, but no "+strings.ToUpper(b)+" device answered")
	}
	switch {
	case len(passed) > 0:
		return AutoChoice{Backend: "cpu", Reason: "auto: " + strings.Join(passed, "; "), Skipped: true}
	case has["webgpu"]:
		return AutoChoice{Backend: "cpu", Reason: "auto: this binary's GPU backend is webgpu, which auto does not pick; -backend webgpu asks for it", Skipped: true}
	}
	return AutoChoice{Backend: "cpu", Reason: "auto: this binary has no GPU backend"}
}

// withAutoBackend returns o with a Backend of "auto" resolved, so every later check by backend name (int4 layout, fit
// guard, banner) sees the backend that will run. Each exported entry point that takes Options calls it first.
func (o Options) withAutoBackend() Options {
	if o.Backend == "auto" {
		o.Backend, o.BackendAuto = AutoBackend().Backend, true
	}
	return o
}

// CompiledBackends lists the backends this binary can run: "cpu" plus every name a linked module registered from init().
// It is the compiled-in truth, not the accepted --backend values: --backend accepts "metal" on a CPU-only build and
// falls back.
func CompiledBackends() []string {
	backendMu.RLock()
	names := make([]string, 0, len(backendRegistry)+1)
	for name := range backendRegistry {
		names = append(names, name)
	}
	backendMu.RUnlock()
	sort.Strings(names)
	return append([]string{"cpu"}, names...)
}

var (
	memProbeMu sync.RWMutex
	memProbes  = map[string]func() (freeBytes int64, ok bool){}
)

// RegisterMemoryProbe registers a live free-memory query for a named backend, for Model.Plan and `goinfer-chat fit`, so
// decoder need not import the GPU packages that know how to ask (the RegisterBackend shape). ok=false means unknown (no
// device, no driver, a query error), which Plan treats as "cannot judge, proceed": a probe must never fabricate a number
// to avoid returning ok=false. The metal probe is a fixed fraction of hw.memsize, not a live query, matching the metal
// resident guard's own budget (darwin's UBC makes "available" memory unreliable).
func RegisterMemoryProbe(name string, probe func() (freeBytes int64, ok bool)) {
	memProbeMu.Lock()
	defer memProbeMu.Unlock()
	memProbes[name] = probe
}

// FreeBytesFor calls the named backend's registered memory probe. ok=false when no probe is registered (an unlinked GPU
// module, or "cpu", which uses HostRAMAvailableBytes directly) or when the probe reports unknown.
func FreeBytesFor(name string) (freeBytes int64, ok bool) {
	memProbeMu.RLock()
	p := memProbes[name]
	memProbeMu.RUnlock()
	if p == nil {
		return 0, false
	}
	return p()
}

// NewBackend returns the named backend. "auto" resolves through AutoBackend; "" and "cpu" are the pure-Go CPU backend;
// other names resolve through the registry. A GPU name with no linked module ("webgpu", "cuda", "metal") returns the CPU
// backend together with an explanatory error, so the caller can keep running and report the fallback.
func NewBackend(name string) (Backend, error) {
	if name == "auto" {
		name = AutoBackend().Backend
	}
	switch name {
	case "", "cpu":
		return &cpuBackend{}, nil
	}
	backendMu.RLock()
	factory := backendRegistry[name]
	backendMu.RUnlock()
	if factory != nil {
		return factory()
	}
	// The root cmd/serve cannot enable a GPU backend with a build tag: it lives in a submodule entrypoint, so name that one.
	if name == "webgpu" {
		return &cpuBackend{}, fmt.Errorf("decoder: webgpu backend not built in; build the submodule entrypoint (go build -tags gpu github.com/townsendmerino/goinfer/gpu/cmd/serve) — not `-tags gpu` on the root cmd/serve; using cpu")
	}
	if name == "cuda" {
		return &cpuBackend{}, fmt.Errorf("decoder: cuda backend not built in; build the submodule entrypoint (CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve) — not `-tags cuda` on the root cmd/serve; using cpu")
	}
	if name == "metal" {
		// Options.Validate accepts "metal", so an untagged build reaching here falls back to CPU rather than returning a nil backend.
		return &cpuBackend{}, fmt.Errorf("decoder: metal backend not built in; build the submodule entrypoint (go build github.com/townsendmerino/goinfer/metal/cmd/serve, darwin) — not `-tags metal` on the root cmd/serve; using cpu")
	}
	return nil, fmt.Errorf("decoder: unknown backend %q (have: cpu, webgpu, cuda, metal)", name)
}

// cpuBackend dispatches the matmul to linalg.MatmulBT (SIMD dot kernels parallelized across output columns).
type cpuBackend struct{}

func (*cpuBackend) Name() string { return "cpu" }

func (*cpuBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, b, dst, M, K, N)
}

func (*cpuBackend) Close() error { return nil }

// backendNames resolves what the caller requested against what will execute, and why they differ. NewBackend answers a
// not-built-in request with the CPU backend AND an error, a deliberate fallback; callers must report eff, not req, in the
// status line (history: docs/code-notes/decoder.md#backendNames). req is normalised ("" means cpu); reason is "" when
// nothing was declined.
func backendNames(requested string, beErr error) (req, eff, reason string) {
	req = requested
	if req == "" {
		req = "cpu"
	}
	if beErr == nil {
		return req, req, ""
	}
	// The fallback is always to CPU (see the three not-built-in branches in NewBackend).
	return req, "cpu", beErr.Error()
}

// BackendSummary renders the load banner's backend field so it can never name a backend that is
// not executing. When the request was honoured it is just the name; when it was not, it is the
// transition and the reason, on one line:
//
//	metal
//	requested metal → running on cpu: decoder: metal backend not built in; build the …
func BackendSummary(req, eff, reason string) string {
	if reason == "" || req == eff {
		return eff
	}
	return fmt.Sprintf("requested %s → running on %s: %s", req, eff, reason)
}

// withBackendNames records the requested/effective backend split on a freshly built Model. Every Model construction
// calls it, so a banner cannot start naming a backend that is not running when a load path is added.
func (m *Model) withBackendNames(requested string, beErr error) *Model {
	m.reqBackend, m.effBackend, m.beDecline = backendNames(requested, beErr)
	return m
}
