package decoder

import (
	"fmt"
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
// GPU acceleration WITHOUT pulling github.com/cogentcore/webgpu (cgo) into
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

// QuantBackend is an optional Backend extension: a backend that can run the
// int8×int8 (W8A8) weight matmul on-device. weightMat type-asserts for it and
// routes the W8A8 path through it — keeping the weight resident, keyed by the q8
// slice's backing pointer — falling back to the CPU kernel when the backend
// doesn't implement it or a call declines (returns false on any GPU error, so
// results stay correct). The fused qkv / gate-up batch dispatches
// (MatmulBTW8A8Batch) are a CPU optimization and are NOT routed here yet; full
// decode coverage needs a batch equivalent (a follow-on).
type QuantBackend interface {
	MatmulW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) bool
}

var (
	backendMu       sync.RWMutex
	backendRegistry = map[string]func() (Backend, error){}
)

// RegisterBackend registers a named Backend factory. The goinfer/gpu module
// calls this from init() (under `-tags gpu`) to make "webgpu" available
// without the decoder importing the cgo WebGPU implementation. Safe for
// concurrent use; a later registration of the same name replaces the earlier.
func RegisterBackend(name string, factory func() (Backend, error)) {
	backendMu.Lock()
	defer backendMu.Unlock()
	backendRegistry[name] = factory
}

// NewBackend returns the named backend. "" and "cpu" always resolve to the
// pure-Go CPU backend. Other names resolve through the registry; "webgpu"
// falls back to CPU with an explanatory error (rather than hard-failing) when
// goinfer/gpu has not been imported, so a `--backend webgpu` flag still runs
// on a build without the GPU module.
func NewBackend(name string) (Backend, error) {
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
	if name == "webgpu" {
		return &cpuBackend{}, fmt.Errorf("decoder: webgpu backend not registered; import github.com/townsendmerino/goinfer/gpu and build `-tags gpu`; using cpu")
	}
	return nil, fmt.Errorf("decoder: unknown backend %q (have: cpu, webgpu)", name)
}

// cpuBackend dispatches the hot matmul to the shared linalg package
// (M7): SIMD dot kernels (AVX2/NEON) parallelized across output columns. The
// math is identical to the previous naive triple-loop — the decoder parity
// tests (which match HF exactly) still pass — just multiple-× faster.
type cpuBackend struct{}

func (*cpuBackend) Name() string { return "cpu" }

func (*cpuBackend) MatmulBT(a, b, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, b, dst, M, K, N)
}

func (*cpuBackend) Close() error { return nil }
