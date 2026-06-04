//go:build gpu

package gpu

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/aikit/encoder"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// init registers the WebGPU backend under the name "webgpu" with BOTH the
// goinfer decoder and the aikit encoder. Their Backend interfaces have the
// same method set, so one *webgpuBackend satisfies both — a `-tags gpu` build
// that blank-imports this package gains GPU matmul on either side without the
// core modules ever importing github.com/cogentcore/webgpu.
func init() {
	decoder.RegisterBackend("webgpu", func() (decoder.Backend, error) {
		return newWebGPUBackend("decoder")
	})
	encoder.RegisterBackend("webgpu", func() (encoder.Backend, error) {
		return newWebGPUBackend("encoder")
	})
}

// webgpuBackend runs MatmulBT on a WebGPU adapter (Vulkan / Metal / D3D12) via
// the Context foundation in this package. Built only under -tags gpu.
//
// Resident weights: a model weight matrix is constant across every token, so
// the first MatmulBT for a given weight uploads it to a GPU storage buffer and
// caches the handle (keyed by the slice's backing pointer); every later call
// only uploads the (small) activation and reads the result back. This removes
// the catastrophic per-token re-upload of the weights (the LM head alone is
// the ~671 MB embedding). On any per-call GPU error it falls back to the CPU
// matmul, so results are always correct.
//
// Still naive beyond that: each matmul is its own synchronous dispatch +
// readback, so decode is latency-bound on per-matmul round-trips. Keeping the
// activations resident on-device across a layer's matmuls (and porting
// norms/rope/softmax to WGSL) is the remaining work for a GPU-fast forward.
type webgpuBackend struct {
	ctx  *Context
	name string

	mu        sync.Mutex // Context is not goroutine-safe
	resident  map[*float32]*ResidentMatrix
	fallbacks int
}

// newWebGPUBackend initializes a WebGPU context for the named consumer
// ("decoder"/"encoder", used only in the backend's reported Name). It returns
// a CPU-equivalent error (not a hard failure) when no adapter is present, so a
// `--backend webgpu` selection still runs on a headless machine.
func newWebGPUBackend(who string) (*webgpuBackend, error) {
	ctx, err := New()
	if err != nil {
		// No adapter (headless / no driver): surface as an error so the
		// caller's registry falls back to CPU with a note.
		return nil, fmt.Errorf("gpu: no WebGPU adapter for %s (%v)", who, err)
	}
	return &webgpuBackend{
		ctx:      ctx,
		name:     "webgpu:" + ctx.Backend(),
		resident: make(map[*float32]*ResidentMatrix),
	}, nil
}

func (b *webgpuBackend) Name() string { return b.name }

// MatmulBT computes dst[M,N] = a · bMatᵀ, with bMat (the weight) uploaded once
// and reused. bMat's identity is its backing pointer — the forward pass passes
// the same weight slice every token.
func (b *webgpuBackend) MatmulBT(a, bMat, dst []float32, M, K, N int) {
	b.mu.Lock()
	out, err := b.matmulLocked(a, bMat, M, K, N)
	b.mu.Unlock()
	if err != nil {
		b.fallbacks++
		linalg.MatmulBT(a, bMat, dst, M, K, N) // correctness over speed
		return
	}
	copy(dst, out)
}

// matmulLocked uploads (or reuses) the resident weight and dispatches. Caller
// holds b.mu.
func (b *webgpuBackend) matmulLocked(a, bMat []float32, M, K, N int) ([]float32, error) {
	if len(bMat) == 0 {
		return nil, fmt.Errorf("gpu: empty weight")
	}
	key := &bMat[0]
	rm := b.resident[key]
	if rm == nil {
		var err error
		if rm, err = b.ctx.UploadMatrix(bMat, N, K); err != nil {
			return nil, err
		}
		b.resident[key] = rm
	}
	return b.ctx.MatmulBTResident(a, rm, M)
}

func (b *webgpuBackend) Close() error {
	b.mu.Lock()
	for _, rm := range b.resident {
		rm.Release()
	}
	b.resident = nil
	b.mu.Unlock()
	b.ctx.Close()
	return nil
}
