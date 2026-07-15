//go:build cuda

package cuda

import (
	_ "embed"

	"github.com/townsendmerino/goinfer/decoder"
)

//go:generate nvcc -arch=sm_75 -ptx megakernel.cu -o megakernel.ptx

// megakernelPTX is the fused decode-layer kernel, compiled by `go generate` (nvcc,
// -arch=sm_75 = Turing / 2070 SUPER) and JIT'd at runtime. The committed file is a
// placeholder; the box regenerates it once megakernel.cu is filled in.
//
//go:embed megakernel.ptx
var megakernelPTX []byte

func init() {
	decoder.RegisterBackend("cuda", func() (decoder.Backend, error) {
		return &cudaBackend{drv: stubDriver{}}, nil
	})
}

// Compile-time seam checks: cudaBackend must satisfy the decoder's backend +
// residency interfaces (catches signature drift against decoder/residency.go).
var (
	_ decoder.Backend          = (*cudaBackend)(nil)
	_ decoder.ResidencyBackend = (*cudaBackend)(nil)
)

// cudaBackend implements decoder.Backend + decoder.ResidencyBackend.
type cudaBackend struct {
	drv driver
}

func (b *cudaBackend) Name() string { return "cuda" }

// MatmulBT is the staged (non-resident) fallback: dst[M,N] = a[M,K]·b[N,K]ᵀ. The
// skeleton computes it on the CPU (correct, unaccelerated) so `--backend cuda`
// yields correct output via fallback before the kernel is wired; the real backend
// dispatches this to a device GEMV.
func (b *cudaBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	for i := 0; i < M; i++ {
		ar := a[i*K : i*K+K]
		for n := 0; n < N; n++ {
			br := bmat[n*K : n*K+K]
			var acc float32
			for k := 0; k < K; k++ {
				acc += ar[k] * br[k]
			}
			dst[i*N+n] = acc
		}
	}
}

// BuildResident builds a resident CUDA decoder from a loaded Model. The skeleton
// declines (ok=false) so the decoder cleanly uses the staged/CPU path; the real
// implementation uploads the quantized projections + KV caches once, JITs
// megakernelPTX, and returns a *cudaResident (see cudaResident.Forward for the
// per-token launch flow). Wired on the box — spec §8.
func (b *cudaBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	_ = megakernelPTX // consumed by the real BuildResident (JIT the kernel)
	return nil, false, nil
}

func (b *cudaBackend) Close() error { return b.drv.Close() }
