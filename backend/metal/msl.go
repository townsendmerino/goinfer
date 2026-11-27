//go:build darwin

package metal

import "github.com/townsendmerino/goinfer/backend/metal/kernels"

// The MSL sources live in metal/kernels, a package that imports nothing from decoder, so the kernel tests beside them
// are cached apart from this package's and a decoder edit does not re-run them (docs/measurements/test-wait-2026-10.md §2).
const (
	batchKernelsHeader = kernels.BatchKernelsHeader
	batchKernels       = kernels.BatchKernels
	deltaNetKernels    = kernels.DeltaNetKernels
	gemma4MoeKernels   = kernels.Gemma4MoeKernels
	gumbelMSLKernels   = kernels.GumbelMSLKernels
	moeKernels         = kernels.MoeKernels
	moeBatchKernels    = kernels.MoeBatchKernels
	pagedFenceSrc      = kernels.PagedFenceSrc
)

var (
	allKernels     = kernels.AllKernels
	prefillKernels = kernels.PrefillKernels
)

// withAttnSoftcap builds a kernel source for a model with an attention-score softcap (kernels.WithAttnSoftcap).
var withAttnSoftcap = kernels.WithAttnSoftcap
