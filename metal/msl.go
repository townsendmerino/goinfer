//go:build darwin

package metal

import "github.com/townsendmerino/goinfer/metal/kernels"

// The MSL sources live in metal/kernels, a package that imports nothing from decoder, so the kernel tests beside them
// are cached apart from this package's and a decoder edit does not re-run them (docs/measurements/test-wait-2026-10.md §2).
const (
	allKernels         = kernels.AllKernels
	batchKernelsHeader = kernels.BatchKernelsHeader
	batchKernels       = kernels.BatchKernels
	deltaNetKernels    = kernels.DeltaNetKernels
	eg2MSL             = kernels.Eg2MSL
	g4aMSL             = kernels.G4aMSL
	gemma4MoeKernels   = kernels.Gemma4MoeKernels
	g4vMSL             = kernels.G4vMSL
	gvMSL              = kernels.GvMSL
	gumbelMSLKernels   = kernels.GumbelMSLKernels
	moeKernels         = kernels.MoeKernels
	moeBatchKernels    = kernels.MoeBatchKernels
	pagedFenceSrc      = kernels.PagedFenceSrc
	prefillKernels     = kernels.PrefillKernels
)
