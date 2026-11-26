//go:build darwin

package kernels

import "github.com/townsendmerino/goinfer/internal/runtime/f16"

// The names package metal's kernel tests used for this package's MSL sources and packers (metal/msl.go, metal/pack.go),
// kept for the tests that moved here so their bodies are unchanged.
const (
	batchKernelsHeader = BatchKernelsHeader
	batchKernels       = BatchKernels
	deltaNetKernels    = DeltaNetKernels
	eg2MSL             = Eg2MSL
	g4aMSL             = G4aMSL
	gemma4MoeKernels   = Gemma4MoeKernels
	g4vMSL             = G4vMSL
	gvMSL              = GvMSL
	towerGemmMSL       = TowerGemmMSL
	gumbelMSLKernels   = GumbelMSLKernels
	moeKernels         = MoeKernels
	moeBatchKernels    = MoeBatchKernels
	pagedFenceSrc      = PagedFenceSrc
)

var (
	allKernels     = AllKernels
	prefillKernels = PrefillKernels
)

func packW4A8Row(row []float32) (words []uint32, scales []float32) { return PackW4A8Row(row) }
func f32ToF16(f float32) uint16                                    { return f16.Bits(f) }
