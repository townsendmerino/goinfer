//go:build darwin

package kernels

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/internal/f16"
)

// The names package metal's kernel tests used for this package's MSL sources and packers (metal/msl.go, metal/pack.go),
// kept for the tests that moved here so their bodies are unchanged.
const (
	allKernels         = AllKernels
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
	prefillKernels     = PrefillKernels
)

var withAttnSoftcap = WithAttnSoftcap

func packW4A8Row(row []float32) (words []uint32, scales []float32) { return PackW4A8Row(row) }
func packW4A8RowInto(row []float32, words []uint32, scales []float32) {
	PackW4A8RowInto(row, words, scales)
}
func f32ToF16(f float32) uint16 { return f16.Bits(f) }

// Copies of package metal's pure test helpers (metal/testshared_test.go) the moved tests call.

func argmaxF(v []float32) int {
	bi, bv := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func cosF(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

type parityStats struct {
	steps, exact, hard int
	nan                int // positions whose cosine came back NaN/Inf
	worstTie, minCos   float64
}

func (st *parityStats) observeCos(c float64) {
	if math.IsNaN(c) || math.IsInf(c, 0) {
		st.nan++
	} else if c < st.minCos {
		st.minCos = c
	}
}

// assertParity fails on any NaN/Inf cosine first: `NaN < x` is false, so the floor below cannot catch one.
func assertParity(t *testing.T, what string, st parityStats, minCosBar float64) {
	t.Helper()
	if st.nan > 0 {
		t.Errorf("%s: %d/%d positions had a NaN/Inf logit cosine — degenerate GPU output, the worst "+
			"kind of failure, which the min-cosine floor cannot catch (NaN < x is false)", what, st.nan, st.steps)
	}
	if st.minCos < minCosBar {
		t.Errorf("%s: min logit cosine %.6f < %.2f — gross breakage, not int4 noise", what, st.minCos, minCosBar)
	}
	if st.exact*2 < st.steps {
		t.Errorf("%s: argmax parity %d/%d < 50%% — the control manages ~62%% on this harness", what, st.exact, st.steps)
	}
}
