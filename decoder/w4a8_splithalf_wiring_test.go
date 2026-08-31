package decoder

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestW4A8SplitHalfWiring_matchesCanonical drives the split-half repack THROUGH
// its production caller (quantizeWM, the GGUF/safetensors quantize path) rather
// than calling linalg.RepackInt4SplitHalf directly, and asserts the decode
// matmul is bit-identical with the repack on and off.
//
// aikit already has a kernel-level equivalence test. What that one CANNOT show,
// and this one can, is that goinfer's own load path reaches the repack at all
// and that decode's matmul() then uses the layout it produced: the repack is
// silent, the dispatch is inside aikit, and a wiring that quietly did nothing
// would leave every kernel test still passing (CLAUDE.md's G27 lesson — test a
// call-frequency/condition contract through its caller).
func TestW4A8SplitHalfWiring_matchesCanonical(t *testing.T) {
	const rows, cols = 128, 512 // rows%4==0 and cols%32==0: eligible for BOTH arches' repacks

	rng := rand.New(rand.NewSource(7))
	f32 := make([]float32, rows*cols)
	for i := range f32 {
		f32[i] = float32(rng.NormFloat64())
	}
	act := make([]float32, cols)
	for i := range act {
		act[i] = float32(rng.NormFloat64())
	}

	// Eligibility is a property of the BUILD (arch + AVX2 + shape), not of the
	// toggle, so probe it on a throwaway WeightMat. Skipping without this probe
	// would be indistinguishable from a repack that silently stopped firing.
	probe := linalg.QuantizeInt4(f32, rows, cols, int4GroupSize)
	if !probe.RepackInt4SplitHalf() {
		t.Skipf("split-half repack not eligible here (needs amd64 + AVX2 and NO AVX-512 VNNI, group=%d, cols%%32==0) — on a VNNI host aikit declines on purpose, because canonical uses the VNNI tier there and this layout is AVX2-only", int4GroupSize)
	}

	run := func(enabled bool) []float32 {
		t.Helper()
		defer func(prev bool) { w4a8SplitHalfRepackEnabled = prev }(w4a8SplitHalfRepackEnabled)
		w4a8SplitHalfRepackEnabled = enabled

		w := quantizeWM(linalg.WrapF32(append([]float32(nil), f32...), rows, cols), quantInt4)
		dst := make([]float32, rows)
		matmul(nil, &w, act, dst, 1) // M=1: the only M the split-half kernel serves
		return dst
	}

	off, on := run(false), run(true)
	for i := range off {
		if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
			t.Fatalf("row %d: split-half %v != canonical %v (must be BIT-identical: the layout is a permutation of the same nibbles, so any difference is a packing bug, not rounding)", i, on[i], off[i])
		}
	}
}
