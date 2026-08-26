package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

func sqrt(f float64) float64 { return math.Sqrt(f) }

// G23 — how much would an f32 attention path actually recover?
//
// At K=8192 prefill, acc64 attention is ~70% of the time (MatmulAVAcc64 51.1% +
// MatmulQKAcc64 18.7%). The acc64 comment calls f64 "~3.7× slower than f32", and
// A3 proposes a gated f32 path. But the honest comparison is NOT end-to-end:
//
//   - acc64 reads K/V DIRECTLY by stride, "skipping a kh gather entirely" and
//     "skipping a vt gather+transpose". f32 must pay both, so the f32 arm here
//     includes those gathers — otherwise the ratio flatters f32 by omitting work
//     it cannot avoid.
//   - the f32 branch in attendBatchedHeads is single-threaded by construction
//     (its per-kv-group gather is shared mutable state), so an end-to-end A/B
//     would race parallel-acc64 against serial-f32 and measure the confound.
//     Both arms here are single-threaded, which is the like-for-like comparison.
//
// Shapes are the ones G20's tiling actually calls at an 8k prompt.
func TestG23AttnKernelRatio(t *testing.T) {
	if os.Getenv("GOINFER_G23") == "" {
		t.Skip("set GOINFER_G23=1 to run the A3 kernel comparison")
	}
	const (
		kt    = 256  // query rows per tile (attnRowTile at K=8192)
		hd    = 128  // head dim
		nKeys = 8192 // full prompt depth
		nKV   = 2
		kvDim = nKV * hd
		reps  = 5
	)
	rng := rand.New(rand.NewSource(1))
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	qh := fill(kt * hd)
	keys := fill(nKeys * kvDim)
	vals := fill(nKeys * kvDim)
	scores := make([]float32, kt*nKeys)
	ch := make([]float32, kt*hd)
	avAcc := make([]float64, hd)
	kh := make([]float32, nKeys*hd) // f32 arm must gather
	vt := make([]float32, hd*nKeys) // f32 arm must gather AND transpose

	timeIt := func(name string, fn func()) time.Duration {
		fn() // warm
		best := time.Duration(1<<62 - 1)
		for i := 0; i < reps; i++ {
			t0 := time.Now()
			fn()
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		fmt.Fprintf(os.Stderr, "  %-34s %8.1f ms\n", name, float64(best.Microseconds())/1000)
		return best
	}

	// EQUALIZE PARALLELISM. MatmulBT fans out via parallelCols; the acc64 kernels
	// are plain serial loops. A first pass of this benchmark compared serial acc64
	// against parallel f32 and reported 17.6x against a documented ~3.7x — the
	// ratio was mostly core count. Force MatmulBT serial so the number is the
	// arithmetic-plus-gather truth, then report the parallel figure separately as
	// what it is: a SECOND, separable effect.
	origThreshold := linalg.ParallelThreshold()
	linalg.SetParallelThreshold(1 << 62)
	defer linalg.SetParallelThreshold(origThreshold)

	fmt.Fprintf(os.Stderr, "G23 attention kernels at kt=%d hd=%d nKeys=%d (best of %d), MatmulBT forced SERIAL\n", kt, hd, nKeys, reps)

	// CORRECTNESS FIRST. A ratio between two kernels that compute different things
	// is worthless, and the first version of this benchmark already produced one
	// misleading number (17.6x, from unequal parallelism). So: run both arms once
	// and compare outputs before timing anything.
	{
		accOut := make([]float32, kt*nKeys)
		f32Out := make([]float32, kt*nKeys)
		linalg.MatmulQKAcc64(qh[:kt*hd], keys, accOut, kt, hd, nKeys, 0, kvDim)
		for sIdx := range nKeys {
			copy(kh[sIdx*hd:sIdx*hd+hd], keys[sIdx*kvDim:sIdx*kvDim+hd])
		}
		linalg.MatmulBT(qh[:kt*hd], kh[:nKeys*hd], f32Out, kt, hd, nKeys)
		var maxAbs, dot, na, nb float64
		for i := range accOut {
			d := float64(accOut[i]) - float64(f32Out[i])
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
			dot += float64(accOut[i]) * float64(f32Out[i])
			na += float64(accOut[i]) * float64(accOut[i])
			nb += float64(f32Out[i]) * float64(f32Out[i])
		}
		cos := dot / (sqrt(na) * sqrt(nb))
		fmt.Fprintf(os.Stderr, "  QK arms agree: cosine=%.9f maxAbs=%.3g\n", cos, maxAbs)
		if cos < 0.999999 {
			t.Fatalf("the two QK arms do not compute the same thing (cosine %.9f) — the ratio below would be meaningless", cos)
		}
	}

	qkAcc := timeIt("QK acc64 (direct strided read)", func() {
		linalg.MatmulQKAcc64(qh[:kt*hd], keys, scores[:kt*nKeys], kt, hd, nKeys, 0, kvDim)
	})
	qkF32 := timeIt("QK f32 + kh gather", func() {
		for s := range nKeys { // the gather acc64 skips
			copy(kh[s*hd:s*hd+hd], keys[s*kvDim:s*kvDim+hd])
		}
		linalg.MatmulBT(qh[:kt*hd], kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
	})
	avAcc64 := timeIt("AV acc64 (direct strided read)", func() {
		linalg.MatmulAVAcc64(scores[:kt*nKeys], vals, ch[:kt*hd], avAcc, kt, nKeys, hd, 0, kvDim)
	})
	avF32 := timeIt("AV f32 + vt gather/transpose", func() {
		for s := range nKeys { // the gather+transpose acc64 skips
			vrow := vals[s*kvDim : s*kvDim+hd]
			for d := range hd {
				vt[d*nKeys+s] = vrow[d]
			}
		}
		linalg.MatmulBT(scores[:kt*nKeys], vt[:hd*nKeys], ch[:kt*hd], kt, nKeys, hd)
	})

	totalAcc := qkAcc + avAcc64
	totalF32 := qkF32 + avF32
	ratio := float64(totalAcc) / float64(totalF32)
	// Attention is ~70% of long-context prefill; Amdahl on that share.
	const share = 0.70
	speedup := 1 / (1 - share + share/ratio)
	fmt.Fprintf(os.Stderr,
		"\n  QK  ratio acc64/f32 = %.2fx (serial both)\n  AV  ratio acc64/f32 = %.2fx (serial both)\n"+
			"  COMBINED            = %.2fx\n"+
			"  => end-to-end prefill speedup at %.0f%% attention share: %.2fx\n",
		float64(qkAcc)/float64(qkF32), float64(avAcc64)/float64(avF32), ratio, share*100, speedup)

	// Now the separable second effect: f32 CAN fan out within a matmul, acc64
	// cannot. In the real prefill path acc64 is already parallel at HEAD
	// granularity (G16/G20's worker pool), so this is not additive with the ratio
	// above — it is reported so the two effects are not silently conflated.
	linalg.SetParallelThreshold(origThreshold)
	qkPar := timeIt("QK f32 + gather, PARALLEL MatmulBT", func() {
		for s := range nKeys {
			copy(kh[s*hd:s*hd+hd], keys[s*kvDim:s*kvDim+hd])
		}
		linalg.MatmulBT(qh[:kt*hd], kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
	})
	fmt.Fprintf(os.Stderr, "  f32 intra-matmul parallelism alone: %.2fx (QK)\n",
		float64(qkF32)/float64(qkPar))
}
