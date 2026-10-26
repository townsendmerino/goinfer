package audio

import (
	"math"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// parallelRows must hand every index to exactly one block, whatever the core count or n.
func TestParallelRows_coversEveryIndexOnce(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 64, 1500, 10001} {
		hits := make([]atomic.Int32, n)
		parallelRows(n, func(lo, hi int) {
			if lo < 0 || hi > n || lo > hi {
				t.Errorf("n=%d: block [%d,%d) is out of range", n, lo, hi)
				return
			}
			for i := lo; i < hi; i++ {
				hits[i].Add(1)
			}
		})
		for i := range hits {
			if hits[i].Load() != 1 {
				t.Fatalf("n=%d: index %d was handed out %d times", n, i, hits[i].Load())
			}
		}
	}
}

// geluInto is the exact-GELU contract kernel over the whole buffer (the chunking and the fan-out change nothing), and the planted tanh defect keeps its scalar form.
func TestWhisperGeluInto_isTheKernelAndKeepsTheDefect(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	src := make([]float32, 3*(1<<14)+123) // several chunks and a ragged tail
	for i := range src {
		src[i] = float32(rng.NormFloat64()) * 3
	}
	want := make([]float32, len(src))
	linalg.GELUContractInto(want, src)
	e := &WhisperEncoder{}
	got := append([]float32(nil), src...)
	e.geluInto(got)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: %v, the kernel's %v", i, got[i], want[i])
		}
	}
	e.defect = whisperEncDefectTanhGELU
	got = append(got[:0], src...)
	e.geluInto(got)
	for i := range got {
		if got[i] != e.gelu(src[i]) {
			t.Fatalf("tanh defect, element %d: %v, the scalar's %v", i, got[i], e.gelu(src[i]))
		}
	}
}
