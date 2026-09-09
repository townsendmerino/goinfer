//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// softcapSerial is the reference: the exact loop that shipped at cuda/resident.go and
// cuda/prefill.go before applySoftcap, and that still ships at decoder/forwardn.go,
// decoder/model.go and metal/model.go. Kept verbatim so the gate compares against the thing the
// other four siblings still do, not against a re-derivation of it. This is the WebGPU twin of
// cuda/softcap_test.go — gpu/softcap.go is a byte-identical port of cuda/softcap.go's
// applySoftcap (G6, docs/task-gpu-paths-2026-09.md), so this test is a direct port too.
func softcapSerial(dst []float32, sc float32) {
	for j, v := range dst {
		dst[j] = sc * float32(math.Tanh(float64(v/sc)))
	}
}

// TestApplySoftcap_bitIdentical asserts EXACT equality against the serial reference, not a
// tolerance — see cuda/softcap_test.go's own comment for why bit-identity is structural here
// (each output element is a pure function of the input element at the same index).
func TestApplySoftcap_bitIdentical(t *testing.T) {
	sizes := []int{0, 1, 7, 1024, softcapParallelMin - 1, softcapParallelMin, softcapParallelMin + 1,
		65536, 262144, 262145}
	for _, gomax := range []int{1, 3, 16} {
		prev := runtime.GOMAXPROCS(gomax)
		for _, n := range sizes {
			src := make([]float32, n)
			r := rand.New(rand.NewSource(int64(n) + 1))
			for i := range src {
				src[i] = float32(r.NormFloat64() * 40)
			}
			want := append([]float32(nil), src...)
			softcapSerial(want, 30)
			got := append([]float32(nil), src...)
			applySoftcap(got, 30)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("GOMAXPROCS=%d n=%d: element %d = %v, serial gives %v — applySoftcap "+
						"must be bit-identical to the loop it replaced, and to the siblings that "+
						"still run that loop", gomax, n, i, got[i], want[i])
				}
			}
		}
		runtime.GOMAXPROCS(prev)
	}
}

// TestApplySoftcap_disabled pins the no-op contract. finalSoftcap is 0 for every non-Gemma
// family, and gpu/residency.go's Forward/ForwardN hand that 0 straight to applySoftcap.
func TestApplySoftcap_disabled(t *testing.T) {
	for _, sc := range []float32{0, -1} {
		v := []float32{1, -2, 300, -400, 0}
		want := append([]float32(nil), v...)
		applySoftcap(v, sc)
		for i := range v {
			if v[i] != want[i] {
				t.Errorf("sc=%v: element %d changed from %v to %v — a non-positive softcap must be "+
					"a no-op", sc, i, want[i], v[i])
			}
		}
	}
	applySoftcap(nil, 30) // must not panic
}

// TestApplySoftcap_mutation is the falsifiability check — see cuda/softcap_test.go's own comment
// for why these two specific mutations (a float64 conversion hoisted before the divide; a dropped
// tail element) are the ones worth pinning.
func TestApplySoftcap_mutation(t *testing.T) {
	n := 262145
	src := make([]float32, n)
	r := rand.New(rand.NewSource(9))
	for i := range src {
		src[i] = float32(r.NormFloat64() * 40)
	}
	ref := append([]float32(nil), src...)
	softcapSerial(ref, 30)

	f64div := append([]float32(nil), src...)
	for j, v := range f64div {
		f64div[j] = 30 * float32(math.Tanh(float64(v)/float64(30))) // widened BEFORE the divide
	}
	diff32 := 0
	for i := range ref {
		if f64div[i] != ref[i] {
			diff32++
		}
	}
	if diff32 == 0 {
		t.Error("hoisting the float64 conversion ahead of the division is indistinguishable from " +
			"the shipped form on this data, so the gate cannot catch that edit")
	}

	tail := append([]float32(nil), src...)
	applySoftcap(tail[:n-1], 30)
	if tail[n-1] == ref[n-1] {
		t.Error("dropping the tail element is indistinguishable from processing it, so the gate " +
			"cannot catch an off-by-one in the chunking")
	}
	t.Logf("mutation check: dividing in float64 differs in %d/%d elements; a dropped tail is "+
		"detectable", diff32, n)
}
