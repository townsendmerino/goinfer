//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// softcapSerial is the pre-parallel reference: the exact expression finalizeLogits used.
func softcapSerial(logits []float32, sc float32) {
	for j, v := range logits {
		logits[j] = sc * float32(math.Tanh(float64(v/sc)))
	}
}

// TestMetalSoftcapParallel_bitIdentical is the gate the parallel softcap rests on: fanning the
// per-element sc·tanh(x/sc) across cores must be BYTE-IDENTICAL to the serial loop (every element
// independent, disjoint writes, no reduction/ordering). Probes gemma-scale vocab with mixed-sign,
// mixed-magnitude inputs incl. values past the softcap where tanh saturates.
func TestMetalSoftcapParallel_bitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(30303))
	for _, n := range []int{7, 8192, 262144} { // below threshold (serial), at, and gemma-3 vocab
		for _, sc := range []float32{30.0, 50.0} { // gemma-2/3 = 30, gemma-4 = 50
			a := make([]float32, n)
			for i := range a {
				a[i] = float32(rng.NormFloat64()) * 40 // spans well past ±sc so tanh saturates
			}
			want := append([]float32(nil), a...)
			got := append([]float32(nil), a...)
			softcapSerial(want, sc)
			softcapParallel(got, sc)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("n=%d sc=%g idx=%d: parallel=%v (0x%x) != serial=%v (0x%x)",
						n, sc, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
				}
			}
		}
	}
}

func benchSoftcap(b *testing.B, n int, parallel bool) {
	const sc = float32(30.0)
	base := make([]float32, n)
	rng := rand.New(rand.NewSource(1))
	for i := range base {
		base[i] = float32(rng.NormFloat64()) * 40
	}
	buf := make([]float32, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(buf, base)
		if parallel {
			softcapParallel(buf, sc)
		} else {
			softcapSerial(buf, sc)
		}
	}
}

// gemma-3 vocab 262144, gemma-4 262144; qwen-ish 151936 (non-softcapped, but sizes the loop).
func BenchmarkSoftcap_gemmaVocab_serial(b *testing.B)   { benchSoftcap(b, 262144, false) }
func BenchmarkSoftcap_gemmaVocab_parallel(b *testing.B) { benchSoftcap(b, 262144, true) }
