//go:build gpu

package gpu

import (
	"testing"
	"time"
)

// BenchmarkResidentPackCost measures the CPU byte-shuffle that dominates an int4 resident load
// (docs/completed/task-mellum2-fast-load.md): for every projection the bridge unpacks the
// decoder's 2-nibble/byte int4 into one-nibble-per-element, packNibbles re-packs it to the GPU
// u32 layout, and packF16Pairs converts the scales; all CPU, paid each launch. It extrapolates
// the per-pass cost to Mellum2's ~12 B params to size the lever of storing the GPU layout in the
// .giw (a straight CreateBufferInit). A Benchmark, not a Test: it logs ms and asserts nothing, so
// as a Test* its green would say only that the harness ran. Go runs a benchmark this slow exactly
// once.
func BenchmarkResidentPackCost(b *testing.B) {
	if testing.Short() {
		b.Skip("pack-cost measurement")
	}
	// A representative slab: 256M params (≈ a couple of Mellum2 expert projections), then
	// extrapolate to ~12 B. K multiple of 32 (W4A8 group).
	const N, K = 8192, 32768 // 256M params
	params := N * K

	// decoder int4 storage: 2 nibbles/byte.
	q4 := make([]uint8, N*((K+1)/2))
	for i := range q4 {
		q4[i] = uint8(i * 37)
	}
	scales := make([]float32, N*(K/w4a8GroupSize))
	for i := range scales {
		scales[i] = 0.01
	}

	// Pass 1: decoder 2-nibble/byte → one nibble per element (the buildStacked/uploadProj unpack).
	t0 := time.Now()
	un := make([]uint8, N*K)
	for r := range N {
		row := q4[r*((K+1)/2):]
		d := un[r*K : r*K+K]
		for k := range K {
			b := row[k>>1]
			if k&1 == 0 {
				d[k] = b & 0x0F
			} else {
				d[k] = b >> 4
			}
		}
	}
	unpackMs := time.Since(t0).Seconds() * 1000

	// Pass 2: packNibbles → GPU u32 layout.
	t0 = time.Now()
	_ = packNibbles(un, N, K)
	packMs := time.Since(t0).Seconds() * 1000

	// Scales: f32 → f16 pairs.
	t0 = time.Now()
	_ = packF16Pairs(scales)
	scaleMs := time.Since(t0).Seconds() * 1000

	perB := (unpackMs + packMs + scaleMs) / float64(params) * 1e9 // ms per 1e9 params
	b.Logf("int4 CPU pack for %dM params: unpack %.0f ms + packNibbles %.0f ms + scales %.0f ms = %.0f ms total",
		params/1e6, unpackMs, packMs, scaleMs, unpackMs+packMs+scaleMs)
	b.Logf("→ %.1f ms / 1e9 params  ⇒  ~%.0f s extrapolated to Mellum2's ~12 B params (the per-launch CPU shuffle a GPU-layout .giw removes)",
		perB, perB*12/1000)
}
