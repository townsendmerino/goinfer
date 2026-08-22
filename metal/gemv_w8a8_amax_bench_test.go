//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

const (
	amaxN = 512  // sample rows, not the full vocab
	amaxK = 4096 // a realistic hidden dim
)

// TestGemvW8A8Amax_cpuParity gates gemv_w8a8_amax (metal/kernels.go) — the fused int8 GEMV +
// block-argmax that backs the greedy-decode fast path (model.go's ForwardArgmax, r.pGemvW8Amax,
// launch shape (V*32, 256) — 8 simdgroups/threadgroup, one AmaxPart per group over its 8 rows).
// Sourced from the real allKernels so any future edit here is gated against the shipped kernel.
func TestGemvW8A8Amax_cpuParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w8a8_amax")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	quantRow := func(rng *rand.Rand, row []float32) ([]int8, float32) {
		mx := float32(0)
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > mx {
				mx = a
			}
		}
		sc := mx / 127
		if sc == 0 {
			sc = 1
		}
		q := make([]int8, len(row))
		for i, v := range row {
			r := math.Round(float64(v / sc))
			if r > 127 {
				r = 127
			}
			if r < -127 {
				r = -127
			}
			q[i] = int8(r)
		}
		return q, sc
	}

	rng := rand.New(rand.NewSource(9))
	a := make([]float32, amaxK)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	aq, aSc := quantRow(rng, a)

	bq := make([]int8, amaxN*amaxK)
	bSc := make([]float32, amaxN)
	for n := range amaxN {
		row := make([]float32, amaxK)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		q, sc := quantRow(rng, row)
		copy(bq[n*amaxK:(n+1)*amaxK], q)
		bSc[n] = sc
	}

	// CPU reference: per-tile (8 rows) winner, matching the kernel's own AmaxPart-per-simdgroup
	// tiling (tgs=256 -> 8 simdgroups/threadgroup -> 8 rows/threadgroup).
	const rowsPerTile = 8
	nTiles := (amaxN + rowsPerTile - 1) / rowsPerTile
	refV := make([]float32, nTiles)
	refI := make([]uint32, nTiles)
	logits := make([]float32, amaxN)
	for n := range amaxN {
		var acc int32
		for k := range amaxK {
			acc += int32(aq[k]) * int32(bq[n*amaxK+k])
		}
		logits[n] = float32(acc) * aSc * bSc[n]
	}
	for tile := range nTiles {
		bestV, bestI := float32(math.Inf(-1)), uint32(0)
		for r := range rowsPerTile {
			n := tile*rowsPerTile + r
			if n >= amaxN {
				break
			}
			if logits[n] > bestV || (logits[n] == bestV && uint32(n) < bestI) {
				bestV, bestI = logits[n], uint32(n)
			}
		}
		refV[tile], refI[tile] = bestV, bestI
	}

	q := d.NewCommandQueue()
	// AmaxPart is {float32 v; uint32 i;} = 8 bytes/tile.
	part := d.NewBufferLen(nTiles * 2)
	q.Run1D(pipe, nTiles*256, 256,
		NewBufferInt8(d, aq),
		NewBufferFloats(d, []float32{aSc}),
		NewBufferInt8(d, bq),
		NewBufferFloats(d, bSc),
		part,
		NewBufferU32(d, uint32(amaxK)),
	)
	got := part.Floats()

	maxAbsErr, mismatches := 0.0, 0
	for tile := range nTiles {
		gotV := got[tile*2]
		gotI := math.Float32bits(got[tile*2+1])
		if gotI != refI[tile] {
			mismatches++
			t.Logf("tile %d: index mismatch got=%d want=%d (gotV=%v wantV=%v)", tile, gotI, refI[tile], gotV, refV[tile])
			continue
		}
		if d := math.Abs(float64(gotV - refV[tile])); d > maxAbsErr {
			maxAbsErr = d
		}
	}
	if mismatches > 0 || maxAbsErr > 1e-2 {
		t.Fatalf("gemv_w8a8_amax drifts from CPU reference: %d/%d tile index mismatches, maxAbsErr=%.4g", mismatches, nTiles, maxAbsErr)
	}
	t.Logf("gemv_w8a8_amax %d tiles vs CPU: 0 mismatches, maxAbsErr=%.4g", nTiles, maxAbsErr)
}

// BenchmarkGemvW8A8Amax isolates gemv_w8a8_amax's own throughput at production dispatch shape.
func BenchmarkGemvW8A8Amax(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w8a8_amax")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	rng := rand.New(rand.NewSource(9))
	rnd8 := func(n int) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = int8(rng.Intn(255) - 127)
		}
		return s
	}
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*0.01 + 0.001
		}
		return s
	}
	const rowsPerTile = 8
	nTiles := (amaxN + rowsPerTile - 1) / rowsPerTile
	dAq := NewBufferInt8(d, rnd8(amaxK))
	dAsc := NewBufferFloats(d, []float32{0.01})
	dBq := NewBufferInt8(d, rnd8(amaxN*amaxK))
	dBsc := NewBufferFloats(d, rndf(amaxN))
	dPart := d.NewBufferLen(nTiles * 2)
	uK := NewBufferU32(d, uint32(amaxK))

	q_ := d.NewCommandQueue()
	const reps = 2 // matching gemv_w8a8_coal's conservative bisected point (same per-dispatch GPU work scale)
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, nTiles*256, 256, dAq, dAsc, dBq, dBsc, dPart, uK)
		}
		e.End()
	}

	for range 4 {
		run()
	}
	b.ResetTimer()
	best := time.Hour
	for range b.N {
		t0 := time.Now()
		run()
		if dt := time.Since(t0); dt < best {
			best = dt
		}
	}
	b.ReportMetric(float64(best/time.Duration(reps))/float64(time.Nanosecond), "ns/dispatch(best)")
}
