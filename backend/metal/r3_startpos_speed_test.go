//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR3_startPosSpeed answers the question R3's own SHIPPED result
// (docs/measurements/metal-prefill-floor-2026-09-20.md) left explicitly open: "a startPos-on-speed
// sweep remain[s] unmeasured". The floor's whole point is the common chat/agent-loop turn — a
// SHORT new prompt segment appended to an ALREADY-RESIDENT prefix (decoder/model.go's
// residentPrefillSeed continuing from a resident KV, the peer matrix's own headline workload) —
// but every speed number R3 measured was a batched-vs-sequential comparison starting fresh at
// startPos=0. G-08 (audit-metal-2026-09-12.md) already closed the CORRECTNESS question at
// startPos>0 on a tiny synthetic fixture (metal/prefill_startpos_test.go); this closes the SPEED
// question on a real checkpoint at the shape that matters: a realistic already-resident prefix,
// then a new floor-sized turn continued via PrefillLast(startPos>0) vs the sequential Forward loop
// it replaces.
//
// One resident, one process, interleaved best-of-N per cell (batched, sequential, batched,
// sequential, ...) so drift affects both arms equally. Prefix built once via PrefillLast(embs,0)
// (K=512 — itself above the 64 floor, so this is what a real served session's own history would
// have gone through), then each interleaved rep re-continues from that SAME resident KV state
// with a fresh copy of the turn's embeddings — both arms read the identical prefix and write the
// identical suffix positions on every rep, so nothing about the prefix itself drifts between reps.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR3_startPosSpeed$' -v -timeout 20m
func TestR3_startPosSpeed(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}

	logf := func(format string, args ...any) {
		t.Helper()
		t.Logf(format, args...)
		fmt.Fprintf(os.Stderr, "[r3-startpos] "+format+"\n", args...)
	}

	const prefixLen = 512 // above the 64 floor -- itself built via the batched path, realistically
	const nReps = 6       // matches R3's own n=6 protocol
	turnLens := []int{64, 128}

	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: prefixLen + 256})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built for this model")
	}
	H := rf.r.H
	ctx := context.Background()

	rng := rand.New(rand.NewSource(3))
	genRow := func() []float32 {
		row := make([]float32, H)
		for j := range row {
			row[j] = float32(rng.NormFloat64()) * 0.05
		}
		return row
	}

	// Build the resident prefix ONCE via the batched path itself (realistic: a real session's own
	// earlier turns already went through PrefillLast). Every rep below reuses this exact KV state.
	prefixEmbs := make([][]float32, prefixLen)
	for i := range prefixEmbs {
		prefixEmbs[i] = genRow()
	}
	if _, err := rf.PrefillLast(ctx, prefixEmbs, 0); err != nil {
		t.Fatalf("build prefix: %v", err)
	}
	logf("prefix built: %d tokens via PrefillLast(startPos=0)", prefixLen)

	for _, turnLen := range turnLens {
		t.Run(fmt.Sprintf("turn%d", turnLen), func(t *testing.T) {
			turnEmbs := make([][]float32, turnLen)
			for i := range turnEmbs {
				turnEmbs[i] = genRow()
			}

			runBatched := func() time.Duration {
				t0 := time.Now()
				if _, err := rf.PrefillLast(ctx, turnEmbs, prefixLen); err != nil {
					t.Fatalf("PrefillLast(startPos=%d): %v", prefixLen, err)
				}
				return time.Since(t0)
			}
			runSequential := func() time.Duration {
				t0 := time.Now()
				for i, e := range turnEmbs {
					if _, err := rf.Forward(e, prefixLen+i); err != nil {
						t.Fatalf("Forward(pos=%d): %v", prefixLen+i, err)
					}
				}
				return time.Since(t0)
			}

			var batchedTimes, seqTimes []time.Duration
			for rep := 0; rep < nReps; rep++ {
				batchedTimes = append(batchedTimes, runBatched())
				seqTimes = append(seqTimes, runSequential())
			}

			best := func(ds []time.Duration) time.Duration {
				b := ds[0]
				for _, d := range ds[1:] {
					if d < b {
						b = d
					}
				}
				return b
			}
			bestBatched, bestSeq := best(batchedTimes), best(seqTimes)
			ratio := float64(bestSeq) / float64(bestBatched)

			logf("turn=%d startPos=%d: batched best=%v all=%v | sequential best=%v all=%v | ratio(seq/batched)=%.2fx",
				turnLen, prefixLen, bestBatched, batchedTimes, bestSeq, seqTimes, ratio)

			band := "FLOOR STAYS (<1.3x)"
			switch {
			case ratio >= 2.0:
				band = "SHIPS (>=2.0x)"
			case ratio >= 1.3:
				band = "PARKED (1.3-2.0x)"
			}
			logf("turn=%d startPos=%d: against R3's own registered band -> %s", turnLen, prefixLen, band)
		})
	}
}
