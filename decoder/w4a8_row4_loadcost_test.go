package decoder

import (
	"testing"
	"time"
)

// TestW4A8Row4_loadTimeAndMemoryDelta measures the two numbers the plumbing
// brief flags as feeding the parked .giw-kind decision (docs/task-w4a8-neon-
// bandwidth.md): the load-time cost of the arm64 row4 repack, and the
// resident-memory delta it adds. Both measured from the SAME load path via
// w4a8Row4RepackEnabled (a load-time-measurement-only toggle), not by
// comparing across differently-built binaries.
//
// Not a pass/fail gate — these are informational numbers, logged for the
// record ("measure them like they matter" per the brief), against the real
// 0.5B GGUF fixture.
func TestW4A8Row4_loadTimeAndMemoryDelta(t *testing.T) {
	path := prequantGGUF(t)

	// Warm the OS page cache once (discarded) so neither timed load pays a
	// cold-file-read cost the other doesn't — the repack's own added time is
	// what this measures, not disk/cache state.
	if _, err := Load(path, Options{Quant: "int4"}); err != nil {
		t.Fatalf("warm-up load: %v", err)
	}

	// Order-alternated best-of-3 (this campaign's own contention/clock-ramp
	// lesson — a single fixed-order pass can be contaminated by whichever
	// measurement runs first): rotate which toggle state goes first, min-of-3
	// per state.
	loadTimed := func(enabled bool) (time.Duration, *Model, error) {
		w4a8Row4RepackEnabled = enabled
		t0 := time.Now()
		m, err := Load(path, Options{Quant: "int4"})
		d := time.Since(t0)
		w4a8Row4RepackEnabled = true
		return d, m, err
	}

	tOff := time.Duration(1<<63 - 1)
	tOn := time.Duration(1<<63 - 1)
	var mOff, mOn *Model
	for rep := 0; rep < 3; rep++ {
		firstOff := rep%2 == 0
		var dOff, dOn time.Duration
		var mo, mn *Model
		var err error
		if firstOff {
			dOff, mo, err = loadTimed(false)
			if err != nil {
				t.Fatalf("load (repack disabled): %v", err)
			}
			dOn, mn, err = loadTimed(true)
			if err != nil {
				t.Fatalf("load (repack enabled): %v", err)
			}
		} else {
			dOn, mn, err = loadTimed(true)
			if err != nil {
				t.Fatalf("load (repack enabled): %v", err)
			}
			dOff, mo, err = loadTimed(false)
			if err != nil {
				t.Fatalf("load (repack disabled): %v", err)
			}
		}
		if dOff < tOff {
			tOff, mOff = dOff, mo
		}
		if dOn < tOn {
			tOn, mOn = dOn, mn
		}
	}

	var canonicalBytes, row4Bytes int64
	var repackedCount, int4Count int
	for _, wm := range mOn.w.matmulWeights() {
		q4, q4s, _, ok := wm.Int4()
		if !ok {
			continue
		}
		int4Count++
		canonicalBytes += int64(len(q4)) + int64(len(q4s))*4
		if p4, s4, ok := wm.Int4Row4(); ok {
			repackedCount++
			row4Bytes += int64(len(p4)) + int64(len(s4))*4
		}
	}
	for _, wm := range mOff.w.matmulWeights() {
		if _, _, ok := wm.Int4Row4(); ok {
			t.Fatalf("repack-disabled load has a repacked weight — toggle is not being honored")
		}
	}

	t.Logf("load time: repack disabled %v | repack enabled %v | delta %v (%.1f%% of disabled-load time)",
		tOff, tOn, tOn-tOff, 100*float64(tOn-tOff)/float64(tOff))
	t.Logf("resident memory: %d/%d int4 weights repacked, canonical int4 bytes %.1f MB, "+
		"+row4 bytes %.1f MB (%.1f%% additional RAM for the resident int4 weight set)",
		repackedCount, int4Count, float64(canonicalBytes)/1e6, float64(row4Bytes)/1e6,
		100*float64(row4Bytes)/float64(canonicalBytes))
}
