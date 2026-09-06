package decoder

import (
	"testing"
	"time"
)

// TestW4A8Row4GiwKind_loadTimeAndMemoryDelta measures the comparison the .giw
// kind-4 decision actually turns on: not kind-3-.giw vs kind-4-.giw load time
// (neither does an in-RAM repack — repackW4A8Row4IfEligible is wired only into
// the GGUF/safetensors streaming loaders, never into LoadSerializedWeights,
// docs/task-w4a8-neon-bandwidth.md), but a GGUF load that pays the in-RAM
// repack cost (TestW4A8Row4_loadTimeAndMemoryDelta's own +100ms/+223.6MB
// numbers, on the same fixture) vs. a kind-4 .giw load that gets Int4Row4()
// populated for FREE — zero-copy mmap alias, no repack computation, per
// WrapInt4Row4's own contract.
//
// Same rigor as the sibling test: order-alternated best-of-3, real 0.5B GGUF
// fixture, informational (not a pass/fail gate).
func TestW4A8Row4GiwKind_loadTimeAndMemoryDelta(t *testing.T) {
	path := prequantGGUF(t)

	mGGUF, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("warm-up gguf load: %v", err)
	}
	var eligible int
	for _, wm := range mGGUF.w.matmulWeights() {
		if _, _, ok := wm.Int4Row4(); ok {
			eligible++
		}
	}
	if eligible == 0 {
		t.Skip("no row4-eligible int4 weight in this fixture — nothing to compare")
	}

	blob4, err := SerializeWeightsRow4(mGGUF.w, "row4-giwkind-loadcost")
	if err != nil {
		t.Fatalf("SerializeWeightsRow4: %v", err)
	}

	loadGGUFTimed := func() (time.Duration, *Model) {
		t0 := time.Now()
		m, lerr := Load(path, Options{Quant: "int4"})
		if lerr != nil {
			t.Fatalf("gguf load: %v", lerr)
		}
		return time.Since(t0), m
	}
	loadKind4Timed := func() (time.Duration, *Weights) {
		t0 := time.Now()
		w, lerr := LoadSerializedWeights(blob4)
		if lerr != nil {
			t.Fatalf("kind-4 giw load: %v", lerr)
		}
		return time.Since(t0), w
	}

	tGGUF := time.Duration(1<<63 - 1)
	tKind4 := time.Duration(1<<63 - 1)
	var mGGUFBest *Model
	var wKind4Best *Weights
	for rep := range 3 {
		ggufFirst := rep%2 == 0
		var dG time.Duration
		var dK time.Duration
		var mg *Model
		var wk *Weights
		if ggufFirst {
			dG, mg = loadGGUFTimed()
			dK, wk = loadKind4Timed()
		} else {
			dK, wk = loadKind4Timed()
			dG, mg = loadGGUFTimed()
		}
		if dG < tGGUF {
			tGGUF, mGGUFBest = dG, mg
		}
		if dK < tKind4 {
			tKind4, wKind4Best = dK, wk
		}
	}

	var canonicalBytes, row4BytesGGUF, row4BytesKind4 int64
	var repackedGGUF, repackedKind4, int4Count int
	for i, wm := range mGGUFBest.w.matmulWeights() {
		q4, q4s, _, ok := wm.Int4()
		if !ok {
			continue
		}
		int4Count++
		canonicalBytes += int64(len(q4)) + int64(len(q4s))*4
		if p4, s4, ok := wm.Int4Row4(); ok {
			repackedGGUF++
			row4BytesGGUF += int64(len(p4)) + int64(len(s4))*4
		}
		if p4, s4, ok := wKind4Best.matmulWeights()[i].Int4Row4(); ok {
			repackedKind4++
			row4BytesKind4 += int64(len(p4)) + int64(len(s4))*4
		}
	}
	if repackedKind4 != eligible {
		t.Fatalf("kind-4 .giw: %d/%d int4 weights carry Int4Row4(), want %d", repackedKind4, int4Count, eligible)
	}

	t.Logf("load time: GGUF+in-RAM-repack %v | kind-4 .giw (row4 already on disk) %v | delta %v (%.1f%% of GGUF load time)",
		tGGUF, tKind4, tKind4-tGGUF, 100*float64(tKind4-tGGUF)/float64(tGGUF))
	t.Logf("resident memory: GGUF path repacked %d/%d (canonical %.1f MB + row4 %.1f MB = %.1f%% MORE resident, an in-RAM copy);"+
		" kind-4 .giw path repacked %d/%d (row4 bytes are mmap-aliased, not heap-copied — no comparable RSS addition)",
		repackedGGUF, int4Count, float64(canonicalBytes)/1e6, float64(row4BytesGGUF)/1e6, 100*float64(row4BytesGGUF)/float64(canonicalBytes),
		repackedKind4, int4Count)

	blob3, err := SerializeWeights(mGGUFBest.w, "row4-giwkind-loadcost-k3")
	if err != nil {
		t.Fatalf("SerializeWeights (kind 3): %v", err)
	}
	t.Logf("on-disk size: kind-3 .giw %.1f MB | kind-4 .giw %.1f MB | delta +%.1f MB (+%.1f%%)",
		float64(len(blob3))/1e6, float64(len(blob4))/1e6, float64(len(blob4)-len(blob3))/1e6,
		100*float64(len(blob4)-len(blob3))/float64(len(blob3)))
}
