package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"testing"
)

// mem12Alloc returns the bytes fn allocates, the collector off.
func mem12Alloc(fn func()) uint64 {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	fn()
	runtime.ReadMemStats(&b)
	return b.TotalAlloc - a.TotalAlloc
}

// TestMEM12Scratch_moeChunks is SCR-G1's reading (docs/tasks/task-memory-campaign-2026-10.md, MEM12-scratch gates): on
// mixtral-tiny with the expert-major path on, the bytes a 64-token prefill allocates after its first 16-row chunk. The gate
// compares the reading with origin/main's; the test asserts the expert-major path ran.
func TestMEM12Scratch_moeChunks(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "mixtral-tiny"))
	setKnob(t, m, knobMoEExpertMajor, "1")
	defer func(r int) { prefillChunkRows = r }(prefillChunkRows)
	prefillChunkRows = 16
	ids := f1IDs(64, m.w.arch.VocabSize)
	var first, last uint64
	defer func() { prefillChunkDone = nil }()
	prefillChunkDone = func(c int) {
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		switch c {
		case 0:
			first = ms.TotalAlloc
		case 3:
			last = ms.TotalAlloc
		}
	}
	runs := atomic.LoadInt64(&moeExpertMajorRuns)
	mem12Alloc(func() {
		if _, err := m.prefillLogits(context.Background(), ids, m.NewCache(len(ids)+8)); err != nil {
			t.Fatal(err)
		}
	})
	if atomic.LoadInt64(&moeExpertMajorRuns) == runs {
		t.Fatal("the expert-major path never ran: the reading would measure the per-row path")
	}
	if last == 0 || first == 0 {
		t.Fatal("the prefill did not run four chunks")
	}
	fmt.Fprintf(os.Stderr, "[SCR-G1] mixtral-tiny, chunks 2-4 of a 64-token prefill: %d bytes\n", last-first)
}

// TestMEM12Scratch_headLogits is SCR-G2's reading: on llama-tiny, the mean bytes over 50 warm iterations of a K = 4 verify
// round and of a 16-token prefill (a single reading moves by up to 1.4 KB from run to run, more than the prefill's effect). The
// gate compares them with origin/main's.
func TestMEM12Scratch_headLogits(t *testing.T) {
	m := f1Load(t, filepath.Join("..", "testdata", "llama-tiny"))
	ctx := context.Background()
	c := m.NewCache(64)
	prompt := f1IDs(16, m.w.arch.VocabSize)
	seq := []int{3, 5, 7, 9}
	const n = 50
	mean := func(fn func()) float64 {
		fn() // warm: the first call grows what later ones reuse
		return float64(mem12Alloc(func() {
			for range n {
				fn()
			}
		})) / n
	}
	round := mean(func() {
		c.TruncateTo(0)
		if _, err := m.prefillLogits(ctx, prompt, c); err != nil {
			t.Fatal(err)
		}
		if _, err := m.forwardN(ctx, seq, c); err != nil {
			t.Fatal(err)
		}
	})
	prefill := mean(func() {
		c.TruncateTo(0)
		if _, err := m.prefillLogits(ctx, prompt, c); err != nil {
			t.Fatal(err)
		}
	})
	fmt.Fprintf(os.Stderr, "[SCR-G2] llama-tiny, mean of %d: verify round (K=4) %.1f bytes (prefill + round %.1f, less the prefill); prefill (16 tokens) %.1f bytes\n",
		n, round-prefill, round, prefill)
}

// TestMEM12Scratch_identity is SCR-G4's first part: the chunked batched prefill (expert-major on, for the MoE) leaves the
// last logits byte-equal to the sequential decode's, and a verify round's rows equal sequential forwards', on mixtral-tiny
// and llama-tiny, run twice on one cache so the second reads the reused buffers.
func TestMEM12Scratch_identity(t *testing.T) {
	defer func(r int) { prefillChunkRows = r }(prefillChunkRows)
	prefillChunkRows = 16
	ctx := context.Background()
	for _, dir := range []string{"mixtral-tiny", "llama-tiny"} {
		m := f1Load(t, filepath.Join("..", "testdata", dir))
		setKnob(t, m, knobMoEExpertMajor, "1")
		V := m.w.arch.VocabSize
		ids := f1IDs(48, V)
		seq := m.NewCache(64)
		var want []float32
		for _, id := range ids {
			l, err := m.forward(id, seq)
			if err != nil {
				t.Fatal(err)
			}
			want = append(want[:0], l...)
		}
		var wantN [][]float32
		for _, id := range []int{11, 13, 17} {
			l, err := m.forward(id, seq)
			if err != nil {
				t.Fatal(err)
			}
			wantN = append(wantN, append([]float32(nil), l...))
		}
		c := m.NewCache(64)
		for pass := range 2 {
			c.TruncateTo(0)
			got, err := m.prefillLogits(ctx, ids, c)
			if err != nil {
				t.Fatal(err)
			}
			if d := mem12Diff(got, want); d >= 0 {
				t.Errorf("%s pass %d: prefill logits differ from the sequential decode's at %d", dir, pass, d)
			}
			rows, err := m.forwardN(ctx, []int{11, 13, 17}, c)
			if err != nil {
				t.Fatal(err)
			}
			for i, r := range rows {
				if d := mem12Diff(r, wantN[i]); d >= 0 {
					t.Errorf("%s pass %d: verify row %d differs from the sequential forward's at %d", dir, pass, i, d)
				}
			}
		}
	}
}

// mem12Diff is the first index where a and b differ bitwise, or -1.
func mem12Diff(a, b []float32) int {
	if len(a) != len(b) {
		return 0
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}
