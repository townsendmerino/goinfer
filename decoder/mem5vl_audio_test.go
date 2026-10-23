package decoder

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync/atomic"
	"testing"
)

// v4Case is an n-token prompt with an audio run of length a at row 10 and its features.
func v4Case(m *Model, n, a int) (ids []int, feats []float32) {
	_, _, _, _, _, _, vocab := m.Dims()
	ids = make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % vocab
	}
	r := rand.New(rand.NewSource(7))
	feats = make([]float32, a*m.w.arch.HiddenDim)
	for i := range feats {
		feats[i] = float32(r.NormFloat64()) * 0.5
	}
	return ids, feats
}

// TestMEM5VL_audioChunked is V4-G1 (docs/tasks/task-memory-campaign-2026-10.md, "MEM5-VL (4) gates"): the audio prefill in 7- and
// 16-row passes against one pass; the last logits and four decode steps after are byte-equal.
func TestMEM5VL_audioChunked(t *testing.T) {
	ran := 0
	for _, dir := range []string{"../testdata/voxtral-tiny", "../testdata/qwen3asr-tiny"} {
		for _, q := range []string{"", "int4"} {
			m, err := Load(dir, Options{Quant: q})
			if err != nil {
				t.Fatalf("%s %q: %v", dir, q, err)
			}
			ids, feats := v4Case(m, 60, 20)
			run := func(chunk int) ([][]float32, int64) {
				prev := prefillChunkRows
				prefillChunkRows = chunk
				defer func() { prefillChunkRows = prev }()
				cache := m.NewCache(68)
				p0 := atomic.LoadInt64(&prefillPasses)
				lg, err := m.prefillLogitsAudio(context.Background(), ids, feats, 10, 20, cache)
				if err != nil {
					t.Fatal(err)
				}
				passes := atomic.LoadInt64(&prefillPasses) - p0
				return v1Decode(t, m, cache, lg), passes
			}
			want, p1 := run(1 << 20)
			if p1 != 1 {
				t.Errorf("%s %q: the single pass took %d passes", dir, q, p1)
			}
			line := fmt.Sprintf("[V4-G1] %-28s %-6q", dir[len("../testdata/"):], q)
			for _, chunk := range []int{7, 16} {
				got, passes := run(chunk)
				d := 0
				for s := range want {
					d += v1Differ(want[s], got[s])
				}
				line += fmt.Sprintf(" | C=%d passes=%d logits differ %d", chunk, passes, d)
				if d != 0 || passes < 2 {
					t.Errorf("%s %q C=%d: %d logits differ from one pass, in %d passes", dir, q, chunk, d, passes)
				}
			}
			fmt.Fprintln(os.Stderr, line)
			m.Close()
			ran++
		}
	}
	if ran == 0 {
		t.Fatal("no fixture ran")
	}
}

// TestMEM5VL_audioBytes is V4-G2: a 200-row audio prefill with a 40-row run, in 16-row chunks: the pass's row buffers are exactly
// the single pass's scaled to 16 of 200 rows, and its key-sized buffers are the single pass's.
func TestMEM5VL_audioBytes(t *testing.T) {
	m, err := Load("../testdata/voxtral-tiny", Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const n, chunk = 200, 16
	ids, feats := v4Case(m, n, 40)
	run := func(C int) (rows, keys int64) {
		prev, prevSeen := prefillChunkRows, prefillScratchSeen
		defer func() { prefillChunkRows, prefillScratchSeen = prev, prevSeen }()
		prefillChunkRows = C
		prefillScratchSeen = func(ps *prefillPassScratch) { rows, keys = v1RowBytes(ps) }
		if _, err := m.prefillLogitsAudio(context.Background(), ids, feats, 10, 40, m.NewCache(n+8)); err != nil {
			t.Fatal(err)
		}
		return rows, keys
	}
	whole, wholeKeys := run(1 << 20)
	chunked, chunkedKeys := run(chunk)
	fmt.Fprintf(os.Stderr, "[V4-G2] voxtral-tiny: row buffers %d B one pass (%d rows) -> %d B chunked (%d rows); key-sized %d / %d B\n",
		whole, n, chunked, chunk, wholeKeys, chunkedKeys)
	if chunked*n != whole*chunk || chunkedKeys != wholeKeys {
		t.Errorf("row buffers %d B chunked against %d B in one pass (want 16/200 of it); key-sized %d / %d B", chunked, whole, chunkedKeys, wholeKeys)
	}
}
