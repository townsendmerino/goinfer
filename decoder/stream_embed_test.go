package decoder

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestForRowRanges: the ranges cover every row once, each starts on a multiple of align, each gets a reader of its own,
// and the lowest failing range's error is the one returned.
func TestForRowRanges(t *testing.T) {
	defer func() { streamEmbedWorkersForTest = 0 }()
	for _, c := range []struct{ rows, align, workers int }{{37, 1, 4}, {37, 4, 3}, {4096, 1024, 8}, {5, 1, 16}, {1, 1, 4}, {2500, 1024, 4}} {
		streamEmbedWorkersForTest = c.workers
		var mu sync.Mutex
		seen := make([]int, c.rows)
		readers := 0
		err := forRowRanges(c.rows, 8, c.align, func() rowReader {
			mu.Lock()
			readers++
			mu.Unlock()
			return nil
		}, func(lo, hi int, _ rowReader) error {
			if lo%c.align != 0 {
				t.Errorf("%+v: a range starts at row %d, not a multiple of %d", c, lo, c.align)
			}
			mu.Lock()
			defer mu.Unlock()
			for r := lo; r < hi; r++ {
				seen[r]++
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if slices.Min(seen) != 1 || slices.Max(seen) != 1 {
			t.Errorf("%+v: rows covered %d to %d times, want exactly once each", c, slices.Min(seen), slices.Max(seen))
		}
		if readers < 1 || readers > c.workers {
			t.Errorf("%+v: %d readers made, want one per range and at most %d", c, readers, c.workers)
		}
	}
	streamEmbedWorkersForTest = 4
	first, second := errors.New("range at row 10"), errors.New("range at row 30")
	err := forRowRanges(40, 8, 1, func() rowReader { return nil }, func(lo, hi int, _ rowReader) error {
		switch lo {
		case 10:
			return first
		case 30:
			return second
		}
		return nil
	})
	if err != first {
		t.Errorf("got %v, want the lowest failing range's error", err)
	}
}

// embedTableBytes flattens a logit table's storage, whatever its layout, for a byte comparison.
func embedTableBytes(t *testing.T, w linalg.WeightMat) []byte {
	t.Helper()
	var b bytes.Buffer
	b.WriteString(w.Kind() + "/" + w.Int4Layout() + ";")
	put := func(v any) {
		switch x := v.(type) {
		case []float32:
			for _, f := range x {
				b.WriteString(string(appendF32(nil, f)))
			}
		case []int8:
			for _, i := range x {
				b.WriteByte(byte(i))
			}
		case []byte:
			b.Write(x)
		case []uint16:
			for _, u := range x {
				b.WriteByte(byte(u))
				b.WriteByte(byte(u >> 8))
			}
		}
	}
	if f, ok := w.F32(); ok {
		put(f)
	}
	if q8, sc, _, ok := w.Int8(); ok {
		put(q8)
		put(sc)
	}
	if q4, s16, _, ok := w.Int4F16(); ok {
		put(q4)
		put(s16)
	}
	if r4, s4, ok := w.Int4Row4F16(); ok {
		put(r4)
		put(s4)
	}
	if b.Len() < 16 {
		t.Fatalf("a %s table flattened to %d bytes: nothing was compared", w.Kind(), b.Len())
	}
	return b.Bytes()
}

func appendF32(dst []byte, f float32) []byte {
	u := math.Float32bits(f)
	return append(dst, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
}

// TestStreamQuantizedEmbed_rangesGiveTheSequentialTable: for every embedding precision and both layout policies, the
// table built over 2, 3, 5 and 8 row ranges, with a stateless reader and with a block-caching one, is the table one range
// builds.
func TestStreamQuantizedEmbed_rangesGiveTheSequentialTable(t *testing.T) {
	defer func() { streamEmbedWorkersForTest = 0 }()
	const rows, cols = 148, 64
	rng := rand.New(rand.NewSource(7))
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(rng.NormFloat64()) * 0.3
	}
	stateless := func() rowReader {
		return func(r int, dst []float32) error { copy(dst, src[r*cols:(r+1)*cols]); return nil }
	}
	const block = 16
	blockCached := func() rowReader { // loadEmbedTable's reader shape: a cache of block rows, private to one range
		base, buf := -1, []float32(nil)
		return func(r int, dst []float32) error {
			if base < 0 || r < base || r >= base+block {
				base = r - r%block
				buf = slices.Clone(src[base*cols : min(base+block, rows)*cols])
			}
			copy(dst, buf[(r-base)*cols:(r-base+1)*cols])
			return nil
		}
	}
	for _, mode := range []quantMode{quantNone, quantInt8, quantInt8I8, quantInt4} {
		for _, pol := range []struct{ needCanonical, skipRow4 bool }{{true, false}, {true, true}, {false, false}} {
			streamEmbedWorkersForTest = 1
			ref, err := streamQuantizedEmbed(rows, cols, mode, pol.needCanonical, pol.skipRow4, 1, stateless)
			if err != nil {
				t.Fatal(err)
			}
			want := embedTableBytes(t, ref)
			for _, workers := range []int{2, 3, 5, 8} {
				streamEmbedWorkersForTest = workers
				for name, rd := range map[string]struct {
					align int
					mk    func() rowReader
				}{"stateless": {1, stateless}, "block-cached": {block, blockCached}} {
					got, err := streamQuantizedEmbed(rows, cols, mode, pol.needCanonical, pol.skipRow4, rd.align, rd.mk)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(want, embedTableBytes(t, got)) {
						t.Errorf("mode %d %+v, %d ranges, %s reader: the table differs from the one-range table", mode, pol, workers, name)
					}
				}
			}
		}
	}
	streamEmbedWorkersForTest = 3
	boom := errors.New("row 100 unreadable")
	if _, err := streamQuantizedEmbed(rows, cols, quantInt4, true, false, 1, func() rowReader {
		return func(r int, dst []float32) error {
			if r == 100 {
				return boom
			}
			copy(dst, src[r*cols:(r+1)*cols])
			return nil
		}
	}); err != boom {
		t.Errorf("a failed row read returned %v, want the reader's error", err)
	}
}

// TestStreamTranscode_rowRangesWriteTheSameBytes: with the logit tables forced onto several row ranges, a transcode
// writes the bytes the one-range transcode writes: a safetensors directory (the block-cached reader), a GGUF (the shared
// one), with the tables int8-pinned and at int4.
func TestStreamTranscode_rowRangesWriteTheSameBytes(t *testing.T) {
	defer func() { streamEmbedWorkersForTest = 0 }()
	for _, embedInt4 := range []bool{false, true} {
		for _, c := range []struct {
			src    string
			stream func(w io.Writer) error
		}{
			{"../testdata/olmo3-tiny", func(w io.Writer) error {
				_, err := StreamTranscodeDir(context.Background(), "../testdata/olmo3-tiny", w, "int4", embedInt4, GIWTargetCPUArm64, "t")
				return err
			}},
			{"../testdata/gemma2-hd-tiny.gguf", func(w io.Writer) error {
				_, err := StreamTranscodeGGUF(context.Background(), "../testdata/gemma2-hd-tiny.gguf", w, "int4", embedInt4, GIWTargetCPUArm64, "t")
				return err
			}},
		} {
			var ref []byte
			for _, workers := range []int{1, 3, 7} {
				streamEmbedWorkersForTest = workers
				var b bytes.Buffer
				if err := c.stream(&b); err != nil {
					t.Fatalf("%s, %d ranges: %v", c.src, workers, err)
				}
				if workers == 1 {
					ref = b.Bytes()
					continue
				}
				if !bytes.Equal(ref, b.Bytes()) {
					t.Errorf("%s (embed-int4 %v): %d row ranges wrote bytes that differ from one range's", c.src, embedInt4, workers)
				}
			}
		}
	}
}
