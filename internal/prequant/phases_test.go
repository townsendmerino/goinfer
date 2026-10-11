package prequant

import (
	"context"
	"io"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// slowOnce wraps a writer and sleeps for d before its first write.
type slowOnce struct {
	w    io.Writer
	d    time.Duration
	done bool
}

func (s *slowOnce) Write(p []byte) (int, error) {
	if !s.done {
		s.done = true
		time.Sleep(s.d)
	}
	return s.w.Write(p)
}

// TestTranscodeTimed_reportsBothPhases: a directory transcode that succeeds reports a stream time and a self-check
// time; one that fails in the stream reports no self-check. The GGUF branch times the same two spans and needs a
// tokenizer-bearing GGUF, which no tiny fixture is.
//
// A tiny fixture's self-check finishes inside one tick of a coarse clock (Windows), where it reads exactly zero. So
// each span is given a pause of its own, longer than any such tick, and must report at least most of it: the stream's
// pause in the stream, the self-check's in the self-check.
func TestTranscodeTimed_reportsBothPhases(t *testing.T) {
	if got := (Phases{Stream: 20100 * time.Millisecond, SelfCheck: 900 * time.Millisecond}).String(); got != "stream 20.1s, self-check 0.9s" {
		t.Errorf("Phases.String() = %q", got)
	}
	const pause, floor = 80 * time.Millisecond, 40 * time.Millisecond
	wrapSinkForTest = func(w io.Writer) io.Writer { return &slowOnce{w: w, d: pause} }
	transcodeSelfCheck = func(path string) error { time.Sleep(pause); return selfCheck(path) }
	t.Cleanup(func() { wrapSinkForTest, transcodeSelfCheck = nil, selfCheck })

	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	ph, err := TranscodeTimed(context.Background(), src, filepath.Join(t.TempDir(), "p.giw"), "int4", false, decoder.GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	if ph.Stream < floor || ph.SelfCheck < floor {
		t.Errorf("phases stream %s, self-check %s: each span held a %s pause and must report at least %s", ph.Stream, ph.SelfCheck, pause, floor)
	}
	ph, err = TranscodeTimed(context.Background(), src, filepath.Join(t.TempDir(), "bad.giw"), "not-a-real-quant", false, decoder.GIWTargetNone)
	if err == nil {
		t.Fatal("an unknown quant transcoded")
	}
	if ph.SelfCheck != 0 {
		t.Errorf("a transcode that failed in the stream reports a self-check of %s", ph.SelfCheck)
	}
}
