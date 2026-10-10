package prequant

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestTranscodeTimed_reportsBothPhases: a directory transcode that succeeds reports a stream time and a self-check
// time; one that fails in the stream reports no self-check. The GGUF branch times the same two spans and needs a
// tokenizer-bearing GGUF, which no tiny fixture is.
func TestTranscodeTimed_reportsBothPhases(t *testing.T) {
	if got := (Phases{Stream: 20100 * time.Millisecond, SelfCheck: 900 * time.Millisecond}).String(); got != "stream 20.1s, self-check 0.9s" {
		t.Errorf("Phases.String() = %q", got)
	}
	src := filepath.Join("..", "..", "testdata", "qwen3moe-tiny")
	ph, err := TranscodeTimed(context.Background(), src, filepath.Join(t.TempDir(), "p.giw"), "int4", false, decoder.GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	if ph.Stream <= 0 || ph.SelfCheck <= 0 {
		t.Errorf("phases %+v, want both positive", ph)
	}
	ph, err = TranscodeTimed(context.Background(), src, filepath.Join(t.TempDir(), "bad.giw"), "not-a-real-quant", false, decoder.GIWTargetNone)
	if err == nil {
		t.Fatal("an unknown quant transcoded")
	}
	if ph.SelfCheck != 0 {
		t.Errorf("a transcode that failed in the stream reports a self-check of %s", ph.SelfCheck)
	}
}
