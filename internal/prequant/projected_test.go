package prequant

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestProjectedSidecarBytes_neverUnderCounts: the pre-transcode disk check must not pass a sidecar that
// will not fit, so the projection has to be at least what the writer actually produces, at every quant.
// (It used to be the source's size, which a real int4 sidecar exceeds by up to 16% and an int8int8 one by
// ~60%.) Measured against the writer on the tiny fixture here; against real sidecars when written:
// 0.5B/1.5B/7B/Llama-1B/Gemma-4-26B int4 and 0.5B int8int8 projected 1.03–1.14× their actual size.
func TestProjectedSidecarBytes_neverUnderCounts(t *testing.T) {
	src := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(src); err != nil {
		t.Skipf("tiny fixture: %v", err)
	}
	for _, q := range []string{"int4", "int8int8", "int8", ""} {
		m, err := decoder.Load(src, decoder.Options{Quant: q}) // canonical bytes, as the writer needs
		if err != nil {
			t.Fatalf("load at %q: %v", q, err)
		}
		blob, err := decoder.SerializeWeightsForTarget(m.Weights(), "tiny", decoder.GIWTargetForBackend("cpu"))
		m.Close()
		if err != nil {
			t.Fatal(err)
		}
		got, ok := projectedSidecarBytes(src, q)
		if !ok {
			t.Fatalf("projection failed for %q", q)
		}
		if got < int64(len(blob)) {
			t.Errorf("quant %q: projected %d bytes, the writer produced %d — the disk check would pass a sidecar that does not fit", q, got, len(blob))
		}
	}
}
