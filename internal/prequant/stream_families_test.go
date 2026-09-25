package prequant

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// M-09: StreamTranscodeGGUF wrote a HEADER-ONLY bundle for five GGUF families, and the comment
// above canSerialize claimed they were refused before the load.
//
// canSerialize has returned nil unconditionally since v6, so nothing was refused. The gpt-oss,
// laguna, granitehybrid, nemotron_h/_moe and llama4 branches each build every layer and
// `return w, nil` WITHOUT calling sink.layer — so the writer emitted a header declaring N
// layers followed by zero layers. cmd/prequant and `serve --stream-weights` then load the whole
// model resident (defeating the one-layer-peak-RAM contract this path exists for) and fail
// minutes later with "truncated body: unexpected end of data": a broken supported path whose
// error names the symptom and not the cause.
//
// decoder/testdata/gptoss_tiny.gguf is COMMITTED and nothing drove the stream path on it —
// stream_test.go uses glm-tiny, whose generic loader does stream. That gap is why this shipped,
// so closing it is the test. Through StreamTranscodeGGUF directly, as the neighbouring tests
// do: the tiny fixtures carry no tokenizer, so the full Transcode refuses before the weights.
func TestStreamTranscode_perFamilyBodiesCarryTheirLayers(t *testing.T) {
	for name, tc := range map[string]struct {
		path    string
		streams bool // does its GGUF branch drive the sink itself?
	}{
		// The regression, historically: a family routed through the resident-build
		// fallback. S2 (task-never-swap-2026-09.md, 2026-09-23) moved gpt-oss OFF that
		// fallback — its own loadGptOss closure already builds one layer independently of
		// every other, so it now streams natively too, same as glm below. `streams` is
		// this test's own record of that; TestGptOss_streamedMatchesResident is the byte-
		// identity gate that actually proves it (this test only proves the bundle isn't
		// header-only, not which path produced it).
		"gpt-oss": {filepath.Join("..", "..", "decoder", "testdata", "gptoss_tiny.gguf"), true},
		// The control: a family whose generic loader streams natively. Without it, a "fix"
		// that routed EVERYTHING through the resident build would pass unnoticed — and that
		// would silently discard the one-layer-peak-RAM contract for models that have it.
		"glm (streams natively)": {filepath.Join("..", "..", "testdata", "glm-tiny.gguf"), true},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("no fixture at %s", tc.path)
			}
			var body bytes.Buffer
			n, err := decoder.StreamTranscodeGGUF(context.Background(), tc.path, &body,
				"int8", false, decoder.GIWTargetNone, filepath.Base(tc.path))
			if err != nil {
				t.Fatalf("StreamTranscodeGGUF: %v", err)
			}
			if int(n) != body.Len() {
				t.Errorf("returned %d bytes, wrote %d", n, body.Len())
			}
			// THE ASSERTION THAT MATTERS: the body deserializes AND carries its layers. A
			// header-only bundle fails here with "truncated body: unexpected end of data",
			// which is M-09's exact symptom.
			w, err := decoder.LoadSerializedWeights(body.Bytes())
			if err != nil {
				t.Fatalf("LoadSerializedWeights: %v — a header declaring N layers followed by "+
					"zero layers is what M-09 produced", err)
			}
			if len(w.Layers) == 0 {
				t.Fatal("the bundle round-trips with ZERO layers — header-only (M-09)")
			}
			// Not merely present: carrying weights. An all-empty layer list would satisfy the
			// check above.
			if w.Layers[0].QProj.Rows() == 0 {
				t.Error("layer 0 has no Q projection — the layers are present but empty")
			}
		})
	}
}

// TestGptOss_streamedMatchesResident is S2's own registered gate for the first family moved off
// the resident-serialize fallback (needsResidentSerialize, deleted 2026-09-24 when gemma4 — the last
// family on it — began streaming too): the streamed bundle must be byte-identical to the resident-build-then-
// serialize path's output, same shape as stream_test.go's TestStreamTranscodeMatchesResident (glm)
// — extended here per family rather than widening that one, since a failure on one fixture should
// name which family broke, not force a reader to guess from a shared table's row count.
func TestGptOss_streamedMatchesResident(t *testing.T) {
	gguf := filepath.Join("..", "..", "decoder", "testdata", "gptoss_tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny gpt-oss GGUF at %s", gguf)
	}
	for _, quant := range []string{"int4", "int8int8", ""} {
		t.Run("quant="+quant, func(t *testing.T) {
			resident, streamed, label := transcodeBothWays(t, gguf, quant)
			rPre, rPost := giwSplit(t, resident, giwLabelOffset(t, resident), label)
			sPre, sPost := giwSplit(t, streamed, giwLabelOffset(t, streamed), "")
			if !bytes.Equal(rPre, sPre) {
				t.Errorf("bundles differ BEFORE the quant label (%d vs %d B) — the header itself diverges",
					len(rPre), len(sPre))
			}
			if !bytes.Equal(rPost, sPost) {
				n := 0
				for i := 0; i < len(rPost) && i < len(sPost); i++ {
					if rPost[i] != sPost[i] {
						n++
					}
				}
				t.Fatalf("bundles differ AFTER the quant label: %d of %d bytes (lengths %d vs %d) — the "+
					"streaming and resident paths do NOT produce the same weights",
					n, len(rPost), len(rPost), len(sPost))
			}
		})
	}
}

// TestStreamTranscode_ctxCancel_gptoss is S2's own gate for TestStreamTranscode_ctxCancel_M21:
// M-21's cancellation contract ("an already-cancelled context aborts before writing") must still
// hold for gpt-oss now that it streams too, not just for the families that already did.
func TestStreamTranscode_ctxCancel_gptoss(t *testing.T) {
	gguf := filepath.Join("..", "..", "decoder", "testdata", "gptoss_tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny gpt-oss GGUF at %s", gguf)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled
	var body bytes.Buffer
	n, err := decoder.StreamTranscodeGGUF(ctx, gguf, &body, "int4", false, decoder.GIWTargetNone, "gptoss-tiny")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("StreamTranscodeGGUF with a cancelled ctx = (%d, %v); want a context.Canceled error", n, err)
	}
	if body.Len() != 0 {
		t.Errorf("wrote %d bytes after cancellation; want 0 (the transcode must abort before writing)", body.Len())
	}
}
