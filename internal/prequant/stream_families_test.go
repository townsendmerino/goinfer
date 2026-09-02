package prequant

import (
	"bytes"
	"context"
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
		// The regression: a family routed through the resident-build fallback.
		"gpt-oss": {filepath.Join("..", "..", "decoder", "testdata", "gptoss_tiny.gguf"), false},
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
				"int8", false, false, filepath.Base(tc.path))
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
