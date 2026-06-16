package prequant

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGIWRoundTripPreservesRouterBias guards the .giw serialize format against
// silently dropping per-layer fields. The byte-identity test above can't catch a
// field BOTH the streamed and resident paths skip (they share the serializer); only
// a transcode → load → inspect round-trip does. RouterBias (GLM/DeepSeek
// e_score_correction_bias) was added to LayerWeights but initially not to the giw
// format, so a stream-weights GLM lost its routing bias — caught only by the real
// 106B gate. This pins it on the tiny GLM model.
func TestGIWRoundTripPreservesRouterBias(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	// The tiny fixture carries no tokenizer, so go through the weights serializer
	// directly (StreamTranscodeGGUF → LoadSerializedWeights) rather than the full
	// .giw bundle — it's the serialize round-trip we're guarding.
	var body bytes.Buffer
	if _, err := decoder.StreamTranscodeGGUF(gguf, &body, "int4", false, "glm-tiny"); err != nil {
		t.Fatalf("StreamTranscodeGGUF: %v", err)
	}
	w, err := decoder.LoadSerializedWeights(body.Bytes())
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	layers := w.Layers
	if layers[0].Experts != nil {
		t.Errorf("layer 0 (dense prefix) has experts after round-trip")
	}
	for i := 1; i < len(layers); i++ {
		if len(layers[i].RouterBias) == 0 {
			t.Errorf("layer %d RouterBias dropped by the .giw round-trip", i)
		}
	}
}

// TestStreamTranscodeMatchesResident proves the streaming transcode (one layer at a
// time, for a model too large to hold resident) produces a byte-identical .giw
// weights body to the resident path (Load + SerializeWeights). Same input bytes →
// same per-tensor quantization, regardless of load order — so the streaming path is
// a drop-in, not an approximation. Uses the tiny GLM GGUF (generic loader, the
// streaming path); skips if the fixture isn't built.
func TestStreamTranscodeMatchesResident(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	for _, quant := range []string{"int4", "int8int8", ""} {
		t.Run("quant="+quant, func(t *testing.T) {
			// Resident: full load, then serialize.
			m, err := decoder.Load(gguf, decoder.Options{Quant: quant})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			resident, err := decoder.SerializeWeights(m.Weights(), "glm-tiny.gguf")
			m.Close()
			if err != nil {
				t.Fatalf("SerializeWeights: %v", err)
			}
			// Streaming: transcode layer-by-layer into a buffer.
			var streamed bytes.Buffer
			if _, err := decoder.StreamTranscodeGGUF(gguf, &streamed, quant, false, "glm-tiny.gguf"); err != nil {
				t.Fatalf("StreamTranscodeGGUF: %v", err)
			}
			if !bytes.Equal(resident, streamed.Bytes()) {
				t.Fatalf("streamed body (%d B) != resident body (%d B)", streamed.Len(), len(resident))
			}
		})
	}
}
