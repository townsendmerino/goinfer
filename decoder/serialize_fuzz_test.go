package decoder

import (
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"testing"
)

// Track 2.3 (testing campaign): LoadSerializedWeights deserializes goinfer's own
// .giw weights blob. A CRC32 over the whole body gates parsing, so raw-byte
// fuzzing never reaches the body — this target recomputes the CRC for every
// input so the mutator actually exercises the config/weightMat/layer reader.
// Bar: a *SerializeError or a clean *Weights, never a panic/OOM.

// tinyLlamaBlob serializes a minimal valid llama bundle (1 empty layer) using the
// exact config lora_test loads, so the seed parses all the way through the body.
func tinyLlamaBlob() []byte {
	const cfgJSON = `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":1,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	var cfg Config
	if err := json.Unmarshal([]byte(cfgJSON), &cfg); err != nil {
		panic(err)
	}
	w := &Weights{Cfg: cfg, Layers: make([]LayerWeights, 1)}
	blob, err := SerializeWeights(w, "fuzz-seed")
	if err != nil {
		panic(err)
	}
	return blob
}

// withFreshCRC replaces the trailing CRC word so an arbitrary body passes the
// CRC gate and reaches the body parser. A body shorter than the CRC word is
// returned as-is (LoadSerializedWeights handles the too-short case).
func withFreshCRC(body []byte) []byte {
	if len(body) < 4 {
		return body
	}
	out := make([]byte, len(body))
	copy(out, body)
	crc := crc32.ChecksumIEEE(out[:len(out)-4])
	binary.LittleEndian.PutUint32(out[len(out)-4:], crc)
	return out
}

// TestTinyLlamaSeedLoads confirms the fuzz seed is a fully valid blob (so the
// corpus exercises the body reader, not just the header).
func TestTinyLlamaSeedLoads(t *testing.T) {
	if _, err := LoadSerializedWeights(tinyLlamaBlob()); err != nil {
		t.Fatalf("tiny seed should load cleanly: %v", err)
	}
}

// FuzzLoadSerializedWeights mutates a valid blob (CRC refreshed each exec) and
// asserts the loader never panics — a corrupt/hostile body must produce a typed
// *SerializeError, never crash.
func FuzzLoadSerializedWeights(f *testing.F) {
	f.Add(tinyLlamaBlob())
	f.Add([]byte(giwMagic))
	f.Fuzz(func(t *testing.T, body []byte) {
		w, err := LoadSerializedWeights(withFreshCRC(body))
		if err != nil {
			var se *SerializeError
			if !asSerializeError(err, &se) {
				t.Fatalf("want *SerializeError on bad input, got %T: %v", err, err)
			}
			return
		}
		// A successful load must respect its own bounds.
		if len(w.Layers) < 0 || len(w.Layers) > maxSerializedLayers {
			t.Fatalf("loaded implausible layer count %d", len(w.Layers))
		}
	})
}
