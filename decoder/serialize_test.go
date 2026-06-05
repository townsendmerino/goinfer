package decoder

import (
	"context"
	"os"
	"slices"
	"testing"
	"unsafe"
)

// prequantGGUF is the model used for the serialize round-trip test. It skips
// cleanly when absent (like the other GGUF parity tests). Point it at a real
// .gguf via GINFER_PREQUANT_GGUF to run locally.
func prequantGGUF(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GINFER_PREQUANT_GGUF")
	if path == "" {
		path = "../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gguf at %s (set GINFER_PREQUANT_GGUF)", path)
	}
	return path
}

// TestSerializeWeights_roundTrip is the important one: int8int8 GGUF →
// SerializeWeights → LoadSerializedWeights must reconstruct byte-identical
// resident weights AND produce identical greedy generation.
func TestSerializeWeights_roundTrip(t *testing.T) {
	path := prequantGGUF(t)

	m1, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load gguf: %v", err)
	}
	blob, err := SerializeWeights(m1.w, "roundtrip-test")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	t.Logf("serialized blob: %.0f MB", float64(len(blob))/1048576)

	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// Config + shape agree.
	if w2.Cfg.VocabSize != m1.w.Cfg.VocabSize || w2.Cfg.NumLayers != m1.w.Cfg.NumLayers {
		t.Fatalf("config mismatch: got vocab=%d layers=%d", w2.Cfg.VocabSize, w2.Cfg.NumLayers)
	}

	// Every matmul weight is byte-identical.
	a, b := m1.w.matmulWeights(), w2.matmulWeights()
	if len(a) != len(b) {
		t.Fatalf("matmul weight count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].rows != b[i].rows || a[i].cols != b[i].cols || a[i].group != b[i].group || a[i].w8a8 != b[i].w8a8 {
			t.Fatalf("weight %d shape mismatch", i)
		}
		if !slices.Equal(a[i].q8, b[i].q8) {
			t.Fatalf("weight %d q8 bytes differ", i)
		}
		if !slices.Equal(a[i].scales, b[i].scales) {
			t.Fatalf("weight %d scales differ", i)
		}
	}
	t.Logf("byte-identical across %d matmul weights", len(a))

	// Aliasing: the deserialized q8 must point INTO blob (zero-copy); scales must
	// be a copy (independent of blob).
	for _, w := range b {
		if len(w.q8) == 0 {
			continue
		}
		if !aliases(w.q8, blob) {
			t.Errorf("q8 (%d rows) is NOT aliased into the blob — expected zero-copy", w.rows)
		}
		if len(w.scales) > 0 && aliasesF32(w.scales, blob) {
			t.Errorf("scales are aliased into the blob — expected a copy (alignment)")
		}
		break // one is enough
	}

	// Identical greedy generation from a fixed prompt.
	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	tok1 := greedyFirst(t, m1, prompt)
	tok2 := greedyFirst(t, m2, prompt)
	if tok1 != tok2 {
		t.Fatalf("greedy token differs: gguf=%d prequant=%d", tok1, tok2)
	}
	t.Logf("identical greedy token: %d", tok1)
}

// TestLoadSerializedWeights_guards: every corruption returns a *SerializeError,
// none panic.
func TestLoadSerializedWeights_guards(t *testing.T) {
	path := prequantGGUF(t)
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	good, err := SerializeWeights(m.w, "guards")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}

	cases := map[string][]byte{
		"bad magic":   flip(good, 0),
		"flipped crc": flip(good, len(good)-1),
		"truncated":   good[:len(good)/2],
		"empty":       nil,
	}
	for name, blob := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := LoadSerializedWeights(blob) // must not panic
			if err == nil {
				t.Fatalf("%s: expected error, got nil", name)
			}
			var se *SerializeError
			if !asSerializeError(err, &se) {
				t.Fatalf("%s: want *SerializeError, got %T", name, err)
			}
		})
	}
	// Bumped version is rejected too.
	bumped := slices.Clone(good)
	bumped[len(giwMagic)]++ // first byte of the version word
	if _, err := LoadSerializedWeights(bumped); err == nil {
		t.Fatal("bumped version: expected error")
	}
}

func greedyFirst(t *testing.T, m *Model, prompt []int) int {
	t.Helper()
	out, gen := m.Generate(context.Background(), prompt, 1, SamplingParams{Temperature: 0})
	tok, ok := <-out
	for range out {
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !ok {
		t.Fatal("no token generated")
	}
	return tok
}

func flip(b []byte, i int) []byte {
	c := slices.Clone(b)
	c[i] ^= 0xFF
	return c
}

func asSerializeError(err error, target **SerializeError) bool {
	if se, ok := err.(*SerializeError); ok {
		*target = se
		return true
	}
	return false
}

// aliases reports whether s's backing storage lies within blob.
func aliases(s []int8, blob []byte) bool {
	if len(s) == 0 || len(blob) == 0 {
		return false
	}
	sp := uintptr(unsafe.Pointer(&s[0]))
	bp := uintptr(unsafe.Pointer(&blob[0]))
	return sp >= bp && sp < bp+uintptr(len(blob))
}

func aliasesF32(s []float32, blob []byte) bool {
	if len(s) == 0 || len(blob) == 0 {
		return false
	}
	sp := uintptr(unsafe.Pointer(&s[0]))
	bp := uintptr(unsafe.Pointer(&blob[0]))
	return sp >= bp && sp < bp+uintptr(len(blob))
}
