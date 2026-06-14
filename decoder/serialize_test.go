package decoder

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/giw"
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
		if a[i].Rows() != b[i].Rows() || a[i].Cols() != b[i].Cols() || tGroup(a[i]) != tGroup(b[i]) || tW8A8(a[i]) != tW8A8(b[i]) {
			t.Fatalf("weight %d shape mismatch", i)
		}
		if !slices.Equal(tQ8(a[i]), tQ8(b[i])) {
			t.Fatalf("weight %d q8 bytes differ", i)
		}
		if !slices.Equal(tScales(a[i]), tScales(b[i])) {
			t.Fatalf("weight %d scales differ", i)
		}
	}
	t.Logf("byte-identical across %d matmul weights", len(a))

	// Aliasing: the deserialized q8 must point INTO blob (zero-copy); scales must
	// be a copy (independent of blob).
	for _, w := range b {
		if len(tQ8(w)) == 0 {
			continue
		}
		if !aliases(tQ8(w), blob) {
			t.Errorf("q8 (%d rows) is NOT aliased into the blob — expected zero-copy", w.Rows())
		}
		if len(tScales(w)) > 0 && aliasesF32(tScales(w), blob) {
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

// TestLoadGIW_mmap gates the residency substrate's first step: loading a model
// from a .giw file maps the weights (instead of os.ReadFile) so the int8 weights
// are pageable views into the mapping, and that path is bit-exact — the greedy
// token from the mmap'd .giw matches the GGUF it was prequantized from. Writes a
// real bundle to a temp file and loads it through Model.Load's .giw branch.
func TestLoadGIW_mmap(t *testing.T) {
	path := prequantGGUF(t)

	m1, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load gguf: %v", err)
	}
	defer m1.Close()
	blob, err := SerializeWeights(m1.w, "mmap-test")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	// Frame a .giw bundle (weights + a tokenizer half, unused by the decoder load)
	// and write it to disk so Load takes the mmap path.
	giwPath := filepath.Join(t.TempDir(), "model.giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}

	m2, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("load .giw: %v", err)
	}
	defer m2.Close()
	if m2.mmap == nil {
		t.Fatal("loaded .giw but Model.mmap is nil — weights were not mapped")
	}

	// The int8 weights must be views INTO the mapping (zero-copy, pageable), not
	// heap copies — the whole point of the substrate.
	mapped := 0
	for _, w := range m2.w.matmulWeights() {
		q8 := tQ8(w)
		if len(q8) == 0 {
			continue
		}
		if !aliases(q8, m2.mmap) {
			t.Fatalf("a %d-row weight's q8 is NOT aliased into the mmap region — expected pageable residency", w.Rows())
		}
		mapped++
	}
	if mapped == 0 {
		t.Fatal("no int8 weights found to check aliasing")
	}
	t.Logf("%d matmul weights alias the mmap region (pageable)", mapped)

	// Bit-exact: the mmap'd .giw produces the same greedy token as its source GGUF.
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	if g1, g2 := greedyFirst(t, m1, prompt), greedyFirst(t, m2, prompt); g1 != g2 {
		t.Fatalf("greedy token differs: gguf=%d mmap-giw=%d", g1, g2)
	}
}

// TestSerializeWeightsTo_matchesBuffer gates the streaming serializer: writing the
// bundle to an io.Writer must produce bytes byte-for-byte identical to the in-memory
// SerializeWeights (same CRC, same length). Streaming is how a 35B is prequantized
// without a full-blob RAM spike, so it must not diverge from the buffered format.
func TestSerializeWeightsTo_matchesBuffer(t *testing.T) {
	path := prequantGGUF(t)
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	want, err := SerializeWeights(m.w, "stream-test")
	if err != nil {
		t.Fatalf("buffered serialize: %v", err)
	}
	var buf bytes.Buffer
	n, err := SerializeWeightsTo(&buf, m.w, "stream-test")
	if err != nil {
		t.Fatalf("streamed serialize: %v", err)
	}
	if int(n) != len(want) {
		t.Fatalf("streamed length %d != buffered %d", n, len(want))
	}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatal("streamed bytes differ from buffered SerializeWeights")
	}
	t.Logf("streamed %d bytes, byte-identical to buffered", n)
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
