package decoder

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestConfig_giwRoundTrip_nilRawMessage guards the .giw serialize bug where a nil
// json.RawMessage Config field marshals to the literal `null` (4 bytes) and reappears as
// "present" on reload — which made a GGUF-deepseek bundle re-resolve rope from an empty
// rope_parameters and fail "rope_theta must be >0". The optional RawMessage fields carry
// `,omitempty`, so a nil field stays absent across json.Marshal(Cfg) → Unmarshal (the .giw
// round-trip path, serialize.go SerializeWeightsTo / LoadSerializedWeights).
func TestConfig_giwRoundTrip_nilRawMessage(t *testing.T) {
	// A GGUF-deepseek-shaped Config: rope lives in rope_scaling; rope_parameters is nil.
	c := Config{
		ModelType:      "deepseek_v2",
		RoPEGlobalBase: 10000,
		RopeScaling:    json.RawMessage(`{"type":"yarn","factor":40}`),
		// RopeParameters / QuantizationConfig / EOSTokenID intentionally left nil.
	}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var c2 Config
	if err := json.Unmarshal(b, &c2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for name, raw := range map[string]json.RawMessage{
		"rope_parameters":     c2.RopeParameters,
		"quantization_config": c2.QuantizationConfig,
		"eos_token_id":        c2.EOSTokenID,
	} {
		if len(raw) != 0 {
			t.Errorf("nil %s round-tripped to %q (len %d) — a len>0 present-check would mis-fire", name, raw, len(raw))
		}
	}
	if len(c2.RopeScaling) == 0 {
		t.Error("set rope_scaling was lost in the round-trip")
	}
}

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

// TestSerializeQwen35_roundTrip gates the v2 format extension: the qwen3_5_moe
// DeltaNet-hybrid must survive a .giw round-trip — its per-layer delta (linear
// layers) and qattn (softmax layers) tensors are restored, and the greedy decode
// is unchanged. Before v2 these were dropped, so a .giw-loaded hybrid segfaulted
// on the first forward (nil delta). Uses the tiny hybrid checkpoint.
func TestSerializeQwen35_roundTrip(t *testing.T) {
	const ckpt = "../testdata/qwen35-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no hybrid checkpoint at %s — run scripts/pin_qwen35_forward.py", ckpt)
	}
	m1, err := Load(ckpt, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load checkpoint: %v", err)
	}
	defer m1.Close()

	blob, err := SerializeWeights(m1.w, "qwen35-rt")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// Both hybrid layer kinds must have been restored.
	var nDelta, nQattn int
	for i := range w2.Layers {
		if w2.Layers[i].delta != nil {
			nDelta++
		}
		if w2.Layers[i].qattn != nil {
			nQattn++
		}
	}
	if nDelta == 0 || nQattn == 0 {
		t.Fatalf("hybrid tensors lost on round-trip: delta layers=%d qattn layers=%d", nDelta, nQattn)
	}

	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a, b := greedyN(t, m1, prompt, 8), greedyN(t, m2, prompt, 8)
	if !slicesEqualInt(a, b) {
		t.Fatalf("qwen3_5_moe .giw round-trip changed the decode:\n direct:    %v\n round-trip: %v", a, b)
	}
	t.Logf("qwen3_5_moe round-trips: %d delta + %d qattn layers, decode byte-identical", nDelta, nQattn)
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

// TestCanSerialize_refusesUnrepresentable is the C2 gate: the .giw writer only expresses the
// standard block + qwen3_5_moe's extras, so families whose per-layer state it silently drops
// (MLA / Mamba-2 / Gemma-4 PLE / Llama-4) must be REFUSED — else they produce a CRC-valid bundle
// that nil-derefs at the first forward. Table-driven per registered arch; hardware-free.
func TestCanSerialize_refusesUnrepresentable(t *testing.T) {
	refused := map[string]string{
		"deepseek_v2": "MLA", "deepseek_v3": "MLA", "kimi_k2": "MLA",
		"granitemoehybrid": "Mamba-2", "nemotron_h": "Mamba-2",
		"gemma4":              "PLE",
		"gemma4_text":         "PLE", // gemma4 MoE variant: own forward (PLE/per-layer) + gemma4moe experts, both undropped by .giw
		"gemma4_unified_text": "PLE", // real unified text_config model_type: same gemma4 forward + gemma4moe experts
		"llama4_text":         "unsupported",
	}
	for name := range archFeatureProfile {
		cfg := representativeConfig(name)
		if cfg == nil {
			continue
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Errorf("%s: resolveArchitecture: %v", name, err)
			continue
		}
		got := canSerialize(arch)
		if _, mustRefuse := refused[name]; mustRefuse && got == nil {
			t.Errorf("%s: canSerialize returned nil, but the .giw writer drops its per-layer state (%s) — it must be refused", name, refused[name])
		}
		if _, mustRefuse := refused[name]; !mustRefuse && got != nil {
			t.Errorf("%s: canSerialize refused a representable family: %v", name, got)
		}
	}
}

// TestLoadSerialized_finalizesTiedLMHead gates the tied-head half of C2: a tied checkpoint
// round-trips with an empty LMHead, and the loader must set TiedLMHead from that — otherwise the
// head reads untied+empty and the forward emits all-zero logits (greedy loops on token 0).
func TestLoadSerialized_finalizesTiedLMHead(t *testing.T) {
	cfg := representativeConfig("qwen2")
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	emb := linalg.WrapF32(make([]float32, cfg.VocabSize*cfg.HiddenDim), cfg.VocabSize, cfg.HiddenDim)
	w := &Weights{Cfg: *cfg, arch: arch, Embed: emb} // LMHead zero-value ⇒ tied; no layers needed
	blob, err := SerializeWeights(w, "tied-head-test")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if w2.LMHead.Rows() != 0 {
		t.Fatalf("expected empty LMHead on round-trip, got %d rows", w2.LMHead.Rows())
	}
	if !w2.arch.TiedLMHead {
		t.Fatal("round-tripped tied checkpoint has TiedLMHead=false — the forward would emit all-zero logits")
	}
}

// TestLoadSerialized_neverPanicsOnValidCRC is the M17 gate: CRC is integrity, not authenticity —
// a determined blob can flip any field and recompute the trailing CRC, so LoadSerializedWeights
// must NEVER panic on such input (linalg.WrapInt8/Int4 panic on a length/group mismatch and a
// wrong-length WrapF32 defers the panic to first use). It flips each body byte, re-CRCs, and
// asserts the loader returns cleanly. Model-free (constructed int8 bundle), so it runs in CI.
func TestLoadSerialized_neverPanicsOnValidCRC(t *testing.T) {
	cfg := representativeConfig("qwen2")
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	emb := linalg.QuantizeInt8(make([]float32, cfg.VocabSize*cfg.HiddenDim), cfg.VocabSize, cfg.HiddenDim, false)
	blob, err := SerializeWeights(&Weights{Cfg: *cfg, arch: arch, Embed: emb}, "m17")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	for i := 0; i < len(blob)-4; i++ {
		b := append([]byte(nil), blob...)
		b[i] ^= 0xFF
		binary.LittleEndian.PutUint32(b[len(b)-4:], crc32.ChecksumIEEE(b[:len(b)-4])) // make CRC "valid"
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("byte %d flip panicked (M17 violated): %v", i, r)
				}
			}()
			if w, err := LoadSerializedWeights(b); err == nil {
				_ = w // a plausible-but-different bundle may load; the contract is only "no panic"
			}
		}()
	}
}
