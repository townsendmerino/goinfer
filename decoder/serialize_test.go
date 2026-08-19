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
// .gguf via GOINFER_PREQUANT_GGUF to run locally.
func prequantGGUF(t *testing.T) string {
	t.Helper()
	requireHeavyModel(t) // loads a real GGUF (qwen2.5-coder-0.5b, or GOINFER_PREQUANT_GGUF) — auto-fired on the box
	return assetPath(t, "GOINFER_PREQUANT_GGUF")
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

// TestSerializeGemma4MoE_roundTrip gates the v4 format extension: the gemma4
// parallel dense+MoE stack must survive a .giw round-trip. Before v4 canSerialize
// refused all of Gemma 4; now the gemma4 tail carries layer_scalar, the KV-share
// flags, the PLE branch, and the gemma4moe sub-block (router + per-expert scale +
// the three branch norms + the quantized fused experts). The reload must restore
// gemma4moe on every layer and reproduce the greedy decode byte-identically.
func TestSerializeGemma4MoE_roundTrip(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gemma4-moe checkpoint at %s — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	m1, err := Load(ckpt, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load checkpoint: %v", err)
	}
	defer m1.Close()

	blob, err := SerializeWeights(m1.w, "gemma4moe-rt")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// gemma4moe (with all experts) + layer_scalar must be restored on every layer.
	nMoE := 0
	for i := range w2.Layers {
		mo, orig := w2.Layers[i].gemma4moe, m1.w.Layers[i].gemma4moe
		if mo == nil {
			t.Fatalf("layer %d: gemma4moe dropped on round-trip", i)
		}
		nMoE++
		if len(mo.expertsGateUp) != len(orig.expertsGateUp) || len(mo.expertsDown) != len(orig.expertsDown) {
			t.Fatalf("layer %d: expert count changed (%d/%d vs %d/%d)", i,
				len(mo.expertsGateUp), len(mo.expertsDown), len(orig.expertsGateUp), len(orig.expertsDown))
		}
		if _, isF32 := mo.expertsGateUp[0].F32(); isF32 {
			t.Errorf("layer %d: round-tripped experts are f32, not quantized", i)
		}
		if w2.Layers[i].LayerScalar != m1.w.Layers[i].LayerScalar {
			t.Errorf("layer %d: layer_scalar %v != %v", i, w2.Layers[i].LayerScalar, m1.w.Layers[i].LayerScalar)
		}
	}
	if nMoE == 0 {
		t.Fatal("no gemma4moe layers after round-trip")
	}

	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88}
	a, b := greedyN(t, m1, prompt, 8), greedyN(t, m2, prompt, 8)
	if !slicesEqualInt(a, b) {
		t.Fatalf("gemma4-moe .giw round-trip changed the decode:\n direct:     %v\n round-trip: %v", a, b)
	}
	t.Logf("gemma4-moe round-trips: %d MoE layers, decode byte-identical", nMoE)
}

// TestSerializeGemma4E2B_roundTrip exercises the v4 gemma4 tail's DENSE side that
// the PLE-free tiny MoE golden can't: the real E2B has a Per-Layer-Embedding stack
// (model-level PerLayerTokenEmbed/ModelProj/ProjNorm + the per-layer PLEGate/PLEProj/
// PostPLENorm branch), KV-sharing (KVShared), and per-layer layer_scalar — all of
// which .giw v4 must carry. Reload must reproduce the greedy decode byte-identically.
// Skips without the local E2B GGUF.
func TestSerializeGemma4E2B_roundTrip(t *testing.T) {
	requireHeavyModel(t) // loads ~/models gemma-4-E2B (real, multi-GB)
	path := os.Getenv("HOME") + "/models/gemma-4-E2B_q4_0-it.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B gguf (%v)", err)
	}
	m1, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load E2B: %v", err)
	}
	defer m1.Close()
	if m1.w.PerLayerTokenEmbed.Rows() == 0 {
		t.Skip("E2B build has no PLE stack — nothing to gate here")
	}

	blob, err := SerializeWeights(m1.w, "gemma4-e2b-rt")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// Model-level PLE inputs restored.
	if w2.PerLayerTokenEmbed.Rows() != m1.w.PerLayerTokenEmbed.Rows() ||
		w2.PerLayerModelProj.Rows() != m1.w.PerLayerModelProj.Rows() ||
		len(w2.PerLayerProjNorm) != len(m1.w.PerLayerProjNorm) {
		t.Fatalf("PLE model-level inputs lost on round-trip")
	}
	// Per-layer PLE branch + KV-share + layer_scalar restored.
	var nPLE, nKVShared int
	for i := range w2.Layers {
		if w2.Layers[i].PLEGate.Rows() != m1.w.Layers[i].PLEGate.Rows() {
			t.Fatalf("layer %d: PLEGate lost (%d vs %d)", i, w2.Layers[i].PLEGate.Rows(), m1.w.Layers[i].PLEGate.Rows())
		}
		if w2.Layers[i].PLEGate.Rows() > 0 {
			nPLE++
		}
		if w2.Layers[i].KVShared != m1.w.Layers[i].KVShared || w2.Layers[i].VFromK != m1.w.Layers[i].VFromK {
			t.Fatalf("layer %d: KV-share flags changed", i)
		}
		if w2.Layers[i].KVShared {
			nKVShared++
		}
		if w2.Layers[i].LayerScalar != m1.w.Layers[i].LayerScalar {
			t.Fatalf("layer %d: layer_scalar %v != %v", i, w2.Layers[i].LayerScalar, m1.w.Layers[i].LayerScalar)
		}
	}

	m2, err := NewModel(w2, "cpu")
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	prompt := []int{2, 106, 1596, 476, 573} // arbitrary in-vocab ids
	a, b := greedyN(t, m1, prompt, 6), greedyN(t, m2, prompt, 6)
	if !slicesEqualInt(a, b) {
		t.Fatalf("gemma4-E2B .giw round-trip changed the decode:\n direct:     %v\n round-trip: %v", a, b)
	}
	t.Logf("gemma4-E2B round-trips: %d PLE layers, %d KV-shared, decode byte-identical", nPLE, nKVShared)
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

// TestSerialize_unpopulatedLayersOmitsLabel is the dangerous half of B11, gated directly and
// without a heavy asset. writeHeadGlobals used to decide whether to write the v5 quant-label
// field by asking `wr.sink == nil` — "are we the buffered writer" — as a proxy for "do we have
// the real weight data yet". Those disagree for the true GGUF streaming transcode, which writes
// the header on a freshly make()'d, all-zero Layers slice BEFORE any layer has streamed in: at
// that moment quantLabel()'s own "nothing matched" case returns "native" — a REAL quant mode, not
// an empty string — so calling it unconditionally would bake a FALSE "native" label into every
// genuinely-streamed bundle. hasPopulatedLayers() is the correct guard (data availability, not
// writer identity); this asserts it actually withholds the label when the data is not there.
//
// Walks the header by hand (no heavy checkpoint, no architecture resolution needed on write: only
// json.Marshal(w.Cfg) must succeed, and a zero Config does) rather than round-tripping through
// LoadSerializedWeights, which would additionally require a resolvable arch.
func TestSerialize_unpopulatedLayersOmitsLabel(t *testing.T) {
	w := &Weights{Layers: make([]LayerWeights, 4)}
	if w.hasPopulatedLayers() {
		t.Fatal("hasPopulatedLayers() = true on an all-zero Layers slice")
	}
	if got := w.quantLabel(); got != "native" {
		t.Fatalf("quantLabel() on unpopulated Layers = %q — the guard's rationale assumes this is "+
			"the dangerous default \"native\"; if that changed, re-read why hasPopulatedLayers exists", got)
	}

	blob, err := SerializeWeights(w, "")
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	off := len(giwMagic) + 4 + 4 // magic, version(u32), quant(u32)
	readStr := func() string {
		n := binary.LittleEndian.Uint32(blob[off:])
		off += 4
		s := string(blob[off : off+int(n)])
		off += int(n)
		return s
	}
	_ = readStr() // id
	_ = readStr() // config json
	if label := readStr(); label != "" {
		t.Fatalf("label field = %q on an UNPOPULATED w — must be absent, not a guessed quant that "+
			"was never actually resolved from real weight data", label)
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

// TestCanSerialize_refusesUnrepresentable is the C2 gate: the .giw writer only expresses the
// standard block + qwen3_5_moe's extras, so families whose per-layer state it silently drops
// (MLA / Mamba-2 / Gemma-4 PLE / Llama-4) must be REFUSED — else they produce a CRC-valid bundle
// that nil-derefs at the first forward. Table-driven per registered arch; hardware-free.
func TestCanSerialize_refusesUnrepresentable(t *testing.T) {
	// EMPTY AS OF .giw v6 (2026-08-19): every registered family is representable, so nothing should
	// be refused. The map stays because the mechanism stays — a future family with per-layer state
	// the writer has no field for MUST be refused, and this is where that gets recorded.
	//
	// It is worth remembering what this table used to do wrong. It listed MLA / Mamba-2 / Llama-4
	// and asserted (second branch below) that ANYTHING ELSE must be accepted — so when gpt-oss began
	// silently dropping its attention sinks and Laguna began emitting unloadable bundles, this gate
	// was actively enforcing the claim that they were fine. A blocklist checked against a blocklist
	// proves nothing; the real guard is TestSerializeCensus_noSilentFieldDrop, which asks the struct.
	refused := map[string]string{}
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
	// LMHead zero-value ⇒ tied. The layers are present but empty (Rows()==0 ⇒ the per-projection
	// shape checks skip them); the C-06 layer-count check needs len(Layers) == arch.NumLayers.
	w := &Weights{Cfg: *cfg, arch: arch, Embed: emb, Layers: make([]LayerWeights, arch.NumLayers)}
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
