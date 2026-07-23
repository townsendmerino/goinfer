package decoder

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// Track 2.2 (testing campaign): goinfer's INTERPRETATION of a GGUF — metadata →
// Config synthesis — must turn a hostile header into a typed error, never a
// panic. FuzzGGUFConfig fuzzes that path; TestGGUF_hostileDims_typedError is the
// regression for the makeslice panic that surfaced (block_count overflowing int
// → negative NumLayers). The container parse below it (string lengths, map
// pre-sizing) is aikit's and was hardened in v1.2.1, so the fuzzer now reaches
// goinfer's layer instead of dying in the parser.

// --- a minimal, faithful GGUF encoder (matches aikit/embed.parseGGUF) ---

// gguf metadata value type tags (aikit embed gguf value types).
const (
	gtUint32  uint32 = 4
	gtFloat32 uint32 = 6
	gtBool    uint32 = 7
	gtString  uint32 = 8
	gtArray   uint32 = 9
	gtUint64  uint32 = 10
)

// ggufKV is one metadata entry: a key, its value-type tag, and the encoded value
// bytes (without the tag).
type ggufKV struct {
	key string
	typ uint32
	val []byte
}

// ggufTensorDecl is a tensor directory entry (dims + type + offset only — the
// config path never reads tensor data, so the data section can stay empty).
type ggufTensorDecl struct {
	name string
	dims []uint64
	typ  uint32
}

func gU32(v uint32) []byte  { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, v); return b }
func gU64(v uint64) []byte  { b := make([]byte, 8); binary.LittleEndian.PutUint64(b, v); return b }
func gF32(v float32) []byte { return gU32(math.Float32bits(v)) }
func gStr(s string) []byte  { return append(gU64(uint64(len(s))), s...) }

func kvU32(k string, v uint32) ggufKV  { return ggufKV{k, gtUint32, gU32(v)} }
func kvU64(k string, v uint64) ggufKV  { return ggufKV{k, gtUint64, gU64(v)} }
func kvF32(k string, v float32) ggufKV { return ggufKV{k, gtFloat32, gF32(v)} }
func kvStr(k, v string) ggufKV         { return ggufKV{k, gtString, gStr(v)} }

func gBool(v bool) []byte {
	if v {
		return []byte{1}
	}
	return []byte{0}
}

func kvBoolArr(k string, vs []bool) ggufKV {
	b := append(gU32(gtBool), gU64(uint64(len(vs)))...)
	for _, v := range vs {
		b = append(b, gBool(v)...)
	}
	return ggufKV{k, gtArray, b}
}

// buildGGUF encodes a complete GGUF v3 byte image from metadata + tensor decls,
// padding to the 32-byte data alignment with an empty data section.
func buildGGUF(kvs []ggufKV, tensors []ggufTensorDecl) []byte {
	var buf bytes.Buffer
	buf.Write(gU32(0x46554747)) // "GGUF"
	buf.Write(gU32(3))          // version
	buf.Write(gU64(uint64(len(tensors))))
	buf.Write(gU64(uint64(len(kvs))))
	for _, kv := range kvs {
		buf.Write(gStr(kv.key))
		buf.Write(gU32(kv.typ))
		buf.Write(kv.val)
	}
	for _, t := range tensors {
		buf.Write(gStr(t.name))
		buf.Write(gU32(uint32(len(t.dims))))
		for _, d := range t.dims {
			buf.Write(gU64(d))
		}
		buf.Write(gU32(t.typ))
		buf.Write(gU64(0)) // offset
	}
	for buf.Len()%32 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// embedTensor is a token_embd.weight directory entry [in=hidden, out=vocab], the
// dims ggufVocabSize reads (no data needed).
func embedTensor(hidden, vocab uint64) ggufTensorDecl {
	return ggufTensorDecl{"token_embd.weight", []uint64{hidden, vocab}, 0}
}

// TestGGUF_hostileDims_typedError is the regression for the makeslice panic
// FuzzGGUFConfig surfaced: a GGUF whose block_count overflows int (1<<63 →
// negative NumLayers) once panicked make([]LayerWeights, n). The loader must now
// return a typed error, never panic. Also covers a merely-huge positive count.
func TestGGUF_hostileDims_typedError(t *testing.T) {
	cases := []struct {
		name string
		bc   ggufKV
	}{
		{"overflow-negative", kvU64("llama.block_count", 1<<63)},
		{"huge-positive", kvU64("llama.block_count", 1<<40)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := buildGGUF([]ggufKV{
				kvStr("general.architecture", "llama"),
				kvU32("llama.embedding_length", 8),
				tc.bc,
				kvU32("llama.attention.head_count", 4),
				kvU32("llama.attention.head_count_kv", 2),
				kvU32("llama.feed_forward_length", 16),
				kvF32("llama.attention.layer_norm_rms_epsilon", 1e-6),
				kvF32("llama.rope.freq_base", 10000),
			}, []ggufTensorDecl{embedTensor(8, 32)})
			g, err := embed.OpenGGUFBytes(raw)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			defer g.Close()
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("loader panicked on hostile block_count: %v", r)
				}
			}()
			if _, err := buildGGUFWeights(g, quantNone, false); err == nil {
				t.Fatalf("expected a typed error for %s block_count", tc.name)
			}
		})
	}
}

// TestGGUFConfig_hostileMetadata_M16 covers the family builders the old corpus
// never reached: a huge block_count must be a typed error (not a multi-TB makeslice)
// before the per-layer allocation, and a missing head_count that drives hidden/0 must
// be a typed error (not an integer divide panic). ggufConfig is the fuzz contract's
// entry point, so it must honor "never panic" for every architecture.
func TestGGUFConfig_hostileMetadata_M16(t *testing.T) {
	emb := []ggufTensorDecl{embedTensor(8, 32)}
	cases := []struct {
		name string
		kvs  []ggufKV
	}{
		{"granite huge block_count", []ggufKV{
			kvStr("general.architecture", "granitehybrid"),
			kvU32("granitehybrid.embedding_length", 8),
			kvU64("granitehybrid.block_count", 1<<40),
			kvU32("granitehybrid.attention.head_count", 4),
		}},
		{"nemotron huge block_count", []ggufKV{
			kvStr("general.architecture", "nemotron_h"),
			kvU32("nemotron_h.embedding_length", 8),
			kvU64("nemotron_h.block_count", 1<<40),
			kvU32("nemotron_h.attention.head_count", 4),
		}},
		{"llama4 huge block_count", []ggufKV{
			kvStr("general.architecture", "llama4"),
			kvU32("llama4.embedding_length", 8),
			kvU64("llama4.block_count", 1<<40),
			kvU32("llama4.attention.head_count", 4),
		}},
		{"phi3 zero head_count divides hidden", []ggufKV{
			kvStr("general.architecture", "phi3"),
			kvU32("phi3.embedding_length", 8),
			kvU32("phi3.block_count", 2),
			kvU32("phi3.attention.head_count", 0),
			kvU32("phi3.feed_forward_length", 16),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := embed.OpenGGUFBytes(buildGGUF(tc.kvs, emb))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			defer g.Close()
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ggufConfig panicked on hostile metadata: %v", r)
				}
			}()
			if _, err := ggufConfig(g); err == nil {
				t.Fatalf("expected a typed error, got nil")
			}
		})
	}
}

// ggufSeeds builds one tiny, valid GGUF per supported architecture so the mutator
// starts from headers that reach each family config builder.
func ggufSeeds() [][]byte {
	emb := embedTensor(8, 32)
	dense := func(arch string) []byte {
		return buildGGUF([]ggufKV{
			kvStr("general.architecture", arch),
			kvU32(arch+".embedding_length", 8),
			kvU32(arch+".block_count", 2),
			kvU32(arch+".attention.head_count", 4),
			kvU32(arch+".attention.head_count_kv", 2),
			kvU32(arch+".attention.key_length", 2),
			kvU32(arch+".feed_forward_length", 16),
			kvU32(arch+".attention.sliding_window", 4),
			kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6),
			kvF32(arch+".rope.freq_base", 10000),
			kvU32("tokenizer.ggml.eos_token_id", 1),
		}, []ggufTensorDecl{emb})
	}
	gemma4 := buildGGUF([]ggufKV{
		kvStr("general.architecture", "gemma4"),
		kvU32("gemma4.embedding_length", 8),
		kvU32("gemma4.block_count", 2),
		kvU32("gemma4.attention.head_count", 4),
		kvU32("gemma4.attention.head_count_kv", 2),
		kvU32("gemma4.attention.key_length", 4),
		kvU32("gemma4.attention.key_length_swa", 2),
		kvU32("gemma4.feed_forward_length", 16),
		kvU32("gemma4.attention.sliding_window", 4),
		kvU32("gemma4.attention.shared_kv_layers", 1),
		kvU32("gemma4.embedding_length_per_layer_input", 4),
		kvBoolArr("gemma4.attention.sliding_window_pattern", []bool{true, false}),
		kvF32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
	}, []ggufTensorDecl{emb})
	mellum := buildGGUF([]ggufKV{
		kvStr("general.architecture", "mellum"),
		kvU32("mellum.embedding_length", 8),
		kvU32("mellum.block_count", 2),
		kvU32("mellum.attention.head_count", 4),
		kvU32("mellum.attention.head_count_kv", 2),
		kvU32("mellum.attention.key_length", 2),
		kvU32("mellum.feed_forward_length", 16),
		kvU32("mellum.expert_feed_forward_length", 8),
		kvU32("mellum.expert_count", 4),
		kvU32("mellum.expert_used_count", 2),
		kvU32("mellum.attention.sliding_window", 4),
		kvF32("mellum.attention.layer_norm_rms_epsilon", 1e-6),
		kvF32("mellum.rope.freq_base", 10000),
		kvBoolArr("mellum.attention.sliding_window_pattern", []bool{true, false}),
	}, []ggufTensorDecl{emb})

	var seeds [][]byte
	// Every dispatchable architecture gets a seed so the fuzzer exercises its family
	// builder — the div-by-zero / makeslice sites M16 hardened live inside these, and
	// the previous corpus reached none of the last seven (granite/nemotron/deepseek/
	// glm4moe/qwen35moe/phi3/llama4). A generic dense seed is enough to drive each
	// builder's entry; the fuzzer mutates from there.
	for _, a := range []string{
		"llama", "qwen2", "qwen3", "gemma3", "phi3",
		"deepseek2", "glm4moe", "granitehybrid", "nemotron_h", "qwen35moe", "llama4",
	} {
		seeds = append(seeds, dense(a))
	}
	seeds = append(seeds, gemma4, mellum)
	seeds = append(seeds, gU32(0x46554747), buildGGUF(nil, nil)) // raw-edge: truncated, header-only
	return seeds
}

// FuzzGGUFConfig fuzzes goinfer's GGUF metadata → Config interpretation (the
// per-family config builders). With aikit's container parse hardened (v1.2.1),
// arbitrary bytes either fail aikit's typed parse or reach ggufConfig, which must
// never panic. And the dim gate's contract: if validateGGUFDims accepts a config,
// its core dims must be safe to allocate (positive, within bounds) — the gate is
// what stands between a hostile uint64→int overflow and a make() panic/OOM.
func FuzzGGUFConfig(f *testing.F) {
	for _, s := range ggufSeeds() {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		g, err := embed.OpenGGUFBytes(raw)
		if err != nil {
			return // aikit rejected the container — not goinfer's layer
		}
		defer g.Close()
		cfg, err := ggufConfig(g)
		if err != nil {
			return // a typed interpretation error is the correct outcome
		}
		if validateGGUFDims(cfg) == nil { // accepted ⇒ must be safe to allocate
			if cfg.NumLayers <= 0 || cfg.NumLayers > maxGGUFLayers ||
				cfg.HiddenDim <= 0 || cfg.HiddenDim > maxGGUFHidden ||
				cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
				cfg.VocabSize <= 0 || cfg.VocabSize > maxGGUFVocabSize {
				t.Fatalf("validateGGUFDims accepted an unsafe config: %+v", cfg)
			}
		}
	})
}
