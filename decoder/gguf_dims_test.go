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
// panic. This file holds the regression for the makeslice panic that surfaced
// (block_count overflowing int → negative NumLayers), plus a minimal GGUF
// encoder. The broader FuzzGGUFConfig target is deferred until the fuzz-hardened
// aikit lands: the container parse itself (string lengths, map pre-sizing) is
// aikit's and still panics on hostile input, so a full fuzz of this surface
// belongs with that upgrade, not here.

// --- a minimal, faithful GGUF encoder (matches aikit/embed.parseGGUF) ---

// gguf metadata value type tags (aikit embed gguf value types).
const (
	gtUint32  uint32 = 4
	gtFloat32 uint32 = 6
	gtString  uint32 = 8
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
			if _, err := buildGGUFWeights(g, quantNone); err == nil {
				t.Fatalf("expected a typed error for %s block_count", tc.name)
			}
		})
	}
}
