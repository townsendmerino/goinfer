package ggufconfig

import (
	"bytes"
	"encoding/binary"
	"math"
)

// A minimal, faithful GGUF encoder (matches aikit/embed.parseGGUF), the same one decoder/gguf_dims_test.go carries for
// the loader's tests: test files cannot share code across packages.

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

func gU32(v uint32) []byte { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, v); return b }

func gU64(v uint64) []byte { b := make([]byte, 8); binary.LittleEndian.PutUint64(b, v); return b }

func gF32(v float32) []byte { return gU32(math.Float32bits(v)) }

func gStr(s string) []byte { return append(gU64(uint64(len(s))), s...) }

func kvU32(k string, v uint32) ggufKV { return ggufKV{k, gtUint32, gU32(v)} }

func kvU64(k string, v uint64) ggufKV { return ggufKV{k, gtUint64, gU64(v)} }

func kvF32(k string, v float32) ggufKV { return ggufKV{k, gtFloat32, gF32(v)} }

func kvStr(k, v string) ggufKV { return ggufKV{k, gtString, gStr(v)} }

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
