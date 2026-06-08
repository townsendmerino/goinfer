package giw

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func TestBundle_v2_roundtrip(t *testing.T) {
	weights := []byte("the-quantized-weights-blob")
	tok := []byte("metadata-gguf-tokenizer")
	w, tk, err := Read(Write(weights, tok))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(w, weights) || !bytes.Equal(tk, tok) {
		t.Fatalf("round-trip mismatch: weights=%q tok=%q", w, tk)
	}
}

// TestBundle_v1_compat: a v1 bundle (u32 weights length) still loads, so existing
// .giw files keep working after the v2 (u64) bump.
func TestBundle_v1_compat(t *testing.T) {
	weights, tok := []byte("v1-weights"), []byte("v1-tok")
	var b []byte
	b = append(b, bundleMagic...)
	b = binary.LittleEndian.AppendUint32(b, 1) // version 1
	b = binary.LittleEndian.AppendUint32(b, uint32(len(weights)))
	b = append(b, weights...)
	b = binary.LittleEndian.AppendUint32(b, uint32(len(tok)))
	b = append(b, tok...)
	w, tk, err := Read(b)
	if err != nil {
		t.Fatalf("Read v1: %v", err)
	}
	if !bytes.Equal(w, weights) || !bytes.Equal(tk, tok) {
		t.Fatalf("v1 mismatch: weights=%q tok=%q", w, tk)
	}
}

func TestBundle_badVersion(t *testing.T) {
	var b []byte
	b = append(b, bundleMagic...)
	b = binary.LittleEndian.AppendUint32(b, 99)
	if _, _, err := Read(b); err == nil {
		t.Fatal("expected version error")
	}
}

// TestBundle_over4GiB proves the v2 u64 length round-trips a weights blob past
// the v1 u32 (4 GiB) ceiling — the 7B-int4 regime. Allocates >4 GiB, so it is
// opt-in (GIW_BIG=1); the real integration gate is `prequant --quant int4` on a
// 7B, which self-checks emit→Read→LoadSerializedWeights.
func TestBundle_over4GiB(t *testing.T) {
	if os.Getenv("GIW_BIG") == "" {
		t.Skip("set GIW_BIG=1 (allocates >4 GiB)")
	}
	const n = (1 << 32) + 1024 // just past 2^32, where v1 truncated
	weights := make([]byte, n)
	weights[0], weights[n-1] = 0xAB, 0xCD // sentinels at both ends
	tok := []byte("tok")
	w, _, err := Read(Write(weights, tok))
	if err != nil {
		t.Fatalf("Read >4 GiB: %v", err)
	}
	if len(w) != n || w[0] != 0xAB || w[n-1] != 0xCD {
		t.Fatalf(">4 GiB truncated: len=%d (want %d) first=%#x last=%#x", len(w), n, w[0], w[n-1])
	}
}
