package giw

import (
	"bytes"
	"encoding/binary"
	"io"
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

// TestWriteStream_matchesWrite: the streaming framer (placeholder length + WriteAt
// patch, used to prequant a large model without a full-blob RAM spike) must produce
// a file byte-identical to the in-memory Write, and Read must split it correctly.
func TestWriteStream_matchesWrite(t *testing.T) {
	weights := []byte("the-quantized-weights-blob-streamed-in-chunks")
	tok := []byte("metadata-gguf-tokenizer")

	path := t.TempDir() + "/bundle.giw"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	err = WriteStream(f, tok, func(w io.Writer) (int64, error) {
		// stream the weights in two chunks to exercise the running byte count
		n1, _ := w.Write(weights[:10])
		n2, e := w.Write(weights[10:])
		return int64(n1 + n2), e
	})
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		t.Fatalf("WriteStream: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := Write(weights, tok); !bytes.Equal(got, want) {
		t.Fatalf("streamed bundle differs from Write:\n got  %q\n want %q", got, want)
	}
	w, tk, err := Read(got)
	if err != nil {
		t.Fatalf("Read streamed bundle: %v", err)
	}
	if !bytes.Equal(w, weights) || !bytes.Equal(tk, tok) {
		t.Fatalf("round-trip mismatch: weights=%q tok=%q", w, tk)
	}
}

// TestReadTokFile_matchesRead: reading just the tokenizer half from a bundle file
// (seek past the weights, used to serve a large .giw without aliasing its tens of
// GB of weights) returns the same bytes as Read's tok half — for both v2 and v1
// frames.
func TestReadTokFile_matchesRead(t *testing.T) {
	weights := []byte("a-multi-gb-weights-blob-stand-in")
	tok := []byte("the-metadata-gguf-tokenizer-half")

	// v2 (via Write)
	pathV2 := t.TempDir() + "/v2.giw"
	if err := os.WriteFile(pathV2, Write(weights, tok), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTokFile(pathV2)
	if err != nil {
		t.Fatalf("ReadTokFile v2: %v", err)
	}
	if !bytes.Equal(got, tok) {
		t.Fatalf("v2 tok = %q, want %q", got, tok)
	}

	// v1 frame (u32 weights length), hand-built like TestBundle_v1_compat
	var v1 []byte
	v1 = append(v1, bundleMagic...)
	v1 = binary.LittleEndian.AppendUint32(v1, 1)
	v1 = binary.LittleEndian.AppendUint32(v1, uint32(len(weights)))
	v1 = append(v1, weights...)
	v1 = binary.LittleEndian.AppendUint32(v1, uint32(len(tok)))
	v1 = append(v1, tok...)
	pathV1 := t.TempDir() + "/v1.giw"
	if err := os.WriteFile(pathV1, v1, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := ReadTokFile(pathV1); err != nil || !bytes.Equal(got, tok) {
		t.Fatalf("v1 ReadTokFile = %q, err=%v, want %q", got, err, tok)
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

// TestBundle_hostileLength_noPanic is the regression for the cur.take overflow
// FuzzGIWRead found: a v2 weights length near maxint64 made c.off+n wrap negative,
// slipping past the bound check and panicking the slice. Read must return a typed
// error instead.
func TestBundle_hostileLength_noPanic(t *testing.T) {
	var b []byte
	b = append(b, bundleMagic...)
	b = binary.LittleEndian.AppendUint32(b, 2)                  // v2
	b = binary.LittleEndian.AppendUint64(b, 0x7fffffffffffffff) // ~maxint64 weights length
	b = append(b, "some-bytes"...)
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Read panicked on hostile v2 length: %v", r)
		}
	}()
	if _, _, err := Read(b); err == nil {
		t.Fatal("expected a short-read error for an oversized weights length")
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
