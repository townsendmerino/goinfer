package giw

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
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
// GB of weights) returns the same bytes as Read's tok half.
func TestReadTokFile_matchesRead(t *testing.T) {
	weights := []byte("a-multi-gb-weights-blob-stand-in")
	tok := []byte("the-metadata-gguf-tokenizer-half")

	path := t.TempDir() + "/b.giw"
	if err := os.WriteFile(path, Write(weights, tok), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTokFile(path)
	if err != nil {
		t.Fatalf("ReadTokFile: %v", err)
	}
	if !bytes.Equal(got, tok) {
		t.Fatalf("tok = %q, want %q", got, tok)
	}
}

// TestBundle_olderFramesAreRefused: frames v1 (u32 weights length) and v2 (u64 length, blob at offset 17) are not
// read. Each is built by hand, whole and well-formed, so the refusal is for the version and not for a short read, and
// the error names the fix.
func TestBundle_olderFramesAreRefused(t *testing.T) {
	weights, tok := []byte("old-weights-blob"), []byte("old-tok")
	var v1, v2 []byte
	v1 = append(v1, bundleMagic...)
	v1 = binary.LittleEndian.AppendUint32(v1, 1)
	v1 = binary.LittleEndian.AppendUint32(v1, uint32(len(weights)))
	v1 = append(append(v1, weights...), binary.LittleEndian.AppendUint32(nil, uint32(len(tok)))...)
	v1 = append(v1, tok...)
	v2 = append(v2, bundleMagic...)
	v2 = binary.LittleEndian.AppendUint32(v2, 2)
	v2 = binary.LittleEndian.AppendUint64(v2, uint64(len(weights)))
	v2 = append(append(v2, weights...), binary.LittleEndian.AppendUint32(nil, uint32(len(tok)))...)
	v2 = append(v2, tok...)
	// Long enough that every fixed-offset header read succeeds, so only the version can refuse them.
	pad := make([]byte, bundleBlobOffsetV3+16)
	dir := t.TempDir()
	for name, b := range map[string][]byte{"v1": append(v1, pad...), "v2": append(v2, pad...)} {
		want := "this build reads v3"
		if _, _, err := Read(b); err == nil || !strings.Contains(err.Error(), want) || !strings.Contains(err.Error(), "prequant") {
			t.Errorf("Read %s: %v, want a refusal naming the version and the rebuild", name, err)
		}
		path := filepath.Join(dir, name+".giw")
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadTokFile(path); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("ReadTokFile %s: %v, want a version refusal", name, err)
		}
		if _, err := WeightsVersionFile(path); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("WeightsVersionFile %s: %v, want a version refusal", name, err)
		}
	}
}

// TestBundle_v3_blobStartsAligned pins the property v3 exists for: the weights blob begins at file
// offset bundleBlobOffsetV3 (a multiple of 16), so a mapping of the file has it 16-byte aligned —
// through Write AND WriteStream, and both readers agree. Mutating the offset (e.g. back to the 17-byte
// v2 header) turns this red.
func TestBundle_v3_blobStartsAligned(t *testing.T) {
	if bundleBlobOffsetV3%16 != 0 {
		t.Fatalf("bundleBlobOffsetV3 = %d is not a multiple of 16", bundleBlobOffsetV3)
	}
	weights, tok := bytes.Repeat([]byte{0xAB}, 1000), []byte("tok-gguf")
	b := Write(weights, tok)
	if got := binary.LittleEndian.Uint32(b[5:9]); got != 3 {
		t.Fatalf("Write emitted bundle version %d, want 3", got)
	}
	if !bytes.Equal(b[bundleBlobOffsetV3:bundleBlobOffsetV3+len(weights)], weights) {
		t.Fatal("Write: weights blob is not at offset bundleBlobOffsetV3")
	}
	for i, c := range b[17:bundleBlobOffsetV3] {
		if c != 0 {
			t.Fatalf("header pad byte %d is %#x, want zero", 17+i, c)
		}
	}
	w, tk, err := Read(b)
	if err != nil || !bytes.Equal(w, weights) || !bytes.Equal(tk, tok) {
		t.Fatalf("Read v3: %v", err)
	}
	// The blob half must ALIAS data at the aligned offset (zero-copy is the point).
	if &w[0] != &b[bundleBlobOffsetV3] {
		t.Error("Read v3 did not alias the weights blob at the padded offset")
	}

	path := filepath.Join(t.TempDir(), "v3.giw")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteStream(f, tok, func(dst io.Writer) (int64, error) {
		n, err := dst.Write(weights)
		return int64(n), err
	}); err != nil {
		t.Fatal(err)
	}
	f.Close()
	streamed, _ := os.ReadFile(path)
	if !bytes.Equal(streamed, b) {
		t.Fatal("WriteStream is not byte-identical to Write for v3")
	}
	got, err := ReadTokFile(path)
	if err != nil || !bytes.Equal(got, tok) {
		t.Fatalf("ReadTokFile v3: %v %q", err, got)
	}
}

// TestBundle_hostileLength_noPanic pins that a weights length near maxint64, which made c.off+n wrap negative in
// cur.take and slip past the bound check into a panicking slice, makes Read return a typed error instead. Origin
// (FuzzGIWRead): docs/code-notes/internal-giw.md#TestBundle_hostileLength_noPanic.
func TestBundle_hostileLength_noPanic(t *testing.T) {
	var b []byte
	b = append(b, bundleMagic...)
	b = binary.LittleEndian.AppendUint32(b, bundleVersion)
	b = binary.LittleEndian.AppendUint64(b, 0x7fffffffffffffff) // ~maxint64 weights length
	b = append(b, make([]byte, bundleBlobOffsetV3)...)          // past the header pad, so the length is what fails
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Read panicked on a hostile weights length: %v", r)
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

// TestBundle_over4GiB proves the u64 length round-trips a weights blob past
// 4 GiB — the 7B-int4 regime. Allocates >4 GiB, so it is
// opt-in (GIW_BIG=1); the real integration gate is `prequant --quant int4` on a
// 7B, which self-checks emit→Read→LoadSerializedWeights.
func TestBundle_over4GiB(t *testing.T) {
	if os.Getenv("GIW_BIG") == "" {
		t.Skip("set GIW_BIG=1 (allocates >4 GiB)")
	}
	const n = (1 << 32) + 1024 // just past 2^32, where a u32 length would truncate
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

// TestReadTokFile_boundsAlloc is the C-31 gate: ReadTokFile must not allocate from an unbounded
// file-controlled u32 tok length. The exploit (a header, weightsLen=0, tokLen=0xFFFFFFFF)
// would make a 4 GiB allocation before the trailing ReadAt fails; now it's rejected up front. A valid
// bundle written to a file still round-trips (the bound doesn't reject honest files).
func TestReadTokFile_boundsAlloc(t *testing.T) {
	dir := t.TempDir()

	// the exploit: tokLen far larger than the file.
	bad := make([]byte, bundleBlobOffsetV3+4)
	copy(bad, bundleMagic)
	binary.LittleEndian.PutUint32(bad[5:9], bundleVersion)
	binary.LittleEndian.PutUint64(bad[9:17], 0)                         // weightsLen 0, so the tok length is at the blob offset
	binary.LittleEndian.PutUint32(bad[bundleBlobOffsetV3:], 0xFFFFFFFF) // tokLen 4 GiB
	badPath := filepath.Join(dir, "bad.giw")
	if err := os.WriteFile(badPath, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	// The refusal must be the length bound, not some earlier check: any other error would leave the bound untested.
	if _, err := ReadTokFile(badPath); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("ReadTokFile on a tokLen far larger than the file: %v, want the length bound's refusal (C-31)", err)
	}

	// control: a valid bundle round-trips through ReadTokFile.
	tok := []byte("tokenizer-gguf-bytes")
	goodPath := filepath.Join(dir, "good.giw")
	if err := os.WriteFile(goodPath, Write([]byte("weights"), tok), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTokFile(goodPath)
	if err != nil {
		t.Fatalf("ReadTokFile rejected a valid bundle: %v", err)
	}
	if string(got) != string(tok) {
		t.Fatalf("ReadTokFile returned %q, want %q", got, tok)
	}
}

// TestWeightsVersionFile reads the blob version out of a bundle, and refuses a bad magic and a truncated file.
func TestWeightsVersionFile(t *testing.T) {
	blob := append([]byte(weightsMagic), 15, 0, 0, 0, 0xAA, 0xBB) // version 15, then payload
	cases := map[string][]byte{"v3": Write(blob, []byte("tok"))}
	dir := t.TempDir()
	for name, b := range cases {
		p := filepath.Join(dir, name+".giw")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		v, err := WeightsVersionFile(p)
		if err != nil || v != 15 {
			t.Errorf("%s: got (%d, %v), want (15, nil)", name, v, err)
		}
	}
	for name, b := range map[string][]byte{
		"bundle magic": append([]byte("XXXXX"), cases["v3"][5:]...),
		"weights magic": func() []byte {
			b := append([]byte(nil), cases["v3"]...)
			copy(b[bundleBlobOffsetV3:], "XXXXX")
			return b
		}(),
		"truncated": cases["v3"][:bundleBlobOffsetV3+3],
	} {
		p := filepath.Join(dir, "bad.giw")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := WeightsVersionFile(p); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
