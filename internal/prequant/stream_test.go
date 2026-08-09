package prequant

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGIWRoundTripPreservesRouterBias guards the .giw serialize format against
// silently dropping per-layer fields. The byte-identity test above can't catch a
// field BOTH the streamed and resident paths skip (they share the serializer); only
// a transcode → load → inspect round-trip does. RouterBias (GLM/DeepSeek
// e_score_correction_bias) was added to LayerWeights but initially not to the giw
// format, so a stream-weights GLM lost its routing bias — caught only by the real
// 106B gate. This pins it on the tiny GLM model.
func TestGIWRoundTripPreservesRouterBias(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	// The tiny fixture carries no tokenizer, so go through the weights serializer
	// directly (StreamTranscodeGGUF → LoadSerializedWeights) rather than the full
	// .giw bundle — it's the serialize round-trip we're guarding.
	var body bytes.Buffer
	if _, err := decoder.StreamTranscodeGGUF(context.Background(), gguf, &body, "int4", false, "glm-tiny"); err != nil {
		t.Fatalf("StreamTranscodeGGUF: %v", err)
	}
	w, err := decoder.LoadSerializedWeights(body.Bytes())
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	layers := w.Layers
	if layers[0].Experts != nil {
		t.Errorf("layer 0 (dense prefix) has experts after round-trip")
	}
	for i := 1; i < len(layers); i++ {
		if len(layers[i].RouterBias) == 0 {
			t.Errorf("layer %d RouterBias dropped by the .giw round-trip", i)
		}
	}
}

// TestStreamTranscode_ctxCancel_M21 gates the M-21 cancellation contract: an
// already-cancelled context makes StreamTranscodeGGUF abort with the context error
// instead of running the whole multi-GB pass. The sink counts bytes so we also prove
// nothing was written (the ctxWriter refuses the first write).
func TestStreamTranscode_ctxCancel_M21(t *testing.T) {
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled
	var body bytes.Buffer
	n, err := decoder.StreamTranscodeGGUF(ctx, gguf, &body, "int4", false, "glm-tiny")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("StreamTranscodeGGUF with a cancelled ctx = (%d, %v); want a context.Canceled error", n, err)
	}
	if body.Len() != 0 {
		t.Errorf("wrote %d bytes after cancellation; want 0 (the transcode must abort before writing)", body.Len())
	}
}

// giwSplit splits a .giw bundle at the v5 quant-label field and verifies the trailing CRC32-IEEE is
// valid for that bundle. Returns the bytes before the label field and the bytes after it (both
// excluding the CRC). A wrong CRC or a wrong label is a FAILURE, not an exemption — that is what
// keeps the comparison below strict rather than "modulo the label".
func giwSplit(t *testing.T, b []byte, at int, wantLabel string) (pre, post []byte) {
	t.Helper()
	if len(b) < 4 {
		t.Fatalf("bundle too short to carry a CRC (%d B)", len(b))
	}
	body, crc := b[:len(b)-4], binary.LittleEndian.Uint32(b[len(b)-4:])
	if got := crc32.ChecksumIEEE(body); got != crc {
		t.Fatalf("CRC invalid: stored %08x, computed %08x", crc, got)
	}
	if at+4 > len(body) {
		t.Fatalf("label offset %d past end of body (%d B)", at, len(body))
	}
	n := int(binary.LittleEndian.Uint32(body[at : at+4]))
	if n != len(wantLabel) {
		t.Fatalf("label length word at %d = %d, want %d (for %q)", at, n, len(wantLabel), wantLabel)
	}
	if got := string(body[at+4 : at+4+n]); got != wantLabel {
		t.Fatalf("label at %d = %q, want %q", at, got, wantLabel)
	}
	return body[:at], body[at+4+n:]
}

// giwLabelOffset returns the byte offset of the v5 quant-label field by PARSING the .giw header,
// per the format spec in decoder/serialize.go:
//
//	magic [5]byte "GINFW" | version u32 | quant u32 | id str | config str | quantLabel str | …
//	str = u32 length + that many bytes
//
// Parsing beats locating the field as "the first byte where the two bundles differ": that would
// silently relocate itself if the bundles ever diverged EARLIER for some other reason, quietly
// measuring the wrong field and reporting the wrong thing. Here a layout change fails loudly with a
// specific message instead.
func giwLabelOffset(t *testing.T, b []byte) int {
	t.Helper()
	const magic = "GINFW"
	if len(b) < len(magic)+8 {
		t.Fatalf("bundle too short to hold a header (%d B)", len(b))
	}
	if got := string(b[:len(magic)]); got != magic {
		t.Fatalf("bad magic %q, want %q", got, magic)
	}
	off := len(magic) + 4 + 4 // magic + version + legacy quant enum
	// Skip the two length-prefixed strings that precede the label: id, then config JSON.
	for _, field := range []string{"id", "config"} {
		if off+4 > len(b) {
			t.Fatalf("header truncated before %s field (offset %d, len %d)", field, off, len(b))
		}
		n := int(binary.LittleEndian.Uint32(b[off : off+4]))
		if n < 0 || off+4+n > len(b) {
			t.Fatalf("%s field length %d at offset %d overruns the bundle (%d B)", field, n, off, len(b))
		}
		off += 4 + n
	}
	return off
}

// transcodeBothWays returns the resident and streamed .giw bundles for one quant mode, plus the
// resolved quant label the buffer path records.
func transcodeBothWays(t *testing.T, gguf, quant string) (resident, streamed []byte, label string) {
	t.Helper()
	m, err := decoder.Load(gguf, decoder.Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	label = m.Quant()
	resident, err = decoder.SerializeWeights(m.Weights(), "glm-tiny.gguf")
	m.Close()
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	var buf bytes.Buffer
	if _, err := decoder.StreamTranscodeGGUF(context.Background(), gguf, &buf, quant, false, "glm-tiny.gguf"); err != nil {
		t.Fatalf("StreamTranscodeGGUF: %v", err)
	}
	return resident, buf.Bytes(), label
}

func giwFixture(t *testing.T) string {
	t.Helper()
	gguf := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no tiny GLM GGUF at %s — run scripts/pin_glm_tiny_gguf.py", gguf)
	}
	return gguf
}

// TestGiwQuantLabel_headerAsymmetry pins a v5 behaviour that until now lived only as a comment in
// decoder/serialize.go: the BUFFER path (full weights in hand) records the resolved quant label,
// while the STREAMING transcode records "" — it writes the header BEFORE its layers load and cannot
// yet know the resolved quant, so a reader of a streamed bundle falls back to inference (the pre-v5
// behaviour).
//
// This is a gate, not a nicety. That asymmetry is exactly what made the old byte-identity assertion
// wrong the day ac6977f landed, and nothing caught it: testdata/glm-tiny.gguf was untracked, so the
// test self-skipped in CI and had never once run there.
func TestGiwQuantLabel_headerAsymmetry(t *testing.T) {
	gguf := giwFixture(t)
	for _, quant := range []string{"int4", "int8int8", ""} {
		t.Run("quant="+quant, func(t *testing.T) {
			resident, streamed, label := transcodeBothWays(t, gguf, quant)
			if label == "" {
				t.Fatalf("Model.Quant() is empty for quant=%q; the buffer path must resolve a label", quant)
			}
			giwSplit(t, resident, giwLabelOffset(t, resident), label) // buffer path: resolved label
			giwSplit(t, streamed, giwLabelOffset(t, streamed), "")    // streaming path: absent
		})
	}
}

// TestStreamTranscodeMatchesResident proves the streaming transcode (one layer at a time, for a
// model too large to hold resident) produces the same .giw WEIGHTS as the resident path (Load +
// SerializeWeights). Same input bytes → same per-tensor quantization regardless of load order, so
// the streaming path is a drop-in, not an approximation.
//
// Byte-identity is asserted over everything outside the two fields that MUST differ, and each of
// those is pinned exactly rather than skipped: the v5 quant label (see the asymmetry test above;
// giwSplit re-asserts it here) and the trailing CRC32 covering it (giwSplit verifies each bundle's
// CRC independently, so a wrong CRC fails). Measured on this fixture, every remaining byte is equal
// across all three quant modes — a size delta alone would not have shown that, and did not: the
// original diagnosis missed the CRC entirely.
func TestStreamTranscodeMatchesResident(t *testing.T) {
	gguf := giwFixture(t)
	for _, quant := range []string{"int4", "int8int8", ""} {
		t.Run("quant="+quant, func(t *testing.T) {
			resident, streamed, label := transcodeBothWays(t, gguf, quant)
			rPre, rPost := giwSplit(t, resident, giwLabelOffset(t, resident), label)
			sPre, sPost := giwSplit(t, streamed, giwLabelOffset(t, streamed), "")

			if !bytes.Equal(rPre, sPre) {
				t.Errorf("bundles differ BEFORE the quant label (%d vs %d B) — the header itself diverges",
					len(rPre), len(sPre))
			}
			if !bytes.Equal(rPost, sPost) {
				n := 0
				for i := 0; i < len(rPost) && i < len(sPost); i++ {
					if rPost[i] != sPost[i] {
						n++
					}
				}
				t.Fatalf("bundles differ AFTER the quant label: %d of %d bytes (lengths %d vs %d) — the "+
					"streaming and resident paths do NOT produce the same weights",
					n, len(rPost), len(rPost), len(sPost))
			}
		})
	}
}
