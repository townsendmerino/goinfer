package multimodal

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/png"
	"strings"
	"testing"
)

// TestQwenPreprocess_inputPixelLimit_M15 gates M-15: an input image whose raw dimensions exceed
// qwenMaxInputPixels is rejected before pixel decode — a small compressible PNG that decodes to
// a huge canvas can no longer drive a ~GB allocation per request ahead of the model mutex. A
// normal-sized image still preprocesses.
func TestQwenPreprocess_inputPixelLimit_M15(t *testing.T) {
	cfg := QwenDefaultPreprocess()
	enc := func(w, h int) []byte {
		var buf bytes.Buffer
		if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	// A blank 12000x12000 PNG (144 MP) compresses tiny but would allocate ~1.7 GB; must be refused.
	if _, _, err := QwenPreprocess(enc(12000, 12000), cfg); err == nil {
		t.Error("144 MP image accepted (M-15): must reject above the input-pixel limit")
	}
	// A realistic image still works.
	if _, _, err := QwenPreprocess(enc(224, 224), cfg); err != nil {
		t.Errorf("small image rejected: %v", err)
	}
}

// pngWithForgedIHDR takes a tiny valid PNG and rewrites its IHDR chunk to declare huge
// dimensions (recomputing the chunk CRC), while leaving the actual (tiny, real) compressed
// pixel data untouched — a stand-in for a decompression bomb: a few bytes on the wire, a
// canvas too large to ever legitimately decode. If QwenPreprocess only checked bounds AFTER a
// full image.Decode, decoding this file would fail with a corrupt-stream error (IDAT doesn't
// match the forged huge canvas) rather than the pixel-limit error — so the specific error
// returned proves whether the size check runs before or after the costly decode.
func pngWithForgedIHDR(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 4, 4))); err != nil {
		t.Fatal(err)
	}
	data := buf.Bytes()
	const sigLen = 8
	// chunk: 4-byte length, 4-byte type, <length> data, 4-byte CRC.
	ihdrStart := sigLen + 8 // skip signature + IHDR's length/type fields
	binary.BigEndian.PutUint32(data[ihdrStart:], uint32(w))
	binary.BigEndian.PutUint32(data[ihdrStart+4:], uint32(h))
	crc := crc32.ChecksumIEEE(data[sigLen+4 : ihdrStart+13]) // type + data, per PNG spec
	binary.BigEndian.PutUint32(data[ihdrStart+13:], crc)
	return data
}

// TestQwenPreprocess_rejectsBeforeDecode proves the pixel-limit check runs BEFORE
// image.Decode, not after: a forged IHDR declares a canvas far past qwenMaxInputPixels while
// the real IDAT payload stays tiny (an un-decodable mismatch). The pre-decode DecodeConfig
// path never touches IDAT, so it must fail with the pixel-limit error specifically — a
// generic decode-failure error here would mean image.Decode (and its full pixel-buffer
// allocation) ran first.
func TestQwenPreprocess_rejectsBeforeDecode(t *testing.T) {
	cfg := QwenDefaultPreprocess()
	_, _, err := QwenPreprocess(pngWithForgedIHDR(t, 20000, 20000), cfg)
	if err == nil {
		t.Fatal("forged-huge-IHDR image accepted: must reject above the input-pixel limit")
	}
	if !strings.Contains(err.Error(), "pixel input limit") {
		t.Errorf("got error %q, want a pixel-input-limit rejection (proves the check ran before image.Decode, not after a decode failure)", err)
	}
}
