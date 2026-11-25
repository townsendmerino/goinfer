package multimodal

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"testing"
)

// fakePNGHeader builds a valid PNG signature + IHDR chunk declaring width x height (no pixel data needed
// because image.DecodeConfig returns right after IHDR).
func fakePNGHeader(width, height uint32) []byte {
	var b bytes.Buffer
	b.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	var ihdr [13]byte
	binary.BigEndian.PutUint32(ihdr[0:4], width)
	binary.BigEndian.PutUint32(ihdr[4:8], height)
	ihdr[8] = 8 // bit depth
	ihdr[9] = 2 // RGB
	var lenBuf [4]byte
	binary.BigEndian.PutUint32(lenBuf[:], uint32(len(ihdr)))
	b.Write(lenBuf[:])
	b.WriteString("IHDR")
	b.Write(ihdr[:])
	crc := crc32.NewIEEE()
	_, _ = crc.Write([]byte("IHDR"))
	_, _ = crc.Write(ihdr[:])
	binary.BigEndian.PutUint32(lenBuf[:], crc.Sum32())
	b.Write(lenBuf[:])
	return b.Bytes()
}

func TestLfm2VLAndPixtral_rejectDecompressionBombArea(t *testing.T) {
	// 16384x16384 (268 MP) passes the per-axis <= 16384 check but exceeds qwenMaxInputPixels (32 MP).
	bomb := fakePNGHeader(16384, 16384)
	if _, err := Lfm2VLPreprocess(bomb, Lfm2VLPreprocessConfig{}); err == nil {
		t.Fatal("Lfm2VLPreprocess accepted 16384x16384 PNG header")
	}
	if _, _, _, err := PixtralPreprocess(bomb, PixtralPreprocessConfig{}); err == nil {
		t.Fatal("PixtralPreprocess accepted 16384x16384 PNG header")
	}
}
