//go:build gpu

package gpu

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// TestInt4LayoutMatch checks the hypothesis behind the fast int4 resident upload: the
// decoder's int4 storage (2 nibbles/byte, elem k → byte k>>1, low nibble if even) is
// BYTE-IDENTICAL to the GPU packNibbles layout (8 nibbles/u32, elem k at nibble k%8 of word
// k/8) when K is a multiple of 32 (so kp==K, no row padding). If so, the resident upload can
// CreateBufferInit the decoder bytes directly — skipping the unpack + packNibbles that cost
// ~30 s on a 12 B model — with no new .giw format. Same nibble value convention (value+8)
// on both sides, so values are preserved too.
func TestInt4LayoutMatch(t *testing.T) {
	for _, dims := range [][2]int{{4, 32}, {3, 64}, {5, 256}, {2, 4096}} {
		N, K := dims[0], dims[1]
		un := make([]uint8, N*K)
		for i := range un {
			un[i] = uint8((i*7 + 3) % 16) // arbitrary nibbles 0..15
		}
		// GPU layout (what the resident upload currently produces).
		gpuWords := packNibbles(un, N, K)
		gpuBytes := make([]byte, len(gpuWords)*4)
		for i, w := range gpuWords {
			binary.LittleEndian.PutUint32(gpuBytes[i*4:], w)
		}
		// Decoder layout (what a .giw / safetensors int4 WeightMat stores).
		dec := make([]byte, N*((K+1)/2))
		for r := range N {
			row := dec[r*((K+1)/2):]
			for k := range K {
				if k&1 == 0 {
					row[k>>1] |= un[r*K+k] & 0xF
				} else {
					row[k>>1] |= (un[r*K+k] & 0xF) << 4
				}
			}
		}
		if !bytes.Equal(dec, gpuBytes) {
			t.Errorf("N=%d K=%d: decoder int4 layout != GPU layout (len dec=%d gpu=%d)", N, K, len(dec), len(gpuBytes))
		} else {
			t.Logf("N=%d K=%d: decoder int4 bytes == GPU bq bytes (%d B) — direct upload valid", N, K, len(dec))
		}
	}
}
