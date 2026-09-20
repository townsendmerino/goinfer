package decoder

import "testing"

// TestPhilox4x32_knownAnswers pins philox4x32 to the Random123 reference implementation's published
// known-answer vectors (kat_vectors, "philox4x32 10"): counter[4], key[2] -> output[4]. Every backend's
// Philox (CUDA, Metal, WebGPU) must produce these same bits, because the whole temperature-only sampler
// relies on the integer words being identical everywhere; this is the anchor they are all held to.
func TestPhilox4x32_knownAnswers(t *testing.T) {
	for _, c := range []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{0, 0, 0, 0}, [2]uint32{0, 0}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{[4]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, [2]uint32{0xffffffff, 0xffffffff}, [4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, [2]uint32{0xa4093822, 0x299f31d0}, [4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
	} {
		if got := philox4x32(c.ctr, c.key); got != c.want {
			t.Errorf("philox4x32(%08x, %08x) = %08x, want %08x", c.ctr, c.key, got, c.want)
		}
	}
}
