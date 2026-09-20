package decoder

// philox4x32 is Philox4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2,
// 3", SC'11): a counter-based generator, so the k-th output is a pure function of (key, counter) with no
// state to carry. That is what lets a GPU kernel draw one variate per vocabulary entry in parallel and a
// host reimplementation draw the identical bits — the whole point of using it for temperature-only sampling
// (docs/tasks/red-october.md R7b).
//
// It is INTEGER-ONLY (32-bit wrapping multiplies, xors, adds), so every backend can compute exactly the
// same bits: CUDA (__umulhi), Metal (mulhi), and WebGPU (no 64-bit integers; a 16-bit-split mulhi). Only
// the float transform applied afterwards (sampler_gumbel.go) is allowed to differ in the last ulps.
//
// Verified against the Random123 known-answer vectors (philox_test.go); do not "simplify" the round.
const (
	philoxM0 = 0xD2511F53
	philoxM1 = 0xCD9E8D57
	philoxW0 = 0x9E3779B9 // golden ratio
	philoxW1 = 0xBB67AE85 // sqrt(3) - 1
)

func philox4x32(ctr [4]uint32, key [2]uint32) [4]uint32 {
	for r := 0; r < 10; r++ {
		if r > 0 { // the key is bumped between rounds, not before the first
			key[0] += philoxW0
			key[1] += philoxW1
		}
		p0 := uint64(philoxM0) * uint64(ctr[0])
		p1 := uint64(philoxM1) * uint64(ctr[2])
		ctr = [4]uint32{
			uint32(p1>>32) ^ ctr[1] ^ key[0],
			uint32(p1),
			uint32(p0>>32) ^ ctr[3] ^ key[1],
			uint32(p0),
		}
	}
	return ctr
}
