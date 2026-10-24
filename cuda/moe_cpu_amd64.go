//go:build cuda && amd64

package cuda

// w4a8RowsAVX2 is w4a8RowRef over rows consecutive rows (moe_cpu_amd64.s), bit-identical to it; kwords must be a multiple of 8.
//
//go:noescape
func w4a8RowsAVX2(w *uint32, s *uint16, aLo, aHi, sum8 *int32, kwords, rows int, out *float32)

func cpuidex(leaf, sub uint32) (a, b, c, d uint32)

func xgetbv0() (lo uint32)

// moeCPUAsm reports whether this CPU runs w4a8RowsAVX2: AVX, AVX2, FMA and F16C, with the OS saving the YMM state.
var moeCPUAsm = func() bool {
	const avx, fma, f16c, osxsave = 1 << 28, 1 << 12, 1 << 29, 1 << 27
	_, _, c1, _ := cpuidex(1, 0)
	if c1&(avx|fma|f16c|osxsave) != avx|fma|f16c|osxsave || xgetbv0()&6 != 6 {
		return false
	}
	_, b7, _, _ := cpuidex(7, 0)
	return b7&(1<<5) != 0
}()
