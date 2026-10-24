//go:build cuda && !amd64

package cuda

// moeCPUAsm is false off amd64: the host expert GEMVs run w4a8RowRef.
const moeCPUAsm = false

func w4a8RowsAVX2(w *uint32, s *uint16, aLo, aHi, sum8 *int32, kwords, rows int, out *float32) {
	panic("cuda: w4a8RowsAVX2 has no implementation off amd64")
}
