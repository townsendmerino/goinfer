//go:build !arm64

package kernelprint

// row4Emit has no repack off arm64: the .giw writer emits row4 only there, so the reader sees canonical bytes.
func row4Emit(q4 []byte, q4s []uint16, rows, cols, group int) ([]byte, []uint16) { return q4, q4s }
