//go:build cuda && !linux

package cuda

// allocHostRows is words float32s of ordinary host memory, zero-filled, and the function that returns it (the garbage collector does).
func allocHostRows(words int) ([]float32, func()) { return make([]float32, words), func() {} }
