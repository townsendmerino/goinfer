//go:build goinfer_testhooks

package decoder

// SetCUDAKVF16DefaultForTest turns the CUDA f16 KV default (kv16_cuda.go:cudaKVF16Default) on or off for the process and returns what undoes it. The cuda
// package's tests pin it OFF at start, so every contract test that loads a resident without a -kv keeps the exact f32 cache it was written against (the way its
// TestMain pins the flash-decode lane OFF); the tests of the default itself, and of f16, set it or ask for "f16" by name.
func SetCUDAKVF16DefaultForTest(on bool) (restore func()) {
	prev := cudaKVF16Default.Swap(on)
	return func() { cudaKVF16Default.Store(prev) }
}
