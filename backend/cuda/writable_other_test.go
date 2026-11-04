//go:build cuda && !unix

package cuda

// makeWritable is a no-op here: without mmap the checkpoint is read onto the heap, which is writable.
func makeWritable(...[]float32) {}
