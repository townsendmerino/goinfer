//go:build gpu && goinfer_testhooks && !unix

package gpu

// makeWritable is a no-op here: without mmap the checkpoint is read onto the heap, which is writable.
func makeWritable(...[]float32) {}
