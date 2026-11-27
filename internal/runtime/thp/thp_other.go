//go:build !linux

package thp

// Small runs alloc; off Linux there are no transparent huge pages to keep the buffers off.
func Small(alloc func() [][]byte) { alloc() }
