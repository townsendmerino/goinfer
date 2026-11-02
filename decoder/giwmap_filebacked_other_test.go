//go:build !unix && !windows

package decoder

// giwMapIsFileBacked: aikit's mmap.MapReadOnly reads the whole file onto the heap on the remaining platforms, so a .giw at every
// quant is heap memory there, and aliasing changes nothing about what it costs.
const giwMapIsFileBacked = false
