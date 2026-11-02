//go:build !unix

package decoder

// giwMapIsFileBacked: aikit's mmap.MapReadOnly reads the whole file onto the heap off unix, Windows included, so a .giw at every
// quant is heap memory there, and aliasing changes nothing about what it costs (docs/queue-engineering.md, B43).
const giwMapIsFileBacked = false
