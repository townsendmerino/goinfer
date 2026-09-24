//go:build !darwin

package decoder

import "github.com/townsendmerino/aikit/mmap"

// mapGIW maps a .giw with aikit's MAP_PRIVATE read-only mapping. darwin maps it MAP_SHARED instead, for
// Metal's no-copy buffers (giwmap_darwin.go); nothing here wires the mapping.
func mapGIW(path string) ([]byte, error) { return mmap.MapReadOnly(path) }
