//go:build !unix

package decoder

import (
	"fmt"
	"os"
)

// mmapReadOnly is the non-unix fallback for platforms without syscall.Mmap
// (notably Windows): it reads the whole file into the Go heap and returns it.
// Same interface as the unix mmap path, so the .giw loader works unchanged — the
// only difference is the bytes live in the heap instead of the OS page cache, so
// on these platforms a .giw costs the same RAM as before (and the residency
// seam's evict/prefetch hints are inert: nothing to page).
func mmapReadOnly(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, len(data))
	}
	return data, nil
}

// munmap is a no-op where mmapReadOnly returns a heap slice (the GC reclaims it).
func munmap([]byte) error { return nil }
