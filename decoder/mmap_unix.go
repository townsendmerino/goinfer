//go:build unix

package decoder

import (
	"fmt"
	"os"
	"syscall"
)

// mmapReadOnly returns a read-only MAP_PRIVATE mapping of path's whole contents.
// The fd is closed before returning — the mapping survives it. This is the seam
// that makes a .giw's aliased int8/int4 weights *pageable*: the bytes live in the
// OS page cache, faulted in lazily and evictable, instead of being eagerly copied
// onto the Go heap (the prerequisite for weight streaming / expert demand-paging).
//
// This is the unix implementation (darwin/linux/bsd); the //go:build !unix sibling
// in mmap_other.go falls back to a heap read so the package still compiles on
// Windows et al. — callers are platform-agnostic against this pair.
func mmapReadOnly(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // fd no longer needed after mmap

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	sz := st.Size()
	if sz < 8 {
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, sz)
	}
	if sz > int64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("mmap %s: file too large for this platform (%d bytes)", path, sz)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(sz), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}

// munmap releases a mapping returned by mmapReadOnly. Safe on a nil/empty slice.
func munmap(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	return syscall.Munmap(b)
}
