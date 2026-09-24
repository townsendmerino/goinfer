//go:build darwin

package decoder

import (
	"fmt"
	"os"
	"syscall"
)

// mapGIW maps a .giw read-only and MAP_SHARED on darwin, where every other platform uses aikit's
// MAP_PRIVATE mmap.MapReadOnly (giwmap_other.go).
//
// Why shared, and why only here (docs/measurements/s6-alias-2026-09-24.md; the probe is
// metal/alias_sharedprobe_test.go): a Metal no-copy buffer over a window of the mapping is wired by IOKit
// with WRITE intent. Over a PRIVATE mapping that turns every page the GPU reads into a wired anonymous
// copy-on-write copy (+15,131 COW faults for a 16,384-page window, persisting for the mapping's life) — so
// "aliasing" the weights saved no memory — and marks the object true_share, after which any fork() copies
// the whole mapping (the M26 collapse, docs/measurements/m26-alias-fork-collapse-2026-09-24.md). Over a
// SHARED read-only mapping the GPU reads the file's own page-cache pages: 0 COW faults, correct values,
// fork+exec 4-5 ms while wired. The CPU reads either mapping identically.
//
// darwin only: the hazard is XNU's and Metal's, and Linux's expert pager is tuned for the private mapping.
// Nothing writes through the mapping (PROT_READ), and a .giw is replaced by temp+rename, never rewritten in
// place, so a shared mapping cannot observe a concurrent rewrite. Release with mmap.Unmap as before.
func mapGIW(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close() // the mapping survives the fd
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	sz := st.Size()
	if sz < 8 { // same floor as mmap.MapReadOnly: no valid bundle is shorter
		return nil, fmt.Errorf("mmap %s: file too small (%d bytes)", path, sz)
	}
	if sz > int64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("mmap %s: file too large for this platform (%d bytes)", path, sz)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(sz), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return data, nil
}
