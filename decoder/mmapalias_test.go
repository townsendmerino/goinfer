package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// MmapAliasWindow hands Metal the page-aligned window it needs to wrap a weight array without a copy.
// A window that is off by a page either faults (base below the mapping), wires pages the tensor never
// reads, or — the case that would silently corrupt — drops the tensor's last bytes (length rounded
// DOWN). These pin the arithmetic against a REAL mapping at the OS page size.
func TestMmapAliasWindow(t *testing.T) {
	page := os.Getpagesize()
	path := filepath.Join(t.TempDir(), "alias.bin")
	// 3 pages + 100 bytes, so the last page is partial (the case NewBufferNoCopy's length rule bites).
	if err := os.WriteFile(path, make([]byte, 3*page+100), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mmap.Unmap(data)
	m := &Model{mmap: data}
	base := uintptr(unsafe.Pointer(&data[0]))

	for _, tc := range []struct {
		name             string
		start, n         int
		wantOff, wantLen int
		wantOK           bool
	}{
		{"inside one page", 64, 128, 64, page, true},
		{"starts on a page boundary", page, 64, 0, page, true},
		{"ends exactly on a page boundary", page - 64, 64, page - 64, page, true},
		{"straddles a boundary by one byte", page - 1, 2, page - 1, 2 * page, true},
		{"spans pages", 100, 2*page + 5, 100, 3 * page, true},
		{"tail in the partial last page", 3*page + 10, 50, 10, page, true},
		{"runs off the mapping's last page", 3*page + 10, page, 0, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := data[tc.start:min(tc.start+tc.n, len(data))]
			if tc.start+tc.n > len(data) { // the 'off the end' case: forge a slice longer than the file
				b = unsafe.Slice(&data[tc.start], tc.n)
			}
			p, n, off, ok := m.MmapAliasWindow(b, page)
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if uintptr(p)%uintptr(page) != 0 || n%page != 0 {
				t.Errorf("window not page-aligned: base %#x len %d", uintptr(p), n)
			}
			if off != tc.wantOff || n != tc.wantLen {
				t.Errorf("off,len = %d,%d want %d,%d", off, n, tc.wantOff, tc.wantLen)
			}
			// The window must CONTAIN b exactly: b's first byte sits at base+off, its last byte inside.
			if got := uintptr(unsafe.Pointer(&b[0])); got != uintptr(p)+uintptr(off) {
				t.Errorf("b is not at base+off (%#x vs %#x)", got, uintptr(p)+uintptr(off))
			}
			if off+len(b) > n {
				t.Errorf("window ends before b does: off %d + len %d > %d", off, len(b), n)
			}
			if uintptr(p) < base {
				t.Errorf("window starts below the mapping")
			}
		})
	}

	// A heap slice, a nil mapping and a bad page size are all refused.
	if _, _, _, ok := m.MmapAliasWindow(make([]byte, 64), page); ok {
		t.Error("a heap-backed slice must not be given an alias window")
	}
	if _, _, _, ok := (&Model{}).MmapAliasWindow(data[:8], page); ok {
		t.Error("a model with no mapping must not give one")
	}
	if _, _, _, ok := m.MmapAliasWindow(data[:8], 3000); ok {
		t.Error("a non-power-of-two page size must be refused")
	}
}
