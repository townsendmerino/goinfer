package decoder

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// Load must hand the WHOLE .giw mapping to protectGIWMapping (forkinherit_darwin.go). Tested through
// Load, not by calling excludeFromFork directly: the fix is only as good as the one call site that
// every .giw load passes through, and a protector applied to the weights blob (a sub-slice) instead of
// the mapping would leave the bundle header and tokenizer pages — and, on darwin, the entry split
// they cause — outside it.
func TestLoad_protectsTheWholeGIWMappingFromFork(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	blob, err := SerializeWeights(w, "forkinherit")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "forkinherit.giw")
	if err := os.WriteFile(path, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	var calls int
	var got []byte
	prev := protectGIWMapping
	protectGIWMapping = func(b []byte) error { calls++; got = b; return prev(b) }
	t.Cleanup(func() { protectGIWMapping = prev })

	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if calls != 1 {
		t.Fatalf("protectGIWMapping called %d times, want 1", calls)
	}
	if len(m.mmap) == 0 || len(got) != len(m.mmap) || unsafe.Pointer(&got[0]) != unsafe.Pointer(&m.mmap[0]) {
		t.Fatalf("protected %d bytes at %p, the model's mapping is %d bytes at %p — must be the whole mapping",
			len(got), unsafe.Pointer(&got[0]), len(m.mmap), unsafe.Pointer(&m.mmap[0]))
	}
	if st, _ := os.Stat(path); int64(len(got)) != st.Size() {
		t.Errorf("protected %d bytes, file is %d", len(got), st.Size())
	}
}

// excludeFromFork must succeed on a real read-only file mapping (the shape Load gives it) and on an empty
// slice, and — on darwin — refuse a range that is not mapped, so a silently-ignored bad call cannot pass.
func TestExcludeFromFork_realMapping(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.bin")
	if err := os.WriteFile(path, make([]byte, 3*os.Getpagesize()+17), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mmap.Unmap(data)
	if err := excludeFromFork(data); err != nil {
		t.Fatalf("excludeFromFork on a PROT_READ|MAP_PRIVATE mapping: %v", err)
	}
	if err := excludeFromFork(nil); err != nil {
		t.Fatalf("excludeFromFork(nil): %v", err)
	}
}
