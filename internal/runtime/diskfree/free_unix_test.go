//go:build darwin || linux

package diskfree

import (
	"path/filepath"
	"testing"
)

// Every caller stubs Free in its own tests, so this is what fails if the real probe stops reading: an unknown reading
// makes both guards proceed, silently.
func TestFree_readsARealFilesystem(t *testing.T) {
	free, ok := Free(t.TempDir())
	if !ok || free <= 0 {
		t.Errorf("Free(a temp dir) = %d, %v; want some space, known", free, ok)
	}
}

func TestFree_aMissingPathIsUnknown(t *testing.T) {
	if free, ok := Free(filepath.Join(t.TempDir(), "missing")); ok {
		t.Errorf("Free(a missing path) = %d, known; want unknown", free)
	}
}
