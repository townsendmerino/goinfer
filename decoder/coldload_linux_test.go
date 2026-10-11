//go:build linux

package decoder

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/townsendmerino/aikit/mmap"
)

// TestColdLoad_linuxProbesAndActions runs the real kernel calls on a real file: a file written, synced and dropped
// from the page cache samples as not resident; reading it through makes it resident; and turning the read-around off
// and on is accepted. A filesystem that keeps the file resident after the drop cannot show the first step, and the
// test says so and skips.
func TestColdLoad_linuxProbesAndActions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cold.bin")
	const size = 64 << 20
	buf := make([]byte, size)
	for i := range buf {
		buf[i] = byte(i * 31)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	const fadvDontNeed = 4 // POSIX_FADV_DONTNEED
	if _, _, errno := syscall.Syscall6(syscall.SYS_FADVISE64, f.Fd(), 0, 0, fadvDontNeed, 0, 0); errno != 0 {
		t.Skipf("posix_fadvise(DONTNEED): %v", errno)
	}
	mapping, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mmap.Unmap(mapping)

	cold := giwResidentFraction(mapping)
	if cold >= coldResidentWarm {
		t.Skipf("the file still samples %.0f%% resident after the drop: this filesystem cannot show a cold file", 100*cold)
	}
	giwAdviseReadAround(mapping, false)
	giwAdviseReadAround(mapping, true)
	giwReadThrough(mapping)
	if warm := giwResidentFraction(mapping); warm < 0.99 {
		t.Errorf("after reading the mapping through it samples %.0f%% resident (it sampled %.0f%% before), want all of it", 100*warm, 100*cold)
	}
	for i := 0; i < size; i += 4099 {
		if mapping[i] != buf[i] {
			t.Fatalf("byte %d of the mapping is %d, the file holds %d", i, mapping[i], buf[i])
		}
	}
}
