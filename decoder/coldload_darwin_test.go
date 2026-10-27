//go:build darwin

package decoder

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// TestColdLoad_darwinProbesAndActions runs the real kernel calls on a real file: a file written and dropped from
// memory samples as not resident, and giwReadThrough (MADV_WILLNEED) makes it resident. The read-ahead may complete
// after the call returns, so the test waits for it, briefly. A filesystem that keeps the file resident after the drop
// cannot show the first step, and the test says so and skips.
func TestColdLoad_darwinProbesAndActions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cold.bin")
	const size = 64 << 20
	buf := make([]byte, size)
	for i := range buf {
		buf[i] = byte(i * 31)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	mapping, err := mapGIW(path)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mapping)
	const msInvalidate = 0x2
	if _, _, errno := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&mapping[0])), uintptr(len(mapping)), msInvalidate); errno != 0 {
		t.Skipf("msync(MS_INVALIDATE): %v", errno)
	}
	cold := probe.ResidentFraction(mapping)
	if cold >= coldResidentWarm {
		t.Skipf("the file still samples %.0f%% resident after the drop: this filesystem cannot show a cold file", 100*cold)
	}
	giwAdviseReadAround(mapping, false)
	giwAdviseReadAround(mapping, true)
	giwReadThrough(mapping)
	warm, deadline := 0.0, time.Now().Add(10*time.Second)
	for {
		if warm = probe.ResidentFraction(mapping); warm >= 0.99 || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if warm < 0.99 {
		t.Errorf("ten seconds after the prefetch the mapping samples %.0f%% resident (it sampled %.0f%% before), want all of it", 100*warm, 100*cold)
	}
	for i := 0; i < size; i += 4099 {
		if mapping[i] != buf[i] {
			t.Fatalf("byte %d of the mapping is %d, the file holds %d", i, mapping[i], buf[i])
		}
	}
}
