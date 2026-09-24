//go:build darwin

package decoder

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/mmap"
)

// The syscall number and argument order excludeFromFork relies on are real: the same call with an
// invalid inheritance value must come back EINVAL from the kernel's minherit (a wrong syscall number
// would give ENOSYS or act on something else). XNU accepts minherit over an unmapped hole silently, so
// "does it fail on a bad range" is not a usable check; the effect itself is measured by
// metal/alias_forkprobe_test.go (fork 1,092 ms -> 3.8 ms while a page is wired).
func TestExcludeFromFork_syscallReachesMinherit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.bin")
	if err := os.WriteFile(path, make([]byte, 2*os.Getpagesize()), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mmap.Unmap(data)
	const bogusInheritance = 99
	_, _, e := syscall.Syscall(syscall.SYS_MINHERIT, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), bogusInheritance)
	if e != syscall.EINVAL {
		t.Fatalf("minherit(inheritance=%d) = %v, want EINVAL — SYS_MINHERIT is not the minherit(2) excludeFromFork thinks it is", bogusInheritance, e)
	}
	if err := excludeFromFork(data); err != nil {
		t.Fatalf("excludeFromFork: %v", err)
	}
}
