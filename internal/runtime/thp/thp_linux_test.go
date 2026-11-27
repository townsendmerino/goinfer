//go:build linux

package thp

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

// hugeKiB is this process's AnonHugePages in the mapping that holds addr (/proc/self/smaps).
func hugeKiB(t *testing.T, addr uintptr) int {
	t.Helper()
	f, err := os.Open("/proc/self/smaps")
	if err != nil {
		t.Skipf("no /proc/self/smaps: %v", err)
	}
	defer f.Close()
	in := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		var lo, hi uintptr
		if n, err := fmt.Sscanf(line, "%x-%x", &lo, &hi); err == nil && n == 2 && strings.Contains(line, " ") {
			in = addr >= lo && addr < hi
			continue
		}
		if in && strings.HasPrefix(line, "AnonHugePages:") {
			var kb int
			if _, err := fmt.Sscanf(strings.TrimPrefix(line, "AnonHugePages:"), "%d", &kb); err != nil {
				t.Fatalf("smaps %q: %v", line, err)
			}
			return kb
		}
	}
	t.Fatalf("no mapping holds %#x", addr)
	return 0
}

// touch writes one byte per page of b's capacity, so every page is faulted in.
func touch(b []byte) {
	b = b[:cap(b)]
	for i := 0; i < len(b); i += 4096 {
		b[i] = 1
	}
}

func thpAlways(t *testing.T) {
	t.Helper()
	b, err := os.ReadFile("/sys/kernel/mm/transparent_hugepage/enabled")
	if err != nil || !strings.Contains(string(b), "[always]") {
		t.Skipf("transparent huge pages are not \"always\" here (%q): nothing would back a plain buffer with them", strings.TrimSpace(string(b)))
	}
}

// TestSmall_keepsTheWindowsPagesOffHugePages: on a THP "always" host a plain 64 MiB buffer is backed by huge pages once
// touched, and one whose pages are touched inside Small's window is not; the process's THP-disable state is what it was
// before. (Pages touched after the window are not asserted: the next window clears the MADV_NOHUGEPAGE mark.)
func TestSmall_keepsTheWindowsPagesOffHugePages(t *testing.T) {
	thpAlways(t)
	const n = 64 << 20
	plain := make([]byte, 0, n)
	touch(plain)
	if hugeKiB(t, uintptr(unsafe.Pointer(unsafe.SliceData(plain)))) == 0 {
		t.Skip("a plain touched 64 MiB buffer got no huge pages here: the test cannot see a difference")
	}
	var inside []byte
	Small(func() [][]byte {
		inside = make([]byte, 0, n)
		touch(inside)
		return [][]byte{inside}
	})
	if kb := hugeKiB(t, uintptr(unsafe.Pointer(unsafe.SliceData(inside)))); kb != 0 {
		t.Errorf("a buffer touched inside the window holds %d KiB of huge pages", kb)
	}
	if st, _, _ := syscall.RawSyscall(syscall.SYS_PRCTL, prGetTHPDisable, 0, 0); st != 0 {
		t.Errorf("PR_GET_THP_DISABLE = %d after Small, want 0 (the state before it)", st)
	}
}
