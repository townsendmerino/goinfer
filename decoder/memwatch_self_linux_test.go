//go:build linux

package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// fakeProc writes one process into a temp /proc: its stat line (command name as given, so a name with spaces and parentheses is exercised)
// and a status file with the given VmSwap in kB (a negative value leaves the line out, as a kernel without swap accounting does).
func fakeProc(t *testing.T, root string, pid, ppid int, comm string, vmSwapKB int) {
	t.Helper()
	dir := filepath.Join(root, fmt.Sprint(pid))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	stat := fmt.Sprintf("%d (%s) S %d %d 0 0 -1 4194560 100 0 0 0 1 1 0 0 20 0 1 0 12345 1000 100 18446744073709551615\n", pid, comm, ppid, pid)
	status := fmt.Sprintf("Name:\t%s\nVmRSS:\t   100 kB\n", comm)
	if vmSwapKB >= 0 {
		status += fmt.Sprintf("VmSwap:\t%8d kB\n", vmSwapKB)
	}
	for name, body := range map[string]string{"stat": stat, "status": status} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// selfSwapBytes sums VmSwap over the pid and its descendants and nothing else: another process's swap, however large, does not move it, and a
// descendant's does.
func TestSelfSwapBytes_countsTheServerAndItsDescendantsOnly(t *testing.T) {
	root := t.TempDir()
	const MB = 1024
	fakeProc(t, root, 100, 1, "goinfer-serve", 10*MB)
	fakeProc(t, root, 101, 100, "helper", 20*MB)
	fakeProc(t, root, 102, 101, "grand child) (x", 5*MB) // a name with spaces and parentheses: the parent field is counted from the last ')'
	fakeProc(t, root, 103, 100, "no-accounting", -1)     // a descendant with no VmSwap line is skipped, not an error
	fakeProc(t, root, 200, 1, "cloudcode", 3000*MB)      // not a descendant
	fakeProc(t, root, 201, 200, "its-child", 1000*MB)    // a descendant of someone else
	if err := os.MkdirAll(filepath.Join(root, "self"), 0o755); err != nil {
		t.Fatal(err)
	}

	got, ok := selfSwapBytes(root, 100)
	if want := int64(35 * MB * 1024); !ok || got != want {
		t.Fatalf("selfSwapBytes = %d, %v; want %d (10 + 20 + 5 MB)", got, ok, want)
	}

	fakeProc(t, root, 200, 1, "cloudcode", 9000*MB) // the other process's swap grows by 6 GB
	if got2, _ := selfSwapBytes(root, 100); got2 != got {
		t.Errorf("another process's swap moved the figure: %d -> %d", got, got2)
	}
	fakeProc(t, root, 102, 101, "grand child) (x", 605*MB) // a grandchild's swap grows by 600 MB
	if got3, _ := selfSwapBytes(root, 100); got3 != got+600*MB*1024 {
		t.Errorf("a grandchild's +600 MB read as %d -> %d", got, got3)
	}
}

// With no VmSwap line for the process itself (a kernel without swap accounting) or no status at all, the figure is unknown, so the caller
// falls back to the system-wide guard instead of watching a number that is not there.
func TestSelfSwapBytes_unknownWithoutVmSwap(t *testing.T) {
	root := t.TempDir()
	fakeProc(t, root, 100, 1, "goinfer-serve", -1)
	if v, ok := selfSwapBytes(root, 100); ok {
		t.Errorf("a status with no VmSwap read as %d, true", v)
	}
	if v, ok := selfSwapBytes(root, 999); ok {
		t.Errorf("a missing process read as %d, true", v)
	}
}

// The real probe reads this test process's own status.
func TestSelfSwapBytes_realProc(t *testing.T) {
	if v, ok := SelfSwapBytes(); !ok || v < 0 {
		t.Skipf("this kernel reports no VmSwap for the test process (%d, %v)", v, ok)
	}
}
