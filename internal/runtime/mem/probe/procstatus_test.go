package probe

import (
	"os"
	"runtime"
	"testing"
)

func TestParseProcMemory(t *testing.T) {
	got := parseProcMemory("Name:\tcuda.test\nVmHWM:\t 38000000 kB\nVmRSS:\t  1024 kB\nVmSwap:\t     149 kB\nThreads:\t12\n")
	if want := (ProcMemory{HWMKB: 38000000, RSSKB: 1024, SwapKB: 149}); got != want {
		t.Errorf("parseProcMemory = %+v, want %+v", got, want)
	}
	if got := parseProcMemory("Name:\tx\nVmHWM:\t12 MB\n"); got != (ProcMemory{}) {
		t.Errorf("a field in another unit was read: %+v", got)
	}
}

func TestReadProcMemory_self(t *testing.T) {
	m, ok := ReadProcMemory(os.Getpid())
	if runtime.GOOS != "linux" {
		if ok {
			t.Errorf("read %+v where there is no /proc", m)
		}
		return
	}
	if !ok || m.HWMKB <= 0 || m.RSSKB <= 0 || m.HWMKB < m.RSSKB {
		t.Errorf("ReadProcMemory(self) = %+v, %v", m, ok)
	}
}
