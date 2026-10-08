//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// Leak report. A device buffer a test never closes keeps a reference on its device after the Context is closed, so the garbage collector later destroys that device on the finalizer goroutine at an
// arbitrary moment (the webgpu-parity hang; gpu.go finalizerSerial). The live-buffer gauge (bufaccount.go) is exact and the tests run one at a time, so a Context that is closed while the gauge is
// above its value at New had something outlive it. Close calls leakAtClose on the test's goroutine, where the stack names the test.
//
// The report is printed at the end of the run. GOINFER_GPU_LEAKS_FATAL=1 makes any entry fail the package (a test-only switch, not a production read).
var (
	leakMu   sync.Mutex
	leakList []string
)

func init() {
	leakAtClose = func(c *Context) {
		n := liveBufferBytes.Load() - c.liveBase
		if n <= 0 {
			return
		}
		name := "?"
		pcs := make([]uintptr, 32)
		frames := runtime.CallersFrames(pcs[:runtime.Callers(2, pcs)])
		for {
			fr, more := frames.Next()
			if i := strings.LastIndex(fr.Function, ".Test"); i >= 0 {
				name = fr.Function[i+1:]
				break
			}
			if !more {
				break
			}
		}
		leakMu.Lock()
		leakList = append(leakList, fmt.Sprintf("%s: %d bytes of device buffers still live at Context.Close", name, n))
		leakMu.Unlock()
	}
}

func TestMain(m *testing.M) {
	code := m.Run()
	leakMu.Lock()
	if len(leakList) > 0 {
		fmt.Fprintf(os.Stderr, "\ngpu: LEAK REPORT: %d Context(s) closed with caller-owned device buffers still live:\n", len(leakList))
		for _, l := range leakList {
			fmt.Fprintf(os.Stderr, "  %s\n", l)
		}
		if os.Getenv("GOINFER_GPU_LEAKS_FATAL") == "1" && code == 0 {
			code = 1
		}
	}
	leakMu.Unlock()
	os.Exit(code)
}
