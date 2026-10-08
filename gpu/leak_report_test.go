//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Leak report. A device buffer a test never closes keeps a reference on its device after the Context is closed, so the garbage collector later destroys that device on the finalizer goroutine at an
// arbitrary moment (the webgpu-parity hang; gpu.go finalizerSerial). The live-buffer gauge (bufaccount.go) is exact and the tests run one at a time, so a Context that is closed while the gauge is
// above its value at New had something outlive it. Close calls leakAtClose on the test's goroutine, where the stack names the test.
//
// The report is printed at the end of the run. GOINFER_GPU_LEAKS_FATAL=1 makes any entry fail the package (a test-only switch, not a production read).
// leakExempt is set by the two reproducers, which leak a matrix per round on purpose.
var leakExempt atomic.Bool

var (
	leakMu   sync.Mutex
	leakList []leakEntry
)

func init() {
	leakAtClose = func(c *Context) {
		n := liveBufferBytes.Load() - c.liveBase
		if n <= 0 || leakExempt.Load() {
			return
		}
		name := ""
		var near []string // the nearest frames, for a Close that no Test function is on the stack of (a helper's goroutine, a Cleanup)
		pcs := make([]uintptr, 32)
		frames := runtime.CallersFrames(pcs[:runtime.Callers(2, pcs)])
		for {
			fr, more := frames.Next()
			if i := strings.LastIndex(fr.Function, ".Test"); i >= 0 {
				name = fr.Function[i+1:]
				break
			}
			if len(near) < 4 {
				near = append(near, fr.Function[strings.LastIndex(fr.Function, "/")+1:])
			}
			if !more {
				break
			}
		}
		if name == "" {
			name = "(no Test frame; nearest: " + strings.Join(near, " < ") + ")"
		}
		leakMu.Lock()
		leakList = append(leakList, leakEntry{name, n})
		leakMu.Unlock()
	}
}

type leakEntry struct {
	name  string
	bytes int64
}

func TestMain(m *testing.M) {
	code := m.Run()
	leakMu.Lock()
	if len(leakList) > 0 {
		type agg struct {
			n        int
			min, max int64
		}
		by := map[string]*agg{}
		var order []string
		for _, e := range leakList {
			a := by[e.name]
			if a == nil {
				a = &agg{min: e.bytes, max: e.bytes}
				by[e.name] = a
				order = append(order, e.name)
			}
			a.n++
			a.min, a.max = min(a.min, e.bytes), max(a.max, e.bytes)
		}
		fmt.Fprintf(os.Stderr, "\ngpu: LEAK REPORT: %d Context(s) closed with caller-owned device buffers still live, %d distinct callers:\n", len(leakList), len(order))
		for _, name := range order {
			a := by[name]
			fmt.Fprintf(os.Stderr, "  %4d x  %d..%d bytes  %s\n", a.n, a.min, a.max, name)
		}
		if os.Getenv("GOINFER_GPU_LEAKS_FATAL") == "1" && code == 0 {
			code = 1
		}
	}
	leakMu.Unlock()
	os.Exit(code)
}
