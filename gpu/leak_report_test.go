//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Leak report. A device buffer a test never closes keeps a reference on its device after the Context is closed, so the garbage collector later destroys that device on the finalizer goroutine at an
// arbitrary moment (the webgpu-parity hang; gpu.go finalizerSerial). The live-buffer gauge (bufaccount.go) is exact and the tests run one at a time, so a Context that is closed while the gauge is
// above its value at New had something outlive it. Close calls leakAtClose on the test's goroutine, where the stack names the test.
//
// The report is printed at the end of the run and any entry FAILS the package (the suite was made leak-free on 2026-10-08, so a new leak is a regression); GOINFER_GPU_LEAKS_FATAL=0 turns the failure off (a test-only switch,
// not a production read).
// leakExempt is set by the two reproducers, which leak a matrix per round on purpose.
var leakExempt atomic.Bool

var (
	leakMu   sync.Mutex
	leakList []leakEntry
)

// liveRec is the creation record of one counted wrapper still live: where in a test file it was made, and in what order.
type liveRec struct {
	seq   int64
	bytes int64
	site  string // file:line of the outermost *_test.go frame
}

var (
	traceMu   sync.Mutex
	traceLive = map[uintptr]liveRec{} // keyed by address, never by the pointer: holding the wrapper would keep a leaked one (and its device) alive forever
)

func init() {
	allocTrace = func(key any, n int64, alloc bool) {
		traceMu.Lock()
		defer traceMu.Unlock()
		k := reflect.ValueOf(key).Pointer()
		if !alloc {
			delete(traceLive, k)
			return
		}
		site := "?"
		via := "" // the library function that made the wrapper: the first non-test frame that is not the accounting itself
		pcs := make([]uintptr, 24)
		frames := runtime.CallersFrames(pcs[:runtime.Callers(3, pcs)])
		for {
			fr, more := frames.Next()
			if strings.HasSuffix(fr.File, "_test.go") {
				site = fmt.Sprintf("%s:%d", fr.File[strings.LastIndex(fr.File, "/")+1:], fr.Line)
				break
			}
			if via == "" && !strings.Contains(fr.Function, "newDeviceBuffer") {
				via = fr.Function[strings.LastIndex(fr.Function, "/")+1:]
			}
			if !more {
				break
			}
		}
		if via != "" {
			site += " via " + via
		}
		traceLive[k] = liveRec{seq: allocSeq.Load(), bytes: n, site: site}
	}
}

// survivorSites summarizes the live wrappers created since a Context was created: "file:line xN (bytes)", largest first.
func survivorSites(since int64) string {
	traceMu.Lock()
	defer traceMu.Unlock()
	type agg struct {
		n     int
		bytes int64
	}
	by := map[string]*agg{}
	for _, r := range traceLive {
		if r.seq <= since {
			continue
		}
		a := by[r.site]
		if a == nil {
			a = &agg{}
			by[r.site] = a
		}
		a.n++
		a.bytes += r.bytes
	}
	var keys []string
	for k := range by {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return by[keys[i]].bytes > by[keys[j]].bytes })
	var out []string
	for _, k := range keys {
		out = append(out, fmt.Sprintf("%s x%d (%d B)", k, by[k].n, by[k].bytes))
	}
	return strings.Join(out, "; ")
}

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
		leakList = append(leakList, leakEntry{name, n, survivorSites(c.allocSeq)})
		leakMu.Unlock()
	}
}

type leakEntry struct {
	name  string
	bytes int64
	sites string // where the survivors were created
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
		sitesOf := map[string]string{}
		var order []string
		for _, e := range leakList {
			sitesOf[e.name] = e.sites
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
			fmt.Fprintf(os.Stderr, "           created at: %s\n", sitesOf[name])
		}
		if os.Getenv("GOINFER_GPU_LEAKS_FATAL") != "0" && code == 0 {
			code = 1
		}
	}
	leakMu.Unlock()
	os.Exit(code)
}
