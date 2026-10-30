package main

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// procPeak is a test binary's memory as read from /proc/<pid>/status while it ran. The never-swap rule judges a test
// process on its own VmHWM and peak VmSwap (CLAUDE.md, "Testing tiers"), and `go test` runs the binary as its child, so the
// sampler finds the binary among the descendants of the `go` process. Linux only: elsewhere Samples stays 0.
type procPeak struct {
	PID        int
	HWMKB      int64 // VmHWM at the last sample; the kernel keeps the high-water mark, so this is the run's peak to within one period
	SwapPeakKB int64 // the largest VmSwap sampled
	Samples    int
	// ByTest is, for each top-level test, the largest VmRSS sampled while it was the oldest test in flight. A test's
	// reading includes whatever earlier tests left resident, so it bounds the test's own footprint from above.
	ByTest map[string]int64
}

// procSamplePeriod is how often the sampler reads /proc. A var so a test can shorten it.
var procSamplePeriod = 500 * time.Millisecond

// startProcSampler samples the descendant of root whose name ends in ".test" until stop is called. inFlight names the
// test running at each sample (results.inFlight); nil attributes nothing. stop joins the goroutine and returns the peak.
func startProcSampler(root int, inFlight func() (string, time.Duration, bool)) (stop func() procPeak) {
	var mu sync.Mutex
	p := procPeak{ByTest: map[string]int64{}}
	done, finished := make(chan struct{}), make(chan struct{})
	sample := func() {
		if p.PID == 0 {
			p.PID = findTestBinary(root)
			if p.PID == 0 {
				return
			}
		}
		st, ok := probe.ReadProcMemory(p.PID)
		if !ok {
			return
		}
		mu.Lock()
		defer mu.Unlock()
		p.Samples++
		p.HWMKB = max(p.HWMKB, st.HWMKB)
		p.SwapPeakKB = max(p.SwapPeakKB, st.SwapKB)
		if inFlight != nil {
			if name, _, ok := inFlight(); ok {
				if top, _, _ := strings.Cut(name, "/"); top != "" {
					p.ByTest[top] = max(p.ByTest[top], st.RSSKB)
				}
			}
		}
	}
	go func() {
		defer close(finished)
		t := time.NewTicker(procSamplePeriod)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				sample()
			}
		}
	}()
	return func() procPeak {
		close(done)
		<-finished
		mu.Lock()
		defer mu.Unlock()
		return p
	}
}

// findTestBinary returns the first descendant of root whose process name ends in ".test", or 0.
func findTestBinary(root int) int {
	ents, err := os.ReadDir("/proc")
	if err != nil {
		return 0
	}
	kids := map[int][]int{}
	names := map[int]string{}
	for _, e := range ents {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(filepath.Join("/proc", e.Name(), "stat"))
		if err != nil {
			continue
		}
		name, ppid, ok := parseProcStat(string(b))
		if !ok {
			continue
		}
		kids[ppid] = append(kids[ppid], pid)
		names[pid] = name
	}
	todo := []int{root}
	for len(todo) > 0 {
		p := todo[0]
		todo = todo[1:]
		cs := kids[p]
		sort.Ints(cs)
		for _, c := range cs {
			if strings.HasSuffix(names[c], ".test") {
				return c
			}
			todo = append(todo, c)
		}
	}
	return 0
}

// parseProcStat returns the process name and parent pid from a /proc/<pid>/stat line. The name is parenthesised and can
// itself hold spaces and parentheses, so the fields after it are found from the LAST closing parenthesis.
func parseProcStat(s string) (name string, ppid int, ok bool) {
	open, closeIdx := strings.IndexByte(s, '('), strings.LastIndexByte(s, ')')
	if open < 0 || closeIdx < open {
		return "", 0, false
	}
	f := strings.Fields(s[closeIdx+1:])
	if len(f) < 2 {
		return "", 0, false
	}
	ppid, err := strconv.Atoi(f[1])
	if err != nil {
		return "", 0, false
	}
	return s[open+1 : closeIdx], ppid, true
}
