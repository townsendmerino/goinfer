package gatecore

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/mem/probe"
)

// ProcPeak records a test binary's VmHWM, peak VmSwap, and per-test VmRSS upper bounds sampled from /proc.
type ProcPeak struct {
	PID        int
	HWMKB      int64
	SwapPeakKB int64
	Samples    int
	ByTest     map[string]int64
}

var procSamplePeriod = 500 * time.Millisecond

// SetProcSamplePeriod overrides the /proc sampling interval and returns a restore closure.
func SetProcSamplePeriod(d time.Duration) (restore func()) {
	old := procSamplePeriod
	procSamplePeriod = d
	return func() { procSamplePeriod = old }
}

// StartProcSampler samples the descendant of root whose name ends in ".test" until stop is called.
func StartProcSampler(root int, inFlight func() (string, time.Duration, bool)) (stop func() ProcPeak) {
	var mu sync.Mutex
	p := ProcPeak{ByTest: map[string]int64{}}
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
	return func() ProcPeak {
		close(done)
		<-finished
		mu.Lock()
		defer mu.Unlock()
		return p
	}
}

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
		name, ppid, ok := ParseProcStat(string(b))
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
		slices.Sort(cs)
		for _, c := range cs {
			if strings.HasSuffix(names[c], ".test") {
				return c
			}
			todo = append(todo, c)
		}
	}
	return 0
}

// ParseProcStat extracts the comm field and parent PID from a /proc/<pid>/stat line.
func ParseProcStat(s string) (name string, ppid int, ok bool) {
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
