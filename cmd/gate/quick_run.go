package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// `gate quick`'s runner: lint steps and test cells share one bounded pool. Every test cell is ONE
// `go test` invocation for ONE package — B19's rule, because `go test -v ./a/ ./b/` prints nothing
// for ./b/ until ./a/ finishes — each tee'd to its own log as it streams, so a slow cell and a hung
// one can be told apart from the log alone.

type quickJob struct {
	Name   string
	Kind   string // "lint" | "test"
	Dir    string
	Env    []string
	Args   []string // argv, program first
	Device bool
	Mem    bool // loads a real checkpoint: at most one such job at once (quickCell.Mem)
	Cell   *quickCell
	// Filter, for a lint step, is a `go list` that drops packages with no files for the step's
	// tags and target: vet given an explicit package the target excludes fails with "build
	// constraints exclude all Go files" rather than skipping it.
	Filter []string
	Canary string // non-empty: this job passes only if its output contains this text
	// QuietOK: the tool exits 0 either way and reports by printing (gofmt -l), so any output is red.
	QuietOK bool
	Mirrors string
	Detail  string // e.g. "3 pkgs"

	expected float64 // seconds, from the last run (scheduling order and the ETA)

	start, end time.Time
	RC         int
	Err        error
	Out        string // lint: the combined output
	LogPath    string
	JSONPath   string

	res                    *results
	Pass, Skip, Fail       int
	Cached, Hidden, NoTest bool
}

func (jb *quickJob) dur() time.Duration { return jb.end.Sub(jb.start) }

// ok is the job's own verdict.
func (jb *quickJob) ok() bool {
	if jb.Kind == "lint" {
		if jb.Canary != "" {
			return strings.Contains(jb.Out, jb.Canary)
		}
		return jb.RC == 0 && jb.Err == nil && (!jb.QuietOK || strings.TrimSpace(jb.Out) == "")
	}
	return jb.RC == 0 && jb.Err == nil && jb.Fail == 0 && !jb.Hidden
}

// runJobs runs every job, at most j at once and at most one device job at once, and a
// checkpoint-loading job ALONE (nothing else running beside it), in the order
// given: lint first (short, and a vet red is worth seeing before a five-minute cell finishes), then
// test cells longest-expected first, so the long pole starts before the short ones pile up.
func runJobs(jobs []*quickJob, j int, rundir string, progress io.Writer, beat time.Duration, t0 time.Time) {
	var mu sync.Mutex
	cond := sync.NewCond(&mu)
	pending := append([]*quickJob(nil), jobs...)
	running := map[*quickJob]bool{}
	devBusy, memBusy := false, false
	// Cells of ONE package never overlap. `go test` runs a package in one process, so its tests
	// were written for that; two processes of the same package at once could share a port, a
	// fixture sidecar or a scratch path, and a red caused by the split is a red about nothing.
	busyRoot := map[*testRoot]bool{}
	rootOf := func(jb *quickJob) *testRoot {
		if jb.Cell == nil {
			return nil
		}
		return jb.Cell.Root
	}
	done := 0

	stopBeat := make(chan struct{})
	beatDone := make(chan struct{})
	go func() {
		defer close(beatDone)
		if beat <= 0 {
			<-stopBeat
			return
		}
		t := time.NewTicker(beat)
		defer t.Stop()
		for {
			select {
			case <-stopBeat:
				return
			case <-t.C:
				mu.Lock()
				line := heartbeatLine(t0, done, len(jobs), running, pending, j)
				mu.Unlock()
				fmt.Fprintln(progress, line)
			}
		}
	}()

	mu.Lock()
	for len(pending) > 0 || len(running) > 0 {
		started := false
		if len(running) < j {
			for i, jb := range pending {
				// A checkpoint loader runs alone: with only the other LOADERS held back, a real-checkpoint f32 load still
				// met too little free memory beside the decoder and metal compiles and was refused by the fit guard; alone
				// it has room.
				if memBusy || (jb.Mem && len(running) > 0) || (jb.Device && devBusy) || (rootOf(jb) != nil && busyRoot[rootOf(jb)]) {
					continue
				}
				pending = append(pending[:i], pending[i+1:]...)
				running[jb] = true
				if jb.Device {
					devBusy = true
				}
				if jb.Mem {
					memBusy = true
				}
				if r := rootOf(jb); r != nil {
					busyRoot[r] = true
				}
				started = true
				go func(jb *quickJob) {
					jb.run(rundir)
					mu.Lock()
					delete(running, jb)
					if jb.Device {
						devBusy = false
					}
					if jb.Mem {
						memBusy = false
					}
					if r := rootOf(jb); r != nil {
						delete(busyRoot, r)
					}
					done++
					fmt.Fprintln(progress, finishLine(jb, done, len(jobs)))
					cond.Broadcast()
					mu.Unlock()
				}(jb)
				break
			}
		}
		if !started {
			cond.Wait()
		}
	}
	mu.Unlock()
	close(stopBeat)
	<-beatDone
}

// heartbeatLine is the progress line (~/.claude/rules/long-tests.md): the START time, not just a
// duration, so a window scrolled back to later still answers "when did this begin"; what is running
// and for how long; done of total; and an expected finish ONLY when every remaining job has a
// recorded duration — an estimate built on guesses is worse than none, because it gets planned
// around.
func heartbeatLine(t0 time.Time, done, total int, running map[*quickJob]bool, pending []*quickJob, j int) string {
	now := time.Now()
	var names []string
	for jb := range running {
		names = append(names, fmt.Sprintf("%s %s", jb.Name, now.Sub(jb.start).Round(time.Second)))
	}
	sort.Strings(names)
	line := fmt.Sprintf("   ... quick: %s elapsed (started %s), %d/%d jobs done; running: %s",
		now.Sub(t0).Round(time.Second), t0.Format("15:04:05"), done, total, strings.Join(names, ", "))
	known, longest, sum := true, 0.0, 0.0
	for jb := range running {
		if jb.expected <= 0 {
			known = false
		}
		rem := jb.expected - now.Sub(jb.start).Seconds()
		if rem < 0 {
			rem = 0
		}
		longest = max(longest, rem)
		sum += rem
	}
	for _, jb := range pending {
		if jb.expected <= 0 {
			known = false
		}
		sum += jb.expected
	}
	if known && total > done {
		eta := max(longest, sum/float64(max(j, 1)))
		line += fmt.Sprintf("; expect done ~%s (from the last run's per-cell times)",
			now.Add(time.Duration(eta*float64(time.Second))).Format("15:04:05"))
	}
	return line
}

func finishLine(jb *quickJob, done, total int) string {
	st := statusWord(jb)
	extra := ""
	if jb.Kind == "test" {
		extra = fmt.Sprintf("  %d pass  %d skip  %d fail", jb.Pass, jb.Skip, jb.Fail)
	}
	return fmt.Sprintf("   [%d/%d] %-7s %-34s %8s%s", done, total, st, jb.Name, jb.dur().Round(100*time.Millisecond), extra)
}

func statusWord(jb *quickJob) string {
	switch {
	case !jb.ok():
		return "FAIL"
	case jb.Kind == "test" && jb.Cached:
		return "CACHED"
	case jb.Kind == "test" && jb.NoTest:
		return "NOTESTS"
	case jb.Kind == "test" && jb.Pass == 0 && jb.Skip > 0:
		return "ALLSKIP"
	}
	return "PASS"
}

func (jb *quickJob) run(rundir string) {
	jb.start = time.Now()
	defer func() { jb.end = time.Now() }()
	base := filepath.Join(rundir, sanitize(jb.Kind+"_"+jb.Name))
	jb.LogPath = base + ".log"
	if jb.Err != nil { // could not even be constructed (no staticcheck, no workspace file): a red
		jb.RC, jb.Out = -1, jb.Err.Error()+"\n"
		_ = os.WriteFile(jb.LogPath, []byte(jb.Out), 0o644)
		return
	}
	if jb.Kind == "test" {
		jb.runTest(base)
		return
	}
	jb.runLint()
	_ = os.WriteFile(jb.LogPath, []byte(strings.Join(jb.Args, " ")+"\n\n"+jb.Out), 0o644)
}

func (jb *quickJob) runLint() {
	args := jb.Args
	if len(jb.Filter) > 0 {
		cmd := exec.Command(jb.Filter[0], jb.Filter[1:]...)
		cmd.Dir, cmd.Env = jb.Dir, jb.Env
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			jb.RC, jb.Err, jb.Out = exitCode(err), err, stderr.String()
			return
		}
		keep := strings.Fields(string(out))
		if len(keep) == 0 {
			jb.Out = "(no selected package has files for these tags and target — nothing to check)\n"
			return
		}
		jb.Detail = fmt.Sprintf("%d pkgs", len(keep))
		args = append(append([]string(nil), args...), keep...)
		jb.Args = args
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir, cmd.Env = jb.Dir, jb.Env
	out, err := cmd.CombinedOutput()
	jb.Out = string(out)
	if err != nil {
		if _, isExit := err.(*exec.ExitError); !isExit {
			jb.Err = err // could not start: a red, never a silent skip
		}
		jb.RC = exitCode(err)
	}
}

func exitCode(err error) int {
	if err == nil {
		return 0
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	return -1
}

// runTest runs one cell: the raw -json stream to <base>.json, the reconstructed `go test -v` text
// (and go's stderr) to <base>.log AS IT ARRIVES, and every event into the cell's own results.
func (jb *quickJob) runTest(base string) {
	jb.JSONPath = base + ".json"
	jb.res = newResults()
	jb.res.cur = jb.Name
	jf, err := os.Create(jb.JSONPath)
	if err != nil {
		jb.Err, jb.RC = err, -1
		return
	}
	defer jf.Close()
	lf, err := os.Create(jb.LogPath)
	if err != nil {
		jb.Err, jb.RC = err, -1
		return
	}
	defer lf.Close()
	fmt.Fprintf(lf, "# %s\n# (in %s)\n\n", strings.Join(jb.Args, " "), jb.Dir)

	cmd := exec.Command(jb.Args[0], jb.Args[1:]...)
	cmd.Dir, cmd.Env = jb.Dir, jb.Env
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		jb.Err, jb.RC = err, -1
		return
	}
	cmd.Stderr = lf
	if err := cmd.Start(); err != nil {
		jb.Err, jb.RC = err, -1
		return
	}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 256*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		jf.Write(line)
		jf.Write([]byte{'\n'})
		if len(line) == 0 || line[0] != '{' {
			lf.Write(line)
			lf.Write([]byte{'\n'})
			continue
		}
		var ev testEvent
		if json.Unmarshal(line, &ev) != nil {
			continue
		}
		if ev.Output != "" {
			lf.WriteString(ev.Output)
		}
		jb.res.add(ev)
	}
	// Drain whatever a scan error left, so the child never blocks on a full pipe.
	_, _ = io.Copy(io.Discard, stdout)
	jb.RC = exitCode(cmd.Wait())

	for k, act := range jb.res.final {
		if isSubtest(k) {
			continue
		}
		switch act {
		case "pass":
			jb.Pass++
		case "skip":
			jb.Skip++
		case "fail":
			jb.Fail++
		}
	}
	for _, lines := range jb.res.pkgOut {
		for _, ln := range lines {
			if strings.HasPrefix(ln, "ok ") && strings.Contains(ln, "(cached)") {
				jb.Cached = true
			}
		}
	}
	// rc-aware, like the heavy tier: a build error, a panic in a goroutine, a timeout or a fatal
	// error exits non-zero WITHOUT a per-test FAIL line, and counting only FAIL lines would call a
	// package that never ran green.
	jb.Hidden = jb.RC != 0 && jb.Fail == 0
	jb.NoTest = jb.RC == 0 && jb.Pass+jb.Skip+jb.Fail == 0
}

// ---- duration history ----

func loadDurations(stateDir string) map[string]float64 {
	d := map[string]float64{}
	if b, err := os.ReadFile(filepath.Join(stateDir, "durations.json")); err == nil {
		_ = json.Unmarshal(b, &d)
	}
	return d
}

// saveDurations records the cells that actually RAN. A cached replay's two seconds says nothing
// about what the cell costs when it re-runs, which is the number the schedule and the ETA need.
func saveDurations(stateDir string, d map[string]float64, jobs []*quickJob) {
	for _, jb := range jobs {
		if jb.Kind == "test" && !jb.Cached && jb.Err == nil && !jb.Hidden {
			d[jb.Name] = jb.dur().Seconds()
		}
	}
	if b, err := json.MarshalIndent(d, "", "  "); err == nil {
		_ = os.MkdirAll(stateDir, 0o755)
		_ = os.WriteFile(filepath.Join(stateDir, "durations.json"), b, 0o644)
	}
}
