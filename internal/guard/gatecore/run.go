package gatecore

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Cell describes one `go test` invocation in a gate matrix.
type Cell struct {
	Name    string
	Pkgs    []string
	Tags    []string
	Run     string
	Env     map[string]string
	Timeout string
	Serial  bool
	Extra   []string
	Dir     string

	// OnStart, when set, is called with the `go` process PID once started and stopped after Wait.
	OnStart func(pid int) (stop func())
}

// GateConfig configures a matrix gate and its decision rules.
type GateConfig struct {
	Name             string
	Desc             string
	Cells            []Cell
	Decision         string // "tally" or "census"
	TopLevelOnly     bool
	ZeroPolicy       string // "no-pass" or "no-tests"
	RCIsFailure      bool
	PkgFailIsFailure bool
	Precondition     func() (why string, ok bool)
}

// CellResult holds one cell's tally, exit code, log path, and elapsed duration.
type CellResult struct {
	Cell             Cell
	RC               int
	Err              error
	LogPath          string
	Pass, Skip, Fail int
	RunLines         int
	Hidden           bool
	Dur              time.Duration
}

var cellHeartbeatInterval = 60 * time.Second

// SetCellHeartbeatInterval overrides the default cell heartbeat period and returns a restore closure.
func SetCellHeartbeatInterval(d time.Duration) (restore func()) {
	old := cellHeartbeatInterval
	cellHeartbeatInterval = d
	return func() { cellHeartbeatInterval = old }
}

// StartCellHeartbeat emits periodic progress lines to stderr until stop is called; stop joins the goroutine.
func StartCellHeartbeat(cell string, res *Results, t0 time.Time) (stop func()) {
	every := cellHeartbeatInterval
	if v := os.Getenv("GOINFER_GATE_HEARTBEAT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			every = d
		}
	}
	if every <= 0 {
		return func() {}
	}
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				last, _ := res.LiveLast.Load().(string)
				if last == "" {
					last = "(no test has finished yet)"
				}
				line := fmt.Sprintf("   ... %s: %s elapsed, %d tests finished, last: %s",
					cell, time.Since(t0).Round(time.Second), res.LiveDone.Load(), last)
				if name, since, ok := res.InFlight(); ok {
					line += fmt.Sprintf(" | in flight: %s (%s)", name, since)
				}
				fmt.Fprintln(os.Stderr, line)
			}
		}
	}()
	return func() { close(done); <-finished }
}

// RunCell executes one cell, tees its raw stream to logDir, and folds its events into res.
func RunCell(c Cell, cfg *GateConfig, res *Results, logDir string) CellResult {
	args := []string{"test", "-json", "-count=1"}
	if len(c.Tags) > 0 {
		args = append(args, "-tags", strings.Join(c.Tags, " "))
	}
	if c.Run != "" {
		args = append(args, "-run", c.Run)
	}
	if c.Timeout != "" {
		args = append(args, "-timeout", c.Timeout)
	}
	if c.Serial {
		args = append(args, "-p", "1")
	}
	args = append(args, c.Extra...)
	args = append(args, c.Pkgs...)

	if _, err := exec.LookPath("go"); err != nil {
		return GradeCellError(c, -1, fmt.Errorf("go toolchain not on PATH: %w", err), "", 0)
	}

	logPath := filepath.Join(logDir, "gate_"+Sanitize(cfg.Name+"_"+c.Name)+".json")
	logf, err := os.Create(logPath)
	if err != nil {
		return GradeCellError(c, -1, err, "", 0)
	}
	defer logf.Close()

	cmd := exec.Command("go", args...)
	cmd.Dir = c.Dir
	cmd.Env = CellEnv(os.Environ(), c.Env)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return GradeCellError(c, -1, err, "", 0)
	}
	cmd.Stderr = logf

	res.Cur = c.Name
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		return GradeCellError(c, -1, err, "", 0)
	}
	stopBeat := StartCellHeartbeat(c.Name, res, t0)
	stopWatch := func() {}
	if c.OnStart != nil {
		stopWatch = c.OnStart(cmd.Process.Pid)
	}
	res.Consume(io.TeeReader(stdout, logf))
	stopBeat()
	waitErr := cmd.Wait()
	dur := time.Since(t0)
	stopWatch()

	rc := 0
	if waitErr != nil {
		if ee, ok := errors.AsType[*exec.ExitError](waitErr); ok {
			rc = ee.ExitCode()
		} else {
			rc = -1
		}
	}

	return GradeCell(c, cfg, res, rc, logPath, dur)
}

// Sanitize turns arbitrary cell or package labels into safe file-name stems.
func Sanitize(s string) string {
	rep := strings.NewReplacer("/", "_", ".", "_", " ", "_", ":", "_")
	return strings.Trim(rep.Replace(s), "_")
}

// CellEnv returns base without ambient GOINFER_MEASURE_TESTS, overlaid with set.
func CellEnv(base []string, set map[string]string) []string {
	env := make([]string, 0, len(base)+len(set))
	for _, kv := range base {
		if !strings.HasPrefix(kv, "GOINFER_MEASURE_TESTS=") {
			env = append(env, kv)
		}
	}
	for k, v := range set {
		env = append(env, k+"="+v)
	}
	return env
}
