// Command lint runs staticcheck as CI does, after proving it is analysing: the binary must be the version
// .github/actions/staticcheck pins, and the canary package (cmd/lint/canary) must produce its U1000 finding. Each target
// then runs with CI's tags, workspace and platform (GOOS=linux), so a Mac run sees what CI's will. It ends with one
// VERDICT line and exits 0 (PASS), 1 (FAIL) or 2 (INCONCLUSIVE).
//
//	go run ./cmd/lint                             # every target, staticcheck from PATH
//	go run ./cmd/lint -only cuda,gpu              # some targets; the rest print n/a
//	go run ./cmd/lint -staticcheck /path/to/bin
//
// GOOS is set for staticcheck's analysis only, never for `go run`: a GOOS=linux `go run` of staticcheck builds a binary
// the Mac cannot exec, and the shell still reports exit 0. It holds the timing lock shared, as touched.py does
// (CLAUDE.md, "Run budget").
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	bin := flag.String("staticcheck", "staticcheck", "staticcheck binary")
	onlyFlag := flag.String("only", "", "comma-separated targets to run (root, gpu, demo/agent, cuda); empty runs all")
	underLock := flag.Bool("under-timing-lock", false, "internal: set by lint's own re-run under the shared timing lock")
	flag.Parse()
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		fmt.Println(verdict(inconclusive, "lint: not in a git repository"))
		os.Exit(2)
	}
	root := strings.TrimSpace(string(out))
	var only map[string]bool
	if *onlyFlag != "" {
		only = map[string]bool{}
		for _, n := range strings.Split(*onlyFlag, ",") {
			only[strings.TrimSpace(n)] = true
		}
	}
	if !*underLock {
		if code, ran := underSharedLock(root); ran {
			os.Exit(code)
		}
	}
	os.Exit(printReport(os.Stdout, "lint", checks(root, *bin, only)).exit)
}

// underSharedLock re-runs lint under `scripts/timing_lock.py run --shared` and returns its exit code, with ran true.
// The lock script exits 75 without running lint when a timed run holds the lock; that becomes an INCONCLUSIVE verdict.
// ran is false, and the caller goes on in this process, when the tree has no lock script.
func underSharedLock(root string) (int, bool) {
	script := filepath.Join(root, "scripts", "timing_lock.py")
	if _, err := os.Stat(script); err != nil {
		return 0, false
	}
	exe, err := os.Executable()
	if err != nil {
		return 0, false
	}
	args := append([]string{script, "run", "--shared", "--label", "lint", "--", exe, "-under-timing-lock"}, os.Args[1:]...)
	cmd := exec.Command("python3", args...)
	cmd.Dir, cmd.Stdout, cmd.Stderr = root, os.Stdout, os.Stderr
	err = cmd.Run()
	var ee *exec.ExitError
	switch {
	case err == nil:
		return 0, true
	case errors.As(err, &ee) && ee.ExitCode() == 75:
		fmt.Println(verdict(inconclusive, "lint: a timed run holds the timing lock (timing_lock.py status names it); wait for it"))
		return 2, true
	case errors.As(err, &ee):
		return ee.ExitCode(), true
	default:
		fmt.Println(verdict(inconclusive, "lint: could not run under the timing lock: "+err.Error()))
		return 2, true
	}
}
