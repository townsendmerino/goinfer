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
	"io"
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
	if cwd, err := os.Getwd(); err == nil {
		*bin = resolveBin(*bin, cwd)
	}
	if !*underLock {
		exe, _ := os.Executable()
		os.Exit(underSharedLock(filepath.Join(root, "scripts", "timing_lock.py"), exe, os.Args[1:], os.Stdout))
	}
	os.Exit(printReport(os.Stdout, "lint", checks(root, *bin, only)).exit)
}

// underSharedLock re-runs lint as exe under `script run --shared` and returns the exit code to leave with. It never falls
// back to an unlocked run: a missing lock script or executable, or a refusal (the script's 75, when a timed run holds the
// lock), is an INCONCLUSIVE verdict and exit 2.
func underSharedLock(script, exe string, argv []string, w io.Writer) int {
	if _, err := os.Stat(script); err != nil {
		fmt.Fprintln(w, verdict(inconclusive, "lint: no timing lock script ("+err.Error()+"); it does not run unlocked"))
		return 2
	}
	if exe == "" {
		fmt.Fprintln(w, verdict(inconclusive, "lint: cannot find its own executable to re-run under the timing lock"))
		return 2
	}
	args := append([]string{script, "run", "--shared", "--label", "lint", "--", exe, "-under-timing-lock"}, argv...)
	cmd := exec.Command("python3", args...)
	cmd.Stdout, cmd.Stderr = w, w
	err := cmd.Run()
	var ee *exec.ExitError
	switch {
	case err == nil:
		return 0
	case errors.As(err, &ee) && ee.ExitCode() == 75:
		fmt.Fprintln(w, verdict(inconclusive, "lint: a timed run holds the timing lock (timing_lock.py status names it); wait for it"))
		return 2
	case errors.As(err, &ee):
		return ee.ExitCode()
	default:
		fmt.Fprintln(w, verdict(inconclusive, "lint: could not run under the timing lock: "+err.Error()))
		return 2
	}
}

// resolveBin makes a relative -staticcheck path absolute against the directory lint was started in, since every run
// executes from the repository root. A bare name is left for PATH lookup.
func resolveBin(bin, cwd string) string {
	if filepath.IsAbs(bin) || !strings.ContainsRune(bin, '/') && !strings.ContainsRune(bin, filepath.Separator) {
		return bin
	}
	return filepath.Join(cwd, bin)
}
