package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
)

// A gate's own gate, committed rather than typed.
//
// The policy requires that a gate land with a demonstration it can FAIL. Running that demonstration
// as an ad-hoc one-liner produced two defects of its own in this repo, and BOTH reported a mutation
// as verified while nothing had been exercised:
//
//   - `command -v staticcheck >/dev/null && staticcheck …` — the binary was not on PATH, the &&
//     short-circuited, the whole check evaluated to nothing, and it was reported as clean.
//   - `python3 lint.py 2>&1 | head -3; echo "exit=$?"` — $? read head's status, not the lint's, so a
//     red mutation printed exit=0.
//
// A mutation check that silently reads the wrong status certifies a gate as falsifiable when nothing
// ran: G-01 inside the mechanism built to prevent G-01.
//
// THE SHELL VERSION DEFENDED AGAINST THAT BY DISCIPLINE — "the status path here contains NO PIPES
// and no && chains" — which is a rule someone has to keep remembering. Here there is no status path
// to get wrong: exec.Cmd.Run returns the command's own error, and a missing `sed` is an explicit
// LookPath failure rather than a short-circuit that evaluates to success. The class is gone by
// construction, which is the whole argument for the migration (E8 §2).
//
// Usage:
//
//	gate mutation <name> <file> <sed-expr> <verify-cmd...>
//
//	<file>      is backed up and restored, including on failure or interrupt.
//	<sed-expr>  is applied in place; it MUST change the file (asserted — a no-op mutation is the
//	            defect that makes a mutation check vacuous, and it happened: float32(v/sc) where
//	            both operands were already float32).
//	<verify>    must EXIT 0 before the mutation and NON-ZERO after it.
//
// Example:
//
//	gate mutation int4-quantizer decoder/weightmat.go \
//	  's/^const int4GroupSize = 32$/const int4GroupSize = 64/' \
//	  go test ./decoder/ -run TestInt4_forwardParity -count=1

// runMutation returns the process exit code: 0 falsifiable, 1 not, 2 misuse.
func runMutation(argv []string, w io.Writer) int {
	if len(argv) < 4 {
		fmt.Fprintln(os.Stderr, "usage: gate mutation <name> <file> <sed-expr> <verify-cmd...>")
		return 2
	}
	name, file, expr, verify := argv[0], argv[1], argv[2], argv[3:]

	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: %v\n", name, err)
		return 2
	}
	if err := os.Chdir(root); err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: %v\n", name, err)
		return 2
	}
	if st, err := os.Stat(file); err != nil || st.IsDir() {
		fmt.Fprintf(os.Stderr, "mutation[%s]: no such file: %s\n", name, file)
		return 2
	}
	// A missing `sed` must be an ERROR, not a silently skipped mutation. This is the exact
	// anti-pattern the script's own header records fixing, one layer up.
	if _, err := exec.LookPath("sed"); err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: sed not on PATH: %v\n", name, err)
		return 2
	}

	original, err := os.ReadFile(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: %v\n", name, err)
		return 2
	}
	info, err := os.Stat(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: %v\n", name, err)
		return 2
	}
	bak := filepath.Join(os.TempDir(), "gate_mutation_"+sanitize(name)+".bak")
	if err := os.WriteFile(bak, original, info.Mode().Perm()); err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: cannot stage a backup: %v\n", name, err)
		return 2
	}
	restore := func() {
		_ = os.WriteFile(file, original, info.Mode().Perm())
		_ = os.Remove(bak)
	}
	// RESTORE ON INTERRUPT, not only on return. This edits a source file in place, so Ctrl-C between
	// the mutation and the restore would otherwise leave a deliberately broken tree behind — and the
	// next thing the operator runs would fail for a reason that has nothing to do with their change.
	// A deferred call does not run on a signal, so the handler is explicit.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigs:
			restore()
			fmt.Fprintf(w, "\n  interrupted — %s restored\n", file)
			os.Exit(130)
		case <-done:
		}
	}()
	defer func() { close(done); signal.Stop(sigs) }()
	defer restore()

	fmt.Fprintf(w, "%s== mutation[%s] ==%s\n", bold, name, off)

	// 1. BASELINE. A gate that is ALREADY RED proves nothing when it goes red under mutation — the
	//    mutation would not be what turned it. Checked first, deliberately.
	if rc := runQuiet(verify); rc != 0 {
		fmt.Fprintf(w, "  %sFAIL%s: the gate is ALREADY RED before any mutation (exit %d).\n", red, off, rc)
		fmt.Fprintf(w, "        A mutation cannot demonstrate falsifiability against a failing baseline.\n")
		return 1
	}
	fmt.Fprintf(w, "  baseline green\n")

	// 2. MUTATE, and assert the mutation actually CHANGED something. A sed expression that matches
	//    nothing leaves a green run that looks like a verified mutation check.
	sed := exec.Command("sed", "-i", expr, file)
	if out, err := sed.CombinedOutput(); err != nil {
		fmt.Fprintf(w, "  %sFAIL%s: sed failed: %v\n%s\n", red, off, err, strings.TrimSpace(string(out)))
		return 1
	}
	mutated, err := os.ReadFile(file)
	if err != nil {
		fmt.Fprintf(w, "  %sFAIL%s: cannot re-read %s: %v\n", red, off, file, err)
		return 1
	}
	if bytes.Equal(mutated, original) {
		fmt.Fprintf(w, "  %sFAIL%s: the sed expression changed NOTHING — this mutation check is vacuous.\n", red, off)
		fmt.Fprintf(w, "        expr: %s\n", expr)
		return 1
	}
	fmt.Fprintf(w, "  mutation applied\n")

	// 3. The gate must now be RED.
	mutRC := runQuiet(verify)
	if mutRC == 0 {
		fmt.Fprintf(w, "  %sFAIL%s: the gate PASSED under mutation — it cannot see the thing it claims to check.\n", red, off)
		return 1
	}
	fmt.Fprintf(w, "  red under mutation (exit %d)\n", mutRC)

	// 4. RESTORE and confirm green again, so a red is attributable to the mutation and not to drift
	//    that happened to coincide with it.
	restore()
	if rc := runQuiet(verify); rc != 0 {
		fmt.Fprintf(w, "  %sFAIL%s: still red after restore (exit %d) — the tree did not come back.\n", red, off, rc)
		return 1
	}
	fmt.Fprintf(w, "  green after restore\n")
	fmt.Fprintf(w, "  %sPASS%s: %s is falsifiable\n", green, off, name)
	return 0
}

// runQuiet runs the verify command and returns its exit code — the command's OWN code, taken from
// the error exec returns. There is no pipeline whose status could be read by mistake.
func runQuiet(argv []string) int {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode()
		}
		return -1 // could not start at all: not a green run
	}
	return 0
}

func repoRoot() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", fmt.Errorf("not inside a git repository: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}
