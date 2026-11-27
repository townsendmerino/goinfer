package gatehygiene

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// RunMutation implements `gate mutation <name> <file> <sed-expr> <verify-cmd...>`.
func RunMutation(argv []string, w io.Writer) (rc int) {
	if len(argv) < 4 {
		fmt.Fprintln(os.Stderr, "usage: gate mutation <name> <file> <sed-expr> <verify-cmd...>")
		return 2
	}
	name, file, expr, verify := argv[0], argv[1], argv[2], argv[3:]

	root, err := gatecore.RepoRoot()
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
	bak := filepath.Join(os.TempDir(), "gate_mutation_"+gatecore.Sanitize(name)+".bak")
	if err := os.WriteFile(bak, original, info.Mode().Perm()); err != nil {
		fmt.Fprintf(os.Stderr, "mutation[%s]: cannot stage a backup: %v\n", name, err)
		return 2
	}
	restore := func() error {
		if err := os.WriteFile(file, original, info.Mode().Perm()); err != nil {
			fmt.Fprintf(w, "  %sFAIL%s: %s NOT restored (%v); the original is at %s\n", gatecore.Red, gatecore.Off, file, err, bak)
			return err
		}
		_ = os.Remove(bak)
		return nil
	}
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigs:
			if restore() == nil {
				fmt.Fprintf(w, "\n  interrupted — %s restored\n", file)
			}
			os.Exit(130)
		case <-done:
		}
	}()
	defer func() { close(done); signal.Stop(sigs) }()
	defer func() {
		if restore() != nil {
			rc = 2
		}
	}()

	fmt.Fprintf(w, "%s== mutation[%s] ==%s\n", gatecore.Bold, name, gatecore.Off)

	if rc := runQuiet(verify); rc != 0 {
		fmt.Fprintf(w, "  %sFAIL%s: the gate is ALREADY RED before any mutation (exit %d).\n", gatecore.Red, gatecore.Off, rc)
		fmt.Fprintf(w, "        A mutation cannot demonstrate falsifiability against a failing baseline.\n")
		return 1
	}
	fmt.Fprintf(w, "  baseline green\n")

	mutated, err := exec.Command("sed", sedArgs(expr, file)...).Output()
	if err != nil {
		msg := err.Error()
		if ee, ok := errors.AsType[*exec.ExitError](err); ok {
			msg = strings.TrimSpace(string(ee.Stderr))
		}
		fmt.Fprintf(w, "  %sFAIL%s: sed failed: %s\n", gatecore.Red, gatecore.Off, msg)
		return 1
	}
	if err := os.WriteFile(file, mutated, info.Mode().Perm()); err != nil {
		fmt.Fprintf(w, "  %sFAIL%s: cannot write %s: %v\n", gatecore.Red, gatecore.Off, file, err)
		return 1
	}
	if bytes.Equal(mutated, original) {
		fmt.Fprintf(w, "  %sFAIL%s: the sed expression changed NOTHING — this mutation check is vacuous.\n", gatecore.Red, gatecore.Off)
		fmt.Fprintf(w, "        expr: %s\n", expr)
		return 1
	}
	fmt.Fprintf(w, "  mutation applied\n")

	mutRC := runQuiet(verify)
	if mutRC == 0 {
		fmt.Fprintf(w, "  %sFAIL%s: the gate PASSED under mutation — it cannot see the thing it claims to check.\n", gatecore.Red, gatecore.Off)
		return 1
	}
	fmt.Fprintf(w, "  red under mutation (exit %d)\n", mutRC)

	if restore() != nil {
		return 2
	}
	if rc := runQuiet(verify); rc != 0 {
		fmt.Fprintf(w, "  %sFAIL%s: still red after restore (exit %d) — the tree did not come back.\n", gatecore.Red, gatecore.Off, rc)
		return 1
	}
	fmt.Fprintf(w, "  green after restore\n")
	fmt.Fprintf(w, "  %sPASS%s: %s is falsifiable\n", gatecore.Green, gatecore.Off, name)
	return 0
}

func runQuiet(argv []string) int {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Run(); err != nil {
		if ee, ok := errors.AsType[*exec.ExitError](err); ok {
			return ee.ExitCode()
		}
		return -1
	}
	return 0
}

func sedArgs(expr, file string) []string { return []string{expr, file} }
