package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

const (
	canaryFinding = "field deadField is unused (U1000)"
	canaryTag     = "goinfer_staticcheck_canary"
	canaryPkg     = "./cmd/lint/canary"
	pinGoMod      = ".github/actions/staticcheck/go.mod"
)

// target is one staticcheck run CI makes. workspace lists the modules (beyond the root) CI's job stitches with
// `go work init`; nil means the job runs with no workspace. TestTargets_matchCI holds this list to ci.yml's.
//
// cgo marks a target whose dependencies only type-check with cgo (gpu/: the WebGPU binding). cgo cannot cross-compile
// from the Mac, so off Linux that target runs for the host's platform instead, and its cell says so;
// TestCgoTargets_noPlatformFiles keeps that equivalent by holding those modules free of OS- and arch-specific files.
type target struct {
	name      string
	args      []string
	workspace []string
	cgo       bool
}

var targets = []target{
	{name: "root", args: []string{"./..."}},
	{name: "gpu", args: []string{"-tags", "gpu goinfer_testhooks", "./gpu/..."}, workspace: []string{"gpu", "demo/agent"}, cgo: true},
	{name: "demo/agent", args: []string{"./demo/agent/..."}, workspace: []string{"gpu", "demo/agent"}},
	{name: "cuda", args: []string{"-tags", "cuda goinfer_testhooks", "./cuda/..."}, workspace: []string{"cuda"}},
}

// ciEnv is the platform CI's staticcheck jobs run on (ubuntu-24.04).
var ciEnv = []string{"GOOS=linux", "GOARCH=amd64", "CGO_ENABLED=0"}

// platformEnv returns the environment t runs under on a host of hostOS/hostArch, and the GOOS it analyses.
func platformEnv(t target, hostOS, hostArch string) ([]string, string) {
	if !t.cgo {
		return ciEnv, "linux"
	}
	if hostOS == "linux" {
		return []string{"GOOS=linux", "GOARCH=" + hostArch, "CGO_ENABLED=1"}, "linux"
	}
	return []string{"GOOS=" + hostOS, "GOARCH=" + hostArch, "CGO_ENABLED=1"}, hostOS
}

// pinnedVersion reads the staticcheck module version (e.g. "0.8.1") from the CI action's go.mod, the one place the pin
// lives.
func pinnedVersion(gomod string) (string, error) {
	m := regexp.MustCompile(`(?m)^\s*(?:require\s+)?honnef\.co/go/tools\s+v(\S+)`).FindStringSubmatch(gomod)
	if m == nil {
		return "", fmt.Errorf("%s does not require honnef.co/go/tools", pinGoMod)
	}
	return m[1], nil
}

// versionMatches reports whether `staticcheck -version` output (e.g. "staticcheck 2026.2.1 (0.8.1)") is the pin.
func versionMatches(out, pin string) bool {
	return strings.Contains(out, "("+pin+")")
}

// canaryFired reports whether staticcheck's output on the canary package names the dead field.
func canaryFired(out string) bool {
	return strings.Contains(out, "canary.go") && strings.Contains(out, canaryFinding)
}

// goDirective reads the go line of root's go.mod, which a scratch go.work must not be older than.
func goDirective(root string) (string, error) {
	b, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return "", err
	}
	m := regexp.MustCompile(`(?m)^go\s+(\S+)`).FindSubmatch(b)
	if m == nil {
		return "", fmt.Errorf("go.mod has no go directive")
	}
	return string(m[1]), nil
}

// workspaceEnv returns the GOWORK setting for t: off for a target CI runs without a workspace, else a scratch go.work
// under dir that uses the root and t's modules, as CI's `go work init` does. The developer's own go.work is never
// read: it may lack a module (the Mac's has no ./cuda) or point at a local aikit CI does not use.
func workspaceEnv(root, dir string, t target) (string, error) {
	if t.workspace == nil {
		return "GOWORK=off", nil
	}
	ver, err := goDirective(root)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "go %s\n\nuse (\n\t%s\n", ver, root)
	for _, m := range t.workspace {
		fmt.Fprintf(&b, "\t%s\n", filepath.Join(root, m))
	}
	b.WriteString(")\n")
	p := filepath.Join(dir, strings.ReplaceAll(t.name, "/", "_")+".go.work")
	if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
		return "", err
	}
	return "GOWORK=" + p, nil
}

// checks returns the gate: the pinned version, the canary under CI's platform, then each target. A target is judged only
// when the canary fired; otherwise a clean result means nothing and the cell is INCONCLUSIVE.
func checks(root, bin string, only map[string]bool) []cell {
	var cells []cell
	ver := cell{name: "staticcheck version"}
	pin := ""
	gomod, err := os.ReadFile(filepath.Join(root, pinGoMod))
	if err == nil {
		pin, err = pinnedVersion(string(gomod))
	}
	out, verr := runIn(root, nil, bin, "-version")
	switch {
	case err != nil:
		ver.outcome, ver.detail = inconclusive, err.Error()
	case verr != nil:
		ver.outcome, ver.detail = inconclusive, fmt.Sprintf("%s -version: %v\n%s", bin, verr, out)
	case !versionMatches(out, pin):
		ver.outcome = inconclusive
		ver.detail = fmt.Sprintf("%s is %q; the pin is v%s. Build it: GOWORK=off go build -C .github/actions/staticcheck "+
			"-o \"$(go env GOPATH)/bin/staticcheck\" honnef.co/go/tools/cmd/staticcheck", bin, strings.TrimSpace(out), pin)
	default:
		ver.outcome, ver.fields = ok, []string{"version=" + strings.TrimSpace(out)}
	}
	cells = append(cells, ver)

	can := cell{name: "canary/linux"}
	out, _ = runIn(root, append([]string{"GOWORK=off"}, ciEnv...), bin, "-tags", canaryTag, canaryPkg)
	if canaryFired(out) {
		can.outcome = ok
	} else {
		can.outcome = inconclusive
		can.detail = "staticcheck did not report \"" + canaryFinding + "\" on " + canaryPkg +
			": it is not analysing, so a clean run proves nothing\n" + out
	}
	cells = append(cells, can)

	scratch, serr := os.MkdirTemp("", "goinfer-lint-")
	if serr == nil {
		defer os.RemoveAll(scratch)
	}
	for _, t := range targets {
		c := cell{name: "staticcheck/" + t.name, fields: []string{"args=" + strings.Join(quoteArgs(t.args), " ")}}
		if only != nil && !only[t.name] {
			c.outcome = na
			cells = append(cells, c)
			continue
		}
		if can.outcome != ok {
			c.outcome, c.detail = inconclusive, "not judged: the canary did not fire"
			cells = append(cells, c)
			continue
		}
		gowork := ""
		err := serr
		if err == nil {
			gowork, err = workspaceEnv(root, scratch, t)
		}
		if err != nil {
			c.outcome, c.detail = inconclusive, "could not set up the workspace: "+err.Error()
			cells = append(cells, c)
			continue
		}
		env, goos := platformEnv(t, runtime.GOOS, runtime.GOARCH)
		if goos != "linux" {
			c.fields = append(c.fields, "goos="+goos+" (cgo cannot cross-compile; CI runs linux)")
		}
		out, err := runIn(root, append([]string{gowork}, env...), bin, t.args...)
		if err != nil {
			c.outcome, c.detail = fail, out
		} else {
			c.outcome = ok
		}
		cells = append(cells, c)
	}
	return cells
}

func quoteArgs(args []string) []string {
	q := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t") {
			a = "'" + a + "'"
		}
		q[i] = a
	}
	return q
}

func runIn(dir string, env []string, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}
