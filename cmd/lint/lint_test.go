package main

import (
	"go/build"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

const repoRoot = "../.."

func TestPinnedVersion(t *testing.T) {
	gomod, err := os.ReadFile(filepath.Join(repoRoot, pinGoMod))
	if err != nil {
		t.Fatal(err)
	}
	v, err := pinnedVersion(string(gomod))
	if err != nil || v == "" {
		t.Fatalf("pinnedVersion = %q, %v", v, err)
	}
	if !versionMatches("staticcheck 2026.2.1 ("+v+")\n", v) {
		t.Errorf("versionMatches rejects the pin's own version string")
	}
	if versionMatches("staticcheck 2025.1.1 (0.6.1)\n", v) {
		t.Errorf("versionMatches accepts an older staticcheck")
	}
}

func TestCanaryFired(t *testing.T) {
	if !canaryFired("cmd/lint/canary/canary.go:8:2: field deadField is unused (U1000)\n") {
		t.Error("a real canary finding did not count as fired")
	}
	// The failure shapes: nothing printed (could not run), an import error (cannot read export data), and a finding
	// that is not the canary's.
	for _, out := range []string{
		"",
		"-: internal error in importing \"internal/cpu\" (unsupported version: 4)\n",
		"other/x.go:1:1: should omit type (ST1023)\n",
	} {
		if canaryFired(out) {
			t.Errorf("canaryFired(%q) = true", out)
		}
	}
}

func fakeStaticcheck(t *testing.T, script string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "staticcheck")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestBrokenStaticcheckIsInconclusive runs the gate with a "staticcheck" that prints nothing and exits 0, the signature of
// a staticcheck that cannot read export data. No cell may pass, and the run must not read PASS.
func TestBrokenStaticcheckIsInconclusive(t *testing.T) {
	cells := checks(repoRoot, fakeStaticcheck(t, "exit 0\n"), nil)
	for _, c := range cells {
		if c.outcome == ok {
			t.Errorf("cell %s passed on a staticcheck that analyses nothing", c.name)
		}
	}
	if r := reconcile(cells); r.outcome != inconclusive || r.exit != 2 {
		t.Errorf("run = %s exit %d, want INCONCLUSIVE exit 2", r.outcome, r.exit)
	}
}

// TestFindingIsFail drives a staticcheck that is the pin, fires the canary, and reports a finding on one target only when
// it is run for CI's platform. That target must FAIL, the others pass, and the run reads FAIL.
func TestFindingIsFail(t *testing.T) {
	gomod, err := os.ReadFile(filepath.Join(repoRoot, pinGoMod))
	if err != nil {
		t.Fatal(err)
	}
	pin, err := pinnedVersion(string(gomod))
	if err != nil {
		t.Fatal(err)
	}
	bin := fakeStaticcheck(t, `case "$*" in
  -version) echo "staticcheck 2026.2.1 (`+pin+`)"; exit 0 ;;
  *`+canaryTag+`*) echo "cmd/lint/canary/canary.go:8:2: `+canaryFinding+`"; exit 1 ;;
  *./cuda/...*) [ "$GOOS" = linux ] && { echo "cuda/x.go:1:1: bad (SA4006)"; exit 1; } ;;
esac
exit 0
`)
	cells := checks(repoRoot, bin, nil)
	got := map[string]outcome{}
	for _, c := range cells {
		got[c.name] = c.outcome
	}
	want := map[string]outcome{
		"staticcheck version": ok, "canary/linux": ok,
		"staticcheck/root": ok, "staticcheck/gpu": ok, "staticcheck/demo/agent": ok, "staticcheck/cuda": fail,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("cells = %v, want %v", got, want)
	}
	if r := reconcile(cells); r.outcome != fail || r.exit != 1 {
		t.Errorf("run = %s exit %d, want FAIL exit 1", r.outcome, r.exit)
	}
	only := checks(repoRoot, bin, map[string]bool{"root": true})
	if r := reconcile(only); r.outcome != ok || r.na != 3 {
		t.Errorf("-only root: run = %s with %d n/a, want PASS with 3 n/a", r.outcome, r.na)
	}
}

func TestWorkspaceEnv(t *testing.T) {
	root, err := filepath.Abs(repoRoot)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if env, err := workspaceEnv(root, dir, target{name: "root"}); err != nil || env != "GOWORK=off" {
		t.Errorf("no-workspace target: %q, %v; want GOWORK=off", env, err)
	}
	env, err := workspaceEnv(root, dir, target{name: "cuda", workspace: []string{"cuda"}})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(strings.TrimPrefix(env, "GOWORK="))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\t" + root + "\n", "\t" + filepath.Join(root, "cuda") + "\n"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("go.work %q lacks use line %q", b, want)
		}
	}
}

// TestTargets_matchCI holds the target list to ci.yml: every `run: staticcheck …` step, with the modules its job's
// `go work init` stitches. A step added to CI and not to lint, or the reverse, is red.
func TestTargets_matchCI(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(repoRoot, ".github/workflows/ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	jobRe := regexp.MustCompile(`^  [A-Za-z0-9_-]+:\s*$`)
	initRe := regexp.MustCompile(`^\s+run: go work init (.+)$`)
	runRe := regexp.MustCompile(`^\s+run: staticcheck (.+)$`)
	type step struct {
		args      string
		workspace []string
	}
	var ci []step
	var ws []string
	for _, line := range strings.Split(string(b), "\n") {
		if jobRe.MatchString(line) {
			ws = nil
		}
		if m := initRe.FindStringSubmatch(line); m != nil {
			ws = nil
			for _, f := range strings.Fields(m[1]) {
				if f != "." {
					ws = append(ws, strings.TrimPrefix(f, "./"))
				}
			}
		}
		if m := runRe.FindStringSubmatch(line); m != nil {
			ci = append(ci, step{args: strings.TrimSpace(m[1]), workspace: ws})
		}
	}
	var mine []step
	for _, tg := range targets {
		mine = append(mine, step{args: strings.Join(quoteArgs(tg.args), " "), workspace: tg.workspace})
	}
	if len(ci) == 0 {
		t.Fatal("found no `run: staticcheck` step in ci.yml; the parser no longer matches its shape")
	}
	if !reflect.DeepEqual(ci, mine) {
		t.Errorf("ci.yml's staticcheck steps\n  %v\ndiffer from cmd/lint's targets\n  %v", ci, mine)
	}
}

func TestPlatformEnv(t *testing.T) {
	plain := target{name: "root"}
	if env, goos := platformEnv(plain, "darwin", "arm64"); goos != "linux" || !reflect.DeepEqual(env, ciEnv) {
		t.Errorf("a cgo-free target on darwin: %v %s, want CI's linux env", env, goos)
	}
	c := target{name: "gpu", cgo: true}
	if env, goos := platformEnv(c, "linux", "amd64"); goos != "linux" || !reflect.DeepEqual(env, []string{"GOOS=linux", "GOARCH=amd64", "CGO_ENABLED=1"}) {
		t.Errorf("a cgo target on linux: %v %s", env, goos)
	}
	if env, goos := platformEnv(c, "darwin", "arm64"); goos != "darwin" || !reflect.DeepEqual(env, []string{"GOOS=darwin", "GOARCH=arm64", "CGO_ENABLED=1"}) {
		t.Errorf("a cgo target on darwin: %v %s", env, goos)
	}
}

// TestCgoTargets_noPlatformFiles is what lets a cgo target's host-platform run off Linux stand in for CI's: every .go file
// in its module must be in or out of the build identically for darwin/arm64 and linux/amd64, under the target's tags.
func TestCgoTargets_noPlatformFiles(t *testing.T) {
	for _, tg := range targets {
		if !tg.cgo {
			continue
		}
		var tags []string
		for i, a := range tg.args {
			if a == "-tags" && i+1 < len(tg.args) {
				tags = strings.Fields(tg.args[i+1])
			}
		}
		ctx := func(goos, goarch string) build.Context {
			c := build.Default
			c.GOOS, c.GOARCH, c.CgoEnabled, c.BuildTags = goos, goarch, true, tags
			return c
		}
		mac, linux := ctx("darwin", "arm64"), ctx("linux", "amd64")
		dir := filepath.Join(repoRoot, strings.TrimSuffix(strings.TrimPrefix(tg.args[len(tg.args)-1], "./"), "/..."))
		n := 0
		err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() && d.Name() == "testdata" {
				return filepath.SkipDir
			}
			if d.IsDir() || !strings.HasSuffix(p, ".go") {
				return nil
			}
			n++
			a, err := mac.MatchFile(filepath.Dir(p), d.Name())
			if err != nil {
				return err
			}
			b, err := linux.MatchFile(filepath.Dir(p), d.Name())
			if err != nil {
				return err
			}
			if a != b {
				t.Errorf("%s: in the darwin build %v, in the linux build %v; target %s's host-platform run no longer covers CI's", p, a, b, tg.name)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if n == 0 {
			t.Errorf("target %s: no .go files under %s", tg.name, dir)
		}
	}
}
