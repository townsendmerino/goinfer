package main

import (
	"fmt"
	"go/build"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
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
	if runtime.GOOS == "windows" {
		t.Skip("the fake staticcheck is a shell script; the lint runs on darwin and linux")
	}
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
	_, cgoOS := platformEnv(target{cgo: true}, runtime.GOOS, runtime.GOARCH)
	want := map[string]outcome{
		"staticcheck version": ok, "canary/linux": ok, "canary/" + cgoOS + "-cgo": ok,
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
	for _, want := range []string{"\t" + strconv.Quote(root) + "\n", "\t" + strconv.Quote(filepath.Join(root, "cuda")) + "\n"} {
		if !strings.Contains(string(b), want) {
			t.Errorf("go.work %q lacks use line %q", b, want)
		}
	}
}

// TestTargets_matchCI holds the target list to ci.yml: every staticcheck step, with the modules its job's `go work init`
// stitches. A step added to CI and not to lint, or the reverse, is red.
func TestTargets_matchCI(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(repoRoot, ".github/workflows/ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	ci := parseCISteps(string(b))
	var mine []ciStep
	for _, tg := range targets {
		mine = append(mine, ciStep{args: strings.Join(quoteArgs(tg.args), " "), workspace: tg.workspace})
	}
	if len(ci) == 0 {
		t.Fatal("found no staticcheck step in ci.yml; the parser no longer matches its shape")
	}
	if !reflect.DeepEqual(ci, mine) {
		t.Errorf("ci.yml's staticcheck steps\n  %v\ndiffer from cmd/lint's targets\n  %v", ci, mine)
	}
}

// TestParseCISteps_blockScalars: a step written as a `run: |` block must be found, or a new one there passes vacuously.
func TestParseCISteps_blockScalars(t *testing.T) {
	yml := `jobs:
  one:
    steps:
      - name: workspace
        run: |
          go work init . ./gpu
          go work sync
      - name: staticcheck
        run: |
          staticcheck -tags 'gpu goinfer_testhooks' ./gpu/...
          staticcheck ./demo/agent/...
  two:
    steps:
      - uses: ./.github/actions/staticcheck
      - run: staticcheck ./...
`
	got := parseCISteps(yml)
	want := []ciStep{
		{args: "-tags 'gpu goinfer_testhooks' ./gpu/...", workspace: []string{"gpu"}},
		{args: "./demo/agent/...", workspace: []string{"gpu"}},
		{args: "./...", workspace: nil},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseCISteps = %v, want %v", got, want)
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

// TestCgoTargets_noPlatformFiles is what lets a cgo target's host-platform run stand in for CI's linux/amd64: every .go
// file in its module, at any depth, must be in or out of the build identically on darwin and linux, amd64 and arm64.
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
		plats := []build.Context{ctx("darwin", "arm64"), ctx("darwin", "amd64"), ctx("linux", "amd64"), ctx("linux", "arm64")}
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
			var in []string
			for _, c := range plats {
				m, err := c.MatchFile(filepath.Dir(p), d.Name())
				if err != nil {
					return err
				}
				in = append(in, fmt.Sprintf("%s/%s=%v", c.GOOS, c.GOARCH, m))
			}
			for _, x := range in[1:] {
				if strings.HasSuffix(x, "=true") != strings.HasSuffix(in[0], "=true") {
					t.Errorf("%s: in or out of the build by platform (%s); target %s's host-platform run no longer covers CI's", p, strings.Join(in, " "), tg.name)
					break
				}
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

// pinnedFake is a staticcheck that reports version v, fires the canary, and reports a finding on ./cuda/....
func pinnedFake(t *testing.T, v string) string {
	return fakeStaticcheck(t, `case "$*" in
  -version) echo "staticcheck 2026.2.1 (`+v+`)"; exit 0 ;;
  *`+canaryTag+`*) echo "cmd/lint/canary/canary.go:8:2: `+canaryFinding+`"; exit 1 ;;
  *./cuda/...*) echo "cuda/x.go:1:1: bad (SA4006)"; exit 1 ;;
esac
exit 0
`)
}

func pin(t *testing.T) string {
	t.Helper()
	gomod, err := os.ReadFile(filepath.Join(repoRoot, pinGoMod))
	if err != nil {
		t.Fatal(err)
	}
	v, err := pinnedVersion(string(gomod))
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// TestOnly_unknownTargetIsInconclusive: a typo in -only judges nothing, so it must not read PASS on the version and
// canary cells alone.
func TestOnly_unknownTargetIsInconclusive(t *testing.T) {
	cells := checks(repoRoot, pinnedFake(t, pin(t)), map[string]bool{"agent": true})
	if r := reconcile(cells); r.outcome != inconclusive || r.exit != 2 {
		t.Errorf("-only agent: run = %s exit %d, want INCONCLUSIVE exit 2", r.outcome, r.exit)
	}
}

// TestUnpinnedStaticcheckJudgesNothing: a staticcheck that is not the pin may still fire the canary; its findings must
// not turn a version mismatch into FAIL, and no target may pass on it.
func TestUnpinnedStaticcheckJudgesNothing(t *testing.T) {
	cells := checks(repoRoot, pinnedFake(t, "0.6.1"), nil)
	for _, c := range cells {
		if strings.HasPrefix(c.name, "staticcheck/") && (c.outcome == ok || c.outcome == fail) {
			t.Errorf("cell %s judged (%s) under an unpinned staticcheck", c.name, c.outcome)
		}
	}
	if r := reconcile(cells); r.outcome != inconclusive {
		t.Errorf("run = %s, want INCONCLUSIVE", r.outcome)
	}
}

// TestWorkspaceEnv_relativeRoot: Go reads a go.work's use paths relative to the go.work file, which is in a temp dir, so
// a relative root must be written absolute, and quoted.
func TestWorkspaceEnv_relativeRoot(t *testing.T) {
	env, err := workspaceEnv(repoRoot, t.TempDir(), target{name: "gpu", workspace: []string{"gpu"}})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(strings.TrimPrefix(env, "GOWORK="))
	if err != nil {
		t.Fatal(err)
	}
	abs, _ := filepath.Abs(repoRoot)
	for _, want := range []string{strconv.Quote(abs), strconv.Quote(filepath.Join(abs, "gpu"))} {
		if !strings.Contains(string(b), "\t"+want+"\n") {
			t.Errorf("go.work %q lacks use line %s", b, want)
		}
	}
}

func fakeLockScript(t *testing.T, exit int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the fake lock script runs under python3 from a shell path")
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "timing_lock.py")
	if err := os.WriteFile(p, []byte("import sys\nsys.exit("+strconv.Itoa(exit)+")\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestUnderSharedLock: lint never runs unlocked by accident. A missing lock script or executable is INCONCLUSIVE, a
// refusal (75) is INCONCLUSIVE, and the re-run's own exit code passes through.
func TestUnderSharedLock(t *testing.T) {
	for _, c := range []struct {
		name       string
		script     string
		exe        string
		want       int
		wantRefuse bool
	}{
		{"no script", filepath.Join(t.TempDir(), "missing.py"), "/bin/true", 2, true},
		{"no executable", fakeLockScript(t, 0), "", 2, true},
		{"refused", fakeLockScript(t, 75), "/bin/true", 2, true},
		{"passes", fakeLockScript(t, 0), "/bin/true", 0, false},
		{"fails", fakeLockScript(t, 1), "/bin/true", 1, false},
	} {
		var out strings.Builder
		got := underSharedLock(c.script, c.exe, nil, &out)
		if got != c.want {
			t.Errorf("%s: exit %d, want %d (output %q)", c.name, got, c.want, out.String())
		}
		if c.wantRefuse != strings.Contains(out.String(), "VERDICT: INCONCLUSIVE") {
			t.Errorf("%s: output %q", c.name, out.String())
		}
	}
}

func TestResolveBin(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix paths")
	}
	cwd := filepath.FromSlash("/work/sub")
	for in, want := range map[string]string{
		"staticcheck":         "staticcheck",
		"./bin/staticcheck":   filepath.Join(cwd, "bin", "staticcheck"),
		"/opt/sc/staticcheck": "/opt/sc/staticcheck",
	} {
		if got := resolveBin(in, cwd); got != filepath.FromSlash(want) && got != want {
			t.Errorf("resolveBin(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestCanaryPerPlatform: a target is judged only by the canary of the environment it runs in, so the cgo target needs a
// canary run under cgo.
func TestCanaryPerPlatform(t *testing.T) {
	cells := checks(repoRoot, pinnedFake(t, pin(t)), nil)
	names := map[string]bool{}
	for _, c := range cells {
		names[c.name] = true
	}
	_, goos := platformEnv(target{cgo: true}, runtime.GOOS, runtime.GOARCH)
	if !names["canary/"+goos+"-cgo"] {
		t.Errorf("no cgo canary cell; cells: %v", names)
	}
}
