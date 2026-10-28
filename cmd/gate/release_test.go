package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The rows of `gate release` (release.go), each against what it reads: a repository built in the test, files on disk, or a canned command.

func fakeRun(outs map[string]string, errs map[string]error) func(dir, name string, args ...string) (string, error) {
	return func(dir, name string, args ...string) (string, error) {
		k := name + " " + strings.Join(args, " ")
		for pre, out := range outs {
			if strings.HasPrefix(k, pre) {
				return out, errs[pre]
			}
		}
		return "", fmt.Errorf("unexpected command %q", k)
	}
}

func TestModRequire(t *testing.T) {
	gomod := "module x\n\nrequire (\n\tgithub.com/townsendmerino/aikit v1.69.0\n\tgithub.com/townsendmerino/aikit/gpu v0.33.5\n)\n\nrequire golang.org/x/sys v0.1.0\n"
	if v := modRequire(gomod, "github.com/townsendmerino/aikit"); v != "v1.69.0" {
		t.Errorf("aikit = %q", v)
	}
	if v := modRequire(gomod, "github.com/townsendmerino/aikit/gpu"); v != "v0.33.5" {
		t.Errorf("aikit/gpu = %q", v)
	}
	if v := modRequire(gomod, "golang.org/x/sys"); v != "v0.1.0" {
		t.Errorf("single-line require = %q", v)
	}
	if v := modRequire(gomod, "github.com/nobody"); v != "" {
		t.Errorf("absent module = %q", v)
	}
}

func writeMods(t *testing.T, root string, aikit map[string]string, gpu map[string]string) {
	t.Helper()
	for _, m := range relModules {
		c := "module " + m + "\n\nrequire (\n"
		if v := aikit[m]; v != "" {
			c += "\tgithub.com/townsendmerino/aikit " + v + "\n"
		}
		if v := gpu[m]; v != "" {
			c += "\tgithub.com/townsendmerino/aikit/gpu " + v + "\n"
		}
		c += ")\n"
		writeTree(t, root, map[string]string{m + "/go.mod": c})
	}
}

func TestRowPins(t *testing.T) {
	root := t.TempDir()
	agree := map[string]string{".": "v1.69.0", "gpu": "v1.69.0", "cuda": "v1.69.0", "metal": "v1.69.0", "demo/agent": "v1.69.0"}
	writeMods(t, root, agree, nil)
	e := &relEnv{root: root}
	if r := rowPins(e); r.State != "done" {
		t.Fatalf("agreeing pins: %+v", r)
	}
	// one module on an older aikit
	bad := map[string]string{".": "v1.69.0", "gpu": "v1.69.0", "cuda": "v1.68.0", "metal": "v1.69.0", "demo/agent": "v1.69.0"}
	writeMods(t, root, bad, nil)
	if r := rowPins(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "aikit differs") {
		t.Errorf("aikit disagreement: %+v", r)
	}
	// a module still requiring aikit's gpu module, which goinfer no longer uses
	writeMods(t, root, agree, map[string]string{"cuda": "v0.35.0"})
	if r := rowPins(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "aikit/gpu") {
		t.Errorf("an aikit/gpu require read as done: %+v", r)
	}
	// an untagged aikit, agreed by all five modules, is owed at a release (SD3), in each form Go writes a pseudo-version
	all := func(v string) map[string]string {
		return map[string]string{".": v, "gpu": v, "cuda": v, "metal": v, "demo/agent": v}
	}
	for _, v := range []string{"v0.0.0-20261027120000-abcdefabcdef", "v1.77.0-rc.1.0.20261027120000-abcdefabcdef", "v1.76.1-0.20261027120000-abcdefabcdef"} {
		writeMods(t, root, all(v), nil)
		if r := rowPins(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "pseudo-version") {
			t.Errorf("pseudo-version %s read as releasable: %+v", v, r)
		}
	}
	// a pre-release tag is a tag
	writeMods(t, root, all("v1.77.0-rc.1"), nil)
	if r := rowPins(e); r.State != "done" {
		t.Errorf("pre-release tag v1.77.0-rc.1 read as owed: %+v", r)
	}
	// nothing readable is not agreement
	writeMods(t, root, nil, nil)
	if r := rowPins(e); r.State != "OWED" {
		t.Errorf("no pins found read as agreement: %+v", r)
	}
}

func TestRowColdUser(t *testing.T) {
	root := t.TempDir()
	e := &relEnv{root: root, prev: "v0.22.0"}
	writeTree(t, root, map[string]string{"docs/measurements/cold-user-2026-10-05-nobara-pc.md": "# Cold-user run: goinfer v0.20.0 on nobara-pc, 2026-10-05\n\nbody\n"})
	if r := rowColdUser(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "v0.20.0") {
		t.Errorf("a record for another tag counted: %+v", r)
	}
	writeTree(t, root, map[string]string{"docs/measurements/cold-user-2026-10-12-nobara-pc.md": "# Cold-user run: goinfer v0.22.0 on nobara-pc, 2026-10-12\n"})
	if r := rowColdUser(e); r.State != "done" || !strings.HasSuffix(r.Record, "cold-user-2026-10-12-nobara-pc.md") {
		t.Errorf("the record for the previous tag was not found: %+v", r)
	}
}

func TestRowGPU(t *testing.T) {
	dir, commit := gpuTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	e := &relEnv{root: dir}
	if r := rowGPU(e); r.State != "OWED" {
		t.Fatalf("no records read as covered: %+v", r)
	}
	writeRec := func(be, c string) {
		b, _ := json.Marshal(gpuRecord{Schema: 1, Backend: be, Commit: c, Date: "2026-10-11T10:00:00Z", Verdict: "PASS"})
		writeTree(t, dir, map[string]string{gpuRecordDirDefault + "/" + be + "-x.json": string(b)})
	}
	writeRec("cuda", head)
	if r := rowGPU(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "metal: no record") {
		t.Errorf("cuda alone: %+v", r)
	}
	writeRec("metal", head)
	commit("docs/f.txt", "doc only")
	if r := rowGPU(e); r.State != "done" {
		t.Errorf("both backends at the same code (a docs commit since): %+v", r)
	}
	commit("decoder/f.txt", "code")
	if r := rowGPU(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "decoder/ changed") {
		t.Errorf("a decoder change since the runs: %+v", r)
	}
}

func TestNeverExecuted(t *testing.T) {
	ids, err := neverExecuted([]byte(`{"entries":[{"id":"a","last_executed":[]},{"id":"b","last_executed":[{"date":"x"}]},{"id":"c"}]}`))
	if err != nil || strings.Join(ids, ",") != "a,c" {
		t.Errorf("never executed = %v, %v; want a,c", ids, err)
	}
}

func TestRowHardware(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, map[string]string{"docs/hardware-coverage.json": `{"entries":[{"id":"a","last_executed":[]},{"id":"new","last_executed":[]},{"id":"ok","last_executed":[{"d":1}]}]}`})
	prev := `{"entries":[{"id":"a","last_executed":[]},{"id":"ok","last_executed":[{"d":1}]}]}`
	e := &relEnv{root: root, prev: "v0.22.0", run: fakeRun(map[string]string{"go test": "ok", "git show": prev}, nil)}
	r := rowHardware(e)
	if r.State != "done" || !strings.Contains(strings.Join(r.Detail, " "), "NEW since v0.22.0 with no record: new") {
		t.Errorf("the diff against the previous tag: %+v", r)
	}
	e.run = fakeRun(map[string]string{"go test": "FAIL: matrix stale"}, map[string]error{"go test": fmt.Errorf("exit 1")})
	if r := rowHardware(e); r.State != "OWED" {
		t.Errorf("a stale matrix: %+v", r)
	}
}

// Branches are judged against origin/main with git cherry, so a rebased copy of a merged fix is not listed.
func TestRowBranches(t *testing.T) {
	dir, commit := gpuTestRepo(t)
	gitIn(t, dir, "branch", "-m", "main")
	gitIn(t, dir, "update-ref", "refs/remotes/origin/main", "HEAD")
	e := &relEnv{root: dir, run: osRun}
	gitIn(t, dir, "checkout", "-q", "-b", "wip")
	commit("docs/f.txt", "wip")
	gitIn(t, dir, "checkout", "-q", "main")
	if r := rowBranches(e); r.State != "done" || !strings.Contains(strings.Join(r.Detail, "\n"), "wip") {
		t.Errorf("a branch with no fix-like subject must be listed but not owed: %+v", r)
	}
	gitIn(t, dir, "checkout", "-q", "-b", "fixes")
	writeTree(t, dir, map[string]string{"cuda/g.txt": "g"})
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "cuda: fix the stale-slot read")
	gitIn(t, dir, "checkout", "-q", "main")
	if r := rowBranches(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, "\n"), "fixes") {
		t.Errorf("an unmerged fix branch: %+v", r)
	}
	// the same fix merged into main: no longer owed
	gitIn(t, dir, "cherry-pick", "fixes")
	gitIn(t, dir, "update-ref", "refs/remotes/origin/main", "HEAD")
	if r := rowBranches(e); r.State != "done" || strings.Contains(strings.Join(r.Detail, "\n"), "fixes") {
		t.Errorf("a cherry-picked (patch-equal) fix is still listed: %+v", r)
	}
}

func TestCIVerdict(t *testing.T) {
	sha := strings.Repeat("a", 40)
	for name, tc := range map[string]struct {
		runs  []ciRun
		state string
	}{
		"green":       {[]ciRun{{sha, "completed", "success", "ci"}}, "done"},
		"in progress": {[]ciRun{{sha, "in_progress", "", "ci"}}, "OWED"},
		"failed":      {[]ciRun{{sha, "completed", "failure", "ci"}}, "OWED"},
		"other sha":   {[]ciRun{{strings.Repeat("b", 40), "completed", "success", "ci"}}, "OWED"},
		"other wf":    {[]ciRun{{sha, "completed", "success", "coverage"}}, "OWED"},
		"none":        {nil, "OWED"},
	} {
		if st, _ := ciVerdict(tc.runs, sha); st != tc.state {
			t.Errorf("%s: %s, want %s", name, st, tc.state)
		}
	}
}

func TestRelRefusal(t *testing.T) {
	dir, _ := gpuTestRepo(t) // a clean repo on main
	e := &relEnv{root: dir, tag: "v0.23.0"}
	if why := relRefusal(e); why != "" {
		t.Fatalf("a clean main with a new tag was refused: %s", why)
	}
	for tag, want := range map[string]string{"0.23.0": "not a version tag", "v0.23": "not a version tag", "v0.23.0-rc1": "not a version tag"} {
		if why := relRefusal(&relEnv{root: dir, tag: tag}); !strings.Contains(why, want) {
			t.Errorf("tag %q: %q", tag, why)
		}
	}
	gitIn(t, dir, "tag", "v0.23.0")
	if why := relRefusal(e); !strings.Contains(why, "already exists") {
		t.Errorf("an existing tag: %q", why)
	}
	gitIn(t, dir, "tag", "-d", "v0.23.0")
	if err := os.WriteFile(filepath.Join(dir, "dirty.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if why := relRefusal(e); !strings.Contains(why, "not clean") {
		t.Errorf("a dirty tree: %q", why)
	}
	os.Remove(filepath.Join(dir, "dirty.txt"))
	gitIn(t, dir, "checkout", "-q", "-b", "side")
	if why := relRefusal(e); !strings.Contains(why, "not on main") {
		t.Errorf("a side branch: %q", why)
	}
}

// The report is done only when every row is, and a single OWED or UNKNOWN row makes the exit code 1.
func TestReportRelease(t *testing.T) {
	ok := func(*relEnv) relRow { return done("a", "r") }
	bad := func(*relEnv) relRow { return owed("b", "r2", "why") }
	unk := func(*relEnv) relRow { return unknown("c", "r3", "offline") }
	var buf bytes.Buffer
	e := &relEnv{tag: "v0.23.0", prev: "v0.22.0"}
	if rc := reportRelease(&buf, e, []func(*relEnv) relRow{ok, ok}); rc != 0 || !strings.Contains(buf.String(), "2 of 2 done") {
		t.Errorf("all done: rc=%d\n%s", rc, buf.String())
	}
	buf.Reset()
	if rc := reportRelease(&buf, e, []func(*relEnv) relRow{ok, bad}); rc != 1 || !strings.Contains(buf.String(), "not ready to tag") || !strings.Contains(buf.String(), "why") {
		t.Errorf("one owed: rc=%d\n%s", rc, buf.String())
	}
	buf.Reset()
	if rc := reportRelease(&buf, e, []func(*relEnv) relRow{ok, unk}); rc != 1 {
		t.Errorf("an unknown row must not read as ready: rc=%d", rc)
	}
}

func TestRowGofmtAndLint(t *testing.T) {
	root := t.TempDir()
	e := &relEnv{root: root}
	e.run = func(dir, name string, args ...string) (string, error) {
		if name == "gofmt" && strings.HasSuffix(dir, "cuda") {
			return "x.go", nil
		}
		return "", nil
	}
	for _, m := range relModules {
		os.MkdirAll(filepath.Join(root, m), 0o755)
	}
	if r := rowGofmt(e); r.State != "OWED" || r.Detail[0] != filepath.Join("cuda", "x.go") {
		t.Errorf("gofmt: %+v", r)
	}
	e.run = fakeRun(map[string]string{"python3": "red"}, map[string]error{"python3": fmt.Errorf("exit 1")})
	if r := rowLint(e); r.State != "OWED" {
		t.Errorf("a red lint: %+v", r)
	}
	e.run = fakeRun(map[string]string{"python3": "green"}, nil)
	if r := rowLint(e); r.State != "done" {
		t.Errorf("a green lint: %+v", r)
	}
}
