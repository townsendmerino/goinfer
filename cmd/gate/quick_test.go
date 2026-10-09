package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

// EQUIVALENCE for `gate quick` (TE7, docs/tasks/task-test-efficiency-2026-09.md). The campaign's
// constraint is that a faster check which stops detecting is worse than a slow one, so each claim
// the selection makes is pinned here, and each test below was shown to go RED when the selection
// was deliberately broken (recorded in the TE7 report):
//
//  1. a mutation in a leaf package makes quick select — by the import graph, not by the
//     cache-checked fallback — and run a failing test, in its own module and across modules;
//  2. a mutation in a parity shared-set file makes quick run the forward goldens, and fail on them;
//  3. the AFFECTED tier never drops a test binary that an independent `go list -deps -test` of
//     that binary says compiles the changed file — checked for every Go and embedded file of the
//     real tree, across all five modules.
//
// (1) and (2) run against a scratch two-module git repo, so nothing in the real tree is ever
// edited. Their real-tree counterparts mutate through `go build -overlay` (also never editing a
// file) and are opt-in, since each compiles and runs real packages.

// fxFiles is the scratch tree: leaf <- mid <- app in the root module, core as the "shared set",
// other unrelated, and sub/ a second module whose test imports leaf.
var fxFiles = map[string]string{
	"go.mod":                        "module example.com/fx\n\ngo 1.21\n",
	"leaf/leaf.go":                  "package leaf\n\nfunc Add(a, b int) int { return a + b }\n",
	"leaf/leaf_test.go":             "package leaf\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif Add(2, 3) != 5 {\n\t\tt.Fatal(\"Add(2,3) != 5\")\n\t}\n}\n",
	"mid/mid.go":                    "package mid\n\nimport \"example.com/fx/leaf\"\n\nfunc Twice(x int) int { return leaf.Add(x, x) }\n",
	"app/app.go":                    "package app\n\nimport \"example.com/fx/mid\"\n\nfunc Run() int { return mid.Twice(4) }\n",
	"app/app_test.go":               "package app\n\nimport \"testing\"\n\nfunc TestRun(t *testing.T) {\n\tif Run() != 8 {\n\t\tt.Fatalf(\"Run() = %d, want 8\", Run())\n\t}\n}\n",
	"other/other.go":                "package other\n\nfunc One() int { return 1 }\n",
	"other/other_test.go":           "package other\n\nimport \"testing\"\n\nfunc TestOne(t *testing.T) {\n\tif One() != 1 {\n\t\tt.Fatal(\"One\")\n\t}\n}\n",
	"core/core.go":                  "package core\n\nfunc Scale(x float64) float64 { return x * 2 }\n",
	"core/core_test.go":             "package core\n\nimport \"testing\"\n\n// The golden: the forward numerics, which a numeric change breaks.\nfunc TestFx_forwardParity(t *testing.T) {\n\tif Scale(1.5) != 3 {\n\t\tt.Fatalf(\"Scale(1.5) = %v, want 3 (golden)\", Scale(1.5))\n\t}\n}\n\n// Blind to the scale factor, like most of a package's tests are to a 1-ulp change.\nfunc TestScaleZero(t *testing.T) {\n\tif Scale(0) != 0 {\n\t\tt.Fatal(\"Scale(0)\")\n\t}\n}\n",
	"testdata/parity_manifest.json": `{"shared_sets": {"core": ["core/core.go"]}, "families": {}}` + "\n",
	"sub/go.mod":                    "module example.com/fx/sub\n\ngo 1.21\n\nrequire example.com/fx v0.0.0\n",
	"sub/s.go":                      "package sub\n",
	"sub/s_test.go":                 "package sub\n\nimport (\n\t\"testing\"\n\n\t\"example.com/fx/leaf\"\n)\n\nfunc TestLeafFromSub(t *testing.T) {\n\tif leaf.Add(1, 1) != 2 {\n\t\tt.Fatal(\"leaf.Add(1,1) != 2 seen from module sub\")\n\t}\n}\n",
}

func needTools(t *testing.T) {
	t.Helper()
	for _, bin := range []string{"go", "git"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not on PATH", bin)
		}
	}
}

// fxRepo writes the scratch tree, commits it, and returns its config.
func fxRepo(t *testing.T) *quickConfig {
	t.Helper()
	needTools(t)
	dir := t.TempDir()
	if r, err := filepath.EvalSymlinks(dir); err == nil {
		dir = r
	}
	for name, body := range fxFiles {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "."}, {"-c", "user.email=q@x", "-c", "user.name=q", "commit", "-qm", "base"}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	qc := &quickConfig{
		Root: dir,
		Modules: []*quickModule{
			{Name: "root", Dir: "."},
			{Name: "sub", Dir: "sub", Work: []string{".", "sub"}},
		},
		GoldenPkg: "example.com/fx/core",
		GoldenRun: goldenRunRE,
		Manifest:  "testdata/parity_manifest.json",
		StateDir:  t.TempDir(),
	}
	if err := qc.finish(); err != nil {
		t.Fatal(err)
	}
	return qc
}

func mutate(t *testing.T, qc *quickConfig, rel, from, to string) {
	t.Helper()
	p := filepath.Join(qc.Root, rel)
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	// A mutation that changes nothing makes the whole check vacuous (mutation.go's own lesson).
	if !bytes.Contains(b, []byte(from)) {
		t.Fatalf("mutation target %q not in %s", from, rel)
	}
	if err := os.WriteFile(p, bytes.Replace(b, []byte(from), []byte(to), 1), 0o644); err != nil {
		t.Fatal(err)
	}
}

func runFx(t *testing.T, qc *quickConfig) (int, string) {
	t.Helper()
	var out bytes.Buffer
	rc := quickMain(qc, quickOpts{Base: "HEAD", Jobs: 4, Timeout: "5m", LogDir: t.TempDir(), NoLint: true,
		Progress: io.Discard}, &out)
	return rc, out.String()
}

// cellLine finds the report's line for one test cell: status, name, tier.
func cellLine(t *testing.T, report, name string) string {
	t.Helper()
	report = regexp.MustCompile("\x1b\\[[0-9;]*m").ReplaceAllString(report, "")
	re := regexp.MustCompile(`(?m)^  \S*?(PASS|FAIL|CACHED|ALLSKIP|NOTESTS)\S*\s+` + regexp.QuoteMeta(name) + `\s+(affected|observer|cache-checked)\b.*$`)
	m := re.FindString(report)
	if m == "" {
		t.Fatalf("no test line for cell %q in report:\n%s", name, report)
	}
	return m
}

// (1) A leaf mutation: the failing tests are selected by the IMPORT GRAPH — as "affected", in the
// changed package, in a transitive importer, and in the other module — and quick exits red. The
// unrelated package is not affected and replays from the cache. The tier assertion is what makes
// this test sensitive to the graph: without it, the cache-checked fallback alone would still run
// the failing tests, and a broken reverse-dependency walk would pass unnoticed.
func TestQuick_leafMutationSelectsAndRunsAFailingTest(t *testing.T) {
	qc := fxRepo(t)
	if rc, out := runFx(t, qc); rc != 0 {
		t.Fatalf("baseline must be GREEN before any mutation (a red would prove nothing), rc=%d:\n%s", rc, out)
	}
	mutate(t, qc, "leaf/leaf.go", "a + b", "a - b")
	rc, out := runFx(t, qc)
	if rc != 1 {
		t.Fatalf("rc=%d after a leaf mutation, want 1 (RED):\n%s", rc, out)
	}
	for _, cell := range []string{"leaf", "app", "sub"} {
		ln := cellLine(t, out, cell)
		if !strings.Contains(ln, "FAIL") || !strings.Contains(ln, "affected") {
			t.Errorf("cell %s: want FAIL in the affected tier, got:\n  %s", cell, ln)
		}
	}
	if ln := cellLine(t, out, "other"); !strings.Contains(ln, "CACHED") || strings.Contains(ln, "affected") {
		t.Errorf("unrelated package must be cache-checked and replayed, got:\n  %s", ln)
	}
	if !strings.Contains(out, "imports mid (leaf/leaf.go)") && !strings.Contains(out, "imports leaf (leaf/leaf.go)") {
		t.Errorf("selection does not say WHY app is in it:\n%s", out)
	}
}

// (2) A shared-set mutation that only the golden can see: quick runs the goldens as their own cell,
// they fail, and the verdict is red — while the package's other test passes, so it is the goldens
// cell that caught it.
func TestQuick_sharedSetMutationRunsAndFailsTheGoldens(t *testing.T) {
	qc := fxRepo(t)
	if rc, out := runFx(t, qc); rc != 0 {
		t.Fatalf("baseline must be GREEN, rc=%d:\n%s", rc, out)
	}
	mutate(t, qc, "core/core.go", "x * 2", "x * 2.0001")
	rc, out := runFx(t, qc)
	if rc != 1 {
		t.Fatalf("rc=%d after a shared-set mutation, want 1:\n%s", rc, out)
	}
	if !strings.Contains(out, "FORWARD GOLDENS") || !strings.Contains(out, "core/core.go ∈ shared set core") {
		t.Errorf("the shared-set trigger is not reported:\n%s", out)
	}
	if ln := cellLine(t, out, "core [goldens]"); !strings.Contains(ln, "FAIL") {
		t.Errorf("goldens cell must FAIL:\n  %s", ln)
	}
	if ln := cellLine(t, out, "core"); !strings.Contains(ln, "PASS") {
		t.Errorf("the rest of core is blind to the mutation and must pass (so the goldens are what caught it):\n  %s", ln)
	}
	if !strings.Contains(out, "goldens  0 passed, 0 skipped, 1 failed") {
		t.Errorf("verdict does not count the golden failure:\n%s", out)
	}
}

// A shared-set change whose goldens all SKIP is not green: no numeric proof ran.
func TestQuick_vacuousGoldensAreRed(t *testing.T) {
	qc := fxRepo(t)
	mutate(t, qc, "core/core_test.go", "func TestFx_forwardParity(t *testing.T) {\n", "func TestFx_forwardParity(t *testing.T) {\n\tt.Skip(\"no tiny fixture\")\n")
	mutate(t, qc, "core/core.go", "x * 2 }", "x * 2 } // touched")
	rc, out := runFx(t, qc)
	if rc != 1 || !strings.Contains(out, "goldens VACUOUS") {
		t.Fatalf("rc=%d; want 1 with 'goldens VACUOUS':\n%s", rc, out)
	}
}

// The goldens quick runs must be the set a deps_hash refresh is gated on.
func TestQuick_goldenRunMatchesRefreshScript(t *testing.T) {
	b, err := os.ReadFile("../../scripts/refresh_parity_hashes.sh")
	if err != nil {
		t.Skipf("no refresh script: %v", err)
	}
	m := regexp.MustCompile(`(?m)^GOLDEN_RE='([^']*)'$`).FindSubmatch(b)
	if m == nil {
		t.Fatal("GOLDEN_RE not found in scripts/refresh_parity_hashes.sh")
	}
	if string(m[1]) != goldenRunRE {
		t.Fatalf("goldenRunRE drifted from the refresh script:\n  quick:  %s\n  script: %s", goldenRunRE, m[1])
	}
}

// The split never loses a test: the main cell -skips exactly what the other cells -run.
func TestQuick_splitCellsPartitionThePackage(t *testing.T) {
	names := []string{"TestA_forwardParity", "TestWalk", "TestB", "TestC"}
	gold := regexp.MustCompile(goldenRunRE)
	walk := regexp.MustCompile(anchored([]string{"TestWalk"}))
	rest := regexp.MustCompile(goldenRunRE + "|" + anchored([]string{"TestWalk"}))
	for _, n := range names {
		in := 0
		if gold.MatchString(n) {
			in++
		}
		if walk.MatchString(n) && !gold.MatchString(n) {
			in++
		}
		if !rest.MatchString(n) {
			in++
		}
		if in != 1 {
			t.Errorf("%s runs in %d cells, want exactly 1", n, in)
		}
	}
}

// (3) THE PROPERTY, exhaustively on the real module graph: for every Go (and embedded) file that
// any test binary in the five modules compiles, the AFFECTED tier contains every test binary that
// an independent per-binary `go list -deps -test` says compiles it.
//
// Independence: quick builds one listing per module and inverts it (file -> owners -> Deps); the
// oracle lists each test binary on its own, with the tags `go test` uses (TestTags, not quick's
// superset ListTags), and reads the file lists straight out of each dependency. They share only the
// environment (workspace + target), which is what the modules are.
func TestQuick_neverDropsAGoListDependent(t *testing.T) {
	needTools(t)
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	qc, err := realQuickConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	qc.StateDir = t.TempDir()
	g, err := loadGraph(qc, "")
	if err != nil {
		t.Fatal(err)
	}
	type key struct{ mod, pkg string }
	oracle := map[string]map[key]bool{} // repo-relative file -> test binaries compiling it
	const tmpl = `{{.ImportPath}}|{{.ForTest}}|{{.Dir}}|{{if .Module}}{{.Module.Path}}{{end}}|` +
		`{{join .GoFiles ","}},{{join .CgoFiles ","}},{{join .EmbedFiles ","}}|` +
		`{{join .TestGoFiles ","}},{{join .XTestGoFiles ","}},{{join .TestEmbedFiles ","}},{{join .XTestEmbedFiles ","}}`
	local := map[string]bool{}
	for _, m := range qc.Modules {
		local[m.Path] = true
	}
	binaries := 0
	for _, m := range qc.Modules {
		env, err := qc.cmdEnv(m, m.listTarget(), "", false)
		if err != nil {
			t.Fatal(err)
		}
		// This module's packages that have tests, listed on their own (not via quick's listing).
		pk := exec.Command("go", "list", "-e", "-tags", strings.Join(m.TestTags, " "),
			"-f", `{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}`, "./...")
		pk.Dir, pk.Env = qc.modDir(m), env
		out, err := pk.Output()
		if err != nil {
			t.Fatalf("%s: go list: %v", m.Name, err)
		}
		for pkg := range strings.FieldsSeq(string(out)) {
			binaries++
			cmd := exec.Command("go", "list", "-e", "-deps", "-test", "-tags", strings.Join(m.TestTags, " "), "-f", tmpl, pkg)
			cmd.Dir, cmd.Env = qc.modDir(m), env
			b, err := cmd.Output()
			if err != nil {
				t.Fatalf("%s: go list -deps -test %s: %v", m.Name, pkg, err)
			}
			for ln := range strings.SplitSeq(strings.TrimSpace(string(b)), "\n") {
				f := strings.Split(ln, "|")
				if len(f) != 6 || !local[f[3]] {
					continue
				}
				base := stripVariant(f[0])
				files := f[4]
				if base == pkg || f[1] == pkg {
					files += "," + f[5] // the package under test and its external test package
				}
				for name := range strings.SplitSeq(files, ",") {
					// "" is an empty list; an absolute name is the generated _testmain.go in the
					// build cache, which no change to the tree can touch.
					if name == "" || filepath.IsAbs(name) {
						continue
					}
					rel, err := filepath.Rel(root, filepath.Join(f[2], name))
					if err != nil {
						t.Fatal(err)
					}
					rel = filepath.ToSlash(rel)
					if oracle[rel] == nil {
						oracle[rel] = map[key]bool{}
					}
					oracle[rel][key{m.Name, pkg}] = true
				}
			}
		}
	}
	if binaries < 20 || len(oracle) < 500 {
		t.Fatalf("oracle too small to mean anything: %d test binaries, %d files — the listing is broken, not the selection", binaries, len(oracle))
	}
	files := make([]string, 0, len(oracle))
	for f := range oracle {
		files = append(files, f)
	}
	sort.Strings(files)
	dropped := 0
	for _, rel := range files {
		_, _, hits := g.affectedBy(qc, rel)
		got := map[key]bool{}
		for _, h := range hits {
			got[key{h.root.Mod.Name, h.root.Pkg}] = true
		}
		for want := range oracle[rel] {
			if !got[want] {
				if dropped++; dropped <= 20 {
					t.Errorf("%s: go list -deps -test says %s [%s] compiles it; quick's AFFECTED tier dropped it", rel, want.pkg, want.mod)
				}
			}
		}
	}
	t.Logf("checked %d files against %d test binaries across %d modules: %d dropped", len(files), binaries, len(qc.Modules), dropped)
}

// ---- real-tree mutations, through -overlay (opt-in: they compile and run real packages) ----

func overlayOf(t *testing.T, root, rel, from, to string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, rel))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(from)) {
		t.Fatalf("mutation target %q not in %s — the mutation would be vacuous", from, rel)
	}
	dir := t.TempDir()
	mut := filepath.Join(dir, filepath.Base(rel))
	if err := os.WriteFile(mut, bytes.Replace(b, []byte(from), []byte(to), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	ov, _ := json.Marshal(map[string]map[string]string{"Replace": {filepath.Join(root, rel): mut}})
	p := filepath.Join(dir, "overlay.json")
	if err := os.WriteFile(p, ov, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func realTree(t *testing.T) *quickConfig {
	t.Helper()
	needTools(t)
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("real-tree mutation: set GOINFER_HEAVY_TESTS=1 (compiles and runs real packages, ~1-3 min)")
	}
	root, _ := filepath.Abs("../..")
	qc, err := realQuickConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	qc.StateDir = t.TempDir()
	return qc
}

func runReal(t *testing.T, qc *quickConfig, file, overlay string, keep func(*quickCell) bool) (int, string) {
	t.Helper()
	var out bytes.Buffer
	rc := quickMain(qc, quickOpts{Files: []string{file}, Overlay: overlay, Jobs: 4, Timeout: "15m",
		LogDir: t.TempDir(), NoLint: true, Progress: io.Discard, Keep: keep}, &out)
	return rc, out.String()
}

// The real-tree (1): internal/cliutil's OnOff flag stops being a bool flag.
func TestQuick_realLeafMutation(t *testing.T) {
	qc := realTree(t)
	file := "internal/cliutil/cliutil.go"
	ov := overlayOf(t, qc.Root, file, "func (f *OnOff) IsBoolFlag() bool { return true }", "func (f *OnOff) IsBoolFlag() bool { return false }")
	keep := func(c *quickCell) bool {
		return c.Tier == "affected" && strings.HasSuffix(c.Root.Pkg, "internal/cliutil")
	}
	rc, out := runReal(t, qc, file, ov, keep)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1:\n%s", rc, out)
	}
	if ln := cellLine(t, out, "internal/cliutil"); !strings.Contains(ln, "FAIL") || !strings.Contains(ln, "affected") {
		t.Fatalf("internal/cliutil must FAIL as affected:\n  %s", ln)
	}
}

// The real-tree (2): a 1-in-dim change to decoder's RMSNorm, a `core` shared-set file. Only the
// goldens cell is run, so this measures exactly "the forward goldens catch a core mutation".
func TestQuick_realCoreMutationFailsTheGoldens(t *testing.T) {
	qc := realTree(t)
	file := "decoder/rmsnorm.go"
	ov := overlayOf(t, qc.Root, file, "math.Sqrt(ss/float64(dim)+eps)", "math.Sqrt(ss/float64(dim+1)+eps)")
	rc, out := runReal(t, qc, file, ov, func(c *quickCell) bool { return c.Golden })
	if rc != 1 {
		t.Fatalf("rc=%d, want 1:\n%s", rc, out)
	}
	if ln := cellLine(t, out, "decoder [goldens]"); !strings.Contains(ln, "FAIL") {
		t.Fatalf("decoder goldens must FAIL:\n  %s", ln)
	}
	m := regexp.MustCompile(`goldens  (\d+) passed, (\d+) skipped, (\d+) failed`).FindStringSubmatch(out)
	if m == nil {
		t.Fatalf("no goldens verdict line:\n%s", out)
	}
	t.Logf("real goldens under the mutation: %s passed, %s skipped, %s failed", m[1], m[2], m[3])
	fmt.Fprintln(os.Stderr, "real goldens under the mutation:", m[0])
}

// ckFixture is a package whose tests reach a real checkpoint in each way this tree's tests do —
// a helper returning a relative path, a package-level table of bare file names, and a
// Join("..", "testdata", dir) — beside a tiny fixture, a production-code architecture name that
// matches the checkpoint directory, and enough plain tests that the loaders are split off.
func ckFixture(t *testing.T) (*quickConfig, *testRoot) {
	t.Helper()
	repo := t.TempDir()
	if r, err := filepath.EvalSymlinks(repo); err == nil {
		repo = r
	}
	big := func(rel string) {
		p := filepath.Join(repo, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		// Sparse: the size is what the detector reads, and no disk is spent on it.
		if err := f.Truncate(minCheckpointBytes); err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	big("testdata/big.gguf")
	big("testdata/table.gguf")
	big("testdata/bigdir/model.safetensors")
	big("testdata/registered.gguf")
	reg := `{"assets": [{"env": "GOINFER_CK_TEST_ASSET", "kind": "file", "candidates": ["$REPO/testdata/registered.gguf"]},
		{"env": "GOINFER_CK_TEST_TINY", "kind": "file", "candidates": ["$REPO/testdata/tiny.gguf"]}]}`
	if err := os.WriteFile(filepath.Join(repo, "testdata", "assets.json"), []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "testdata", "tiny.gguf"), []byte("tiny"), 0o644); err != nil {
		t.Fatal(err)
	}
	var plain strings.Builder
	for i := range minSplitRest {
		fmt.Fprintf(&plain, "func TestPlain%02d(t *testing.T) {}\n", i)
	}
	files := map[string]string{
		"go.mod":     "module example.com/ck\n\ngo 1.21\n",
		"ck/arch.go": "package ck\n\n// Production code: \"bigdir\" here is an architecture name, reached by every test.\nfunc Arch(s string) bool {\n\tswitch s {\n\tcase \"bigdir\":\n\t\treturn true\n\t}\n\treturn false\n}\n",
		"ck/ck_test.go": "package ck\n\nimport (\n\t\"path/filepath\"\n\t\"testing\"\n)\n\nfunc bigPath() string { return \"../testdata/big.gguf\" }\n\nvar models = []string{\"table.gguf\"}\n\nfunc TestHelper(t *testing.T) { _ = bigPath() }\n\nfunc TestTable(t *testing.T) {\n\tfor _, m := range models {\n\t\t_ = m\n\t}\n}\n\nfunc TestDir(t *testing.T) { _ = filepath.Join(\"..\", \"testdata\", \"bigdir\") }\n\nfunc TestTiny(t *testing.T) { _ = \"../testdata/tiny.gguf\"; _ = Arch(\"x\"); _ = \"GOINFER_CK_TEST_TINY\" }\n\n" +
			"func asset(key string) string { return key }\n\nfunc TestRegistry(t *testing.T) { _ = asset(\"GOINFER_CK_TEST_ASSET\") }\n\n" + plain.String(),
	}
	for name, body := range files {
		p := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	qc := &quickConfig{Root: repo, Modules: []*quickModule{{Name: "root", Dir: "."}}, GoldenPkg: "example.com/ck/none",
		GoldenRun: goldenRunRE, StateDir: t.TempDir()}
	if err := qc.finish(); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(repo, "ck")
	r := &testRoot{Mod: qc.Modules[0], Pkg: "example.com/ck/ck", Dir: dir, repo: repo,
		p: &qPkg{Path: "example.com/ck/ck", Dir: dir, Mod: qc.Modules[0],
			Files: []string{filepath.Join(dir, "arch.go"), filepath.Join(dir, "ck_test.go")}}}
	return qc, r
}

// The checkpoint loaders are found through a helper, a package-level table, a directory join and an
// asset-registry key; a tiny fixture (by path or by registry key) and a production-code name that
// happens to match the directory are not loaders.
// They get a [checkpoint] cell of their own, marked Mem, and the split still runs every test
// exactly once.
func TestQuick_checkpointLoadersGetTheirOwnCell(t *testing.T) {
	qc, r := ckFixture(t)
	g := &quickGraph{roots: []*testRoot{r}}
	cells := buildCells(qc, g, &quickSelection{}, false, false)
	sort.Strings(r.loaders)
	if want := []string{"TestDir", "TestHelper", "TestRegistry", "TestTable"}; strings.Join(r.loaders, ",") != strings.Join(want, ",") {
		t.Fatalf("loaders %v, want %v (why: %v)", r.loaders, want, r.loadWhy)
	}
	var ck, rest *quickCell
	for _, c := range cells {
		switch c.Name {
		case "ck [checkpoint]":
			ck = c
		case "ck":
			rest = c
		}
	}
	if ck == nil || rest == nil || len(cells) != 2 {
		t.Fatalf("want a [checkpoint] cell and the rest, got %d cells", len(cells))
	}
	if !ck.Mem || rest.Mem {
		t.Fatalf("Mem: [checkpoint] %v (want true), rest %v (want false)", ck.Mem, rest.Mem)
	}
	run, skip := regexp.MustCompile(ck.Run), regexp.MustCompile(rest.Skip)
	for _, n := range r.tests {
		in := 0
		if run.MatchString(n) {
			in++
		}
		if !skip.MatchString(n) {
			in++
		}
		if in != 1 {
			t.Errorf("%s runs in %d cells, want exactly 1", n, in)
		}
	}
}

// A package too small to split runs whole, and the whole cell takes the run-alone rule.
func TestQuick_smallCheckpointPackageRunsWholeInTheLane(t *testing.T) {
	qc, r := ckFixture(t)
	b, err := os.ReadFile(r.p.Files[1])
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	src = src[:strings.Index(src, "func TestPlain00")]
	if err := os.WriteFile(r.p.Files[1], []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	cells := buildCells(qc, &quickGraph{roots: []*testRoot{r}}, &quickSelection{}, false, false)
	if len(cells) != 1 || !cells[0].Mem || cells[0].Skip != "" {
		t.Fatalf("want one unsplit Mem cell, got %d cells (first: Mem %v, Skip %q)", len(cells), cells[0].Mem, cells[0].Skip)
	}
}

// THE RULE, through the scheduler itself: with four slots, a Mem job overlaps no other job at all,
// and the jobs that are not Mem still run beside each other (the rule costs no parallelism there).
func TestQuick_memJobsRunAlone(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
	dir := t.TempDir()
	mk := func(name string, mem bool) *quickJob {
		log := filepath.Join(dir, name)
		// Each job records its own start and end, in nanoseconds, from its own process.
		script := `python3 -c 'import time; print(time.time_ns())' > "$1"; sleep 0.4; python3 -c 'import time; print(time.time_ns())' >> "$1"`
		return &quickJob{Name: name, Kind: "lint", Dir: dir, Env: os.Environ(), Mem: mem,
			Args: []string{"sh", "-c", script, "sh", log}}
	}
	// A Mem job first in the queue as well as among the others: the start of a run is when nothing
	// is running yet, which is where a check on "another Mem job" alone would let one slip in.
	jobs := []*quickJob{mk("m1", true), mk("f1", false), mk("m2", true), mk("f2", false), mk("f3", false), mk("m3", true)}
	runJobs(jobs, 4, t.TempDir(), io.Discard, 0, time.Now())
	span := map[string][2]int64{}
	for _, jb := range jobs {
		if !jb.ok() {
			t.Fatalf("%s: rc %d %v\n%s", jb.Name, jb.RC, jb.Err, jb.Out)
		}
		b, err := os.ReadFile(filepath.Join(dir, jb.Name))
		if err != nil {
			t.Fatal(err)
		}
		f := strings.Fields(string(b))
		var st, en int64
		fmt.Sscan(f[0], &st)
		fmt.Sscan(f[1], &en)
		span[jb.Name] = [2]int64{st, en}
	}
	overlap := func(a, b [2]int64) bool { return a[0] < b[1] && b[0] < a[1] }
	for _, m := range []string{"m1", "m2", "m3"} {
		for _, o := range []string{"m1", "m2", "m3", "f1", "f2", "f3"} {
			if m != o && overlap(span[m], span[o]) {
				t.Errorf("Mem job %s overlapped %s", m, o)
			}
		}
	}
	if !overlap(span["f1"], span["f2"]) && !overlap(span["f1"], span["f3"]) && !overlap(span["f2"], span["f3"]) {
		t.Error("no two non-Mem jobs ran together: the rule serialised more than the loaders")
	}
}
