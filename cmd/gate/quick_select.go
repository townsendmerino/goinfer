package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// `gate quick`'s SELECTION (TE7(a), docs/tasks/task-test-efficiency-2026-09.md): changed files ->
// their packages -> every test binary, in any of the five modules, that can observe them.
//
// THREE TIERS, because "imports the change" is not the only way a test here observes one:
//
//   - AFFECTED: the test binary compiles a changed file (the file's package is in its
//     `go list -deps -test` closure, or it embeds the file). Its binary is new, so it re-runs.
//     This is the tier the equivalence property pins (quick_test.go): it never drops a package
//     `go list -deps -test` says depends on the change.
//   - OBSERVER: the tests can READ the change as data. Measured 2026-09-28, not assumed: three
//     decoder tests (TestEnvVars_docAndCodeAgree, TestBackendBanner_usesTheReport,
//     TestGoldenNames_matchTheFileOnDisk) open 4,883 files across the whole tree — metal/, cuda/,
//     gpu/, internal/, docs/, even the gitignored vendor/ and .claude/ — so an edit to a metal
//     .go file is observable by decoder's tests although decoder imports nothing in metal. An
//     import-graph selection alone would report GREEN on a change CI then fails.
//   - CACHE-CHECKED: every other test binary. A test can open a path it builds at run time, which
//     no static scan sees, so nothing is dropped: these run WITHOUT -count=1 and Go's test cache —
//     which records every file, directory listing and env var a test binary actually read —
//     replays each one whose recorded inputs did not change (TE7(b): decoder replays in 2 s).
//     That is sound for reads inside the test's own module root, and the cache's one blind spot
//     (cmd/go does not re-check files OUTSIDE the module root) is closed where determinable by
//     forcing -count=1 on a cross-module reader (see crossModuleReader).

// quickModule is one of the tree's Go modules as `gate quick` lists, vets and tests it. Each field
// mirrors how CI treats that module (.github/workflows/ci.yml), because a day-loop check that
// disagrees with CI about tags or workspace is a second definition of "green".
type quickModule struct {
	Name string // display name: root, gpu, cuda, metal, demo/agent
	Dir  string // repo-relative; "." for the root
	Path string // module path, read from its go.mod
	// Work lists the workspace members (repo-relative) this module resolves in. Empty means
	// GOWORK=off, which is how CI builds the root: its checkout has no go.work.
	Work []string
	// ListTags builds the selection graph. It must be a SUPERSET of TestTags (tags that only ADD
	// files), so a dependency that exists only under a heavier tag still selects — over-selection
	// is a cost, under-selection is a wrong verdict.
	ListTags []string
	TestTags []string
	TestArgs []string // e.g. -short, exactly as CI passes it
	Env      map[string]string
	// NativeGOOS is the only GOOS this module builds on ("" = any). Elsewhere its graph and lint
	// run for CrossTarget, and its tests are reported NOT RUN — counted, never dropped.
	NativeGOOS  string
	CrossTarget string
	// Device: the tests drive a GPU, so at most one device test process runs at a time. Two
	// processes contending for Metal is a known crash source (the fault 0x10 tail), and a red
	// caused by the scheduler is a red nobody should have to diagnose.
	Device bool
	Lint   []lintSpec
}

func (m *quickModule) native() bool { return m.NativeGOOS == "" || m.NativeGOOS == runtime.GOOS }

// listTarget is the GOOS/GOARCH this module's graph is listed for: native where it builds, else
// the target CI builds it for.
func (m *quickModule) listTarget() string {
	if m.native() {
		return ""
	}
	return m.CrossTarget
}

// lintSpec is one vet / build / staticcheck invocation. Mirrors names the CI step it reproduces,
// printed with the result, so a drift between this list and ci.yml is visible in the output.
type lintSpec struct {
	Tool    string   // "build" | "vet" | "staticcheck"
	Tags    []string // build tags
	Target  string   // "" = this host; else "GOOS/GOARCH", cross-built with CGO_ENABLED=0
	Scope   string   // "affected" (changed packages + everything importing them) | "changed"
	Mirrors string
}

// quickConfig is everything `gate quick` needs to know about a tree. The real one comes from
// realQuickConfig; the equivalence tests build one over a scratch fixture.
type quickConfig struct {
	Root      string // absolute repo root
	Modules   []*quickModule
	GoVersion string // the go line for the generated workspace files
	GoldenPkg string // import path holding the forward goldens
	GoldenRun string // -run regex selecting them
	Manifest  string // repo-relative parity manifest ("" = none)
	// StateDir holds the generated go.work files and the per-cell duration history. It is OUTSIDE
	// the repo on purpose: a file written inside it would change the directory listings the
	// tree-walking tests read, and invalidate their cached results on every run.
	StateDir string
}

// goldenRunRE is scripts/refresh_parity_hashes.sh's GOLDEN_RE, verbatim: the forward-numeric
// goldens, which compare a forward pass to a committed golden, so a numeric change breaks them.
// Kept identical so "the goldens quick ran" and "the goldens a deps_hash refresh is gated on" are
// the same set; TestQuick_goldenRunMatchesRefreshScript pins that.
const goldenRunRE = `(_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$`

func realQuickConfig(root string) (*quickConfig, error) {
	short := sha256.Sum256([]byte(root))
	cache, err := os.UserCacheDir()
	if err != nil {
		cache = os.TempDir()
	}
	qc := &quickConfig{
		Root:      root,
		GoldenRun: goldenRunRE,
		Manifest:  "testdata/parity_manifest.json",
		StateDir:  filepath.Join(cache, "goinfer-gate-quick", hex.EncodeToString(short[:6])),
	}
	testhooks := []string{"goinfer_testhooks"}
	qc.Modules = []*quickModule{
		{
			Name: "root", Dir: ".",
			// realckpt only ADDS files (no `!realckpt` constraint exists in the tree), so listing
			// with it is a superset of what the tests compile.
			ListTags: []string{"goinfer_testhooks", "realckpt"},
			TestTags: testhooks,
			Lint: []lintSpec{
				{"build", nil, "", "affected", "root-darwin / lint: go build ./..."},
				{"vet", testhooks, "", "affected", "root-darwin: go vet -tags goinfer_testhooks ./..."},
				{"vet", []string{"realckpt", "goinfer_testhooks"}, "linux/amd64", "affected",
					"lint: go vet -tags 'realckpt goinfer_testhooks' ./... (a superset of lint's plain vet)"},
				// linux/amd64, not this host: arm64-only symbols trip the pure-Go `unused` check,
				// and CI analyses the linux target.
				{"staticcheck", nil, "linux/amd64", "changed", "lint: staticcheck ./..."},
			},
		},
		{
			Name: "gpu", Dir: "gpu", Work: []string{".", "gpu"},
			ListTags: []string{"gpu", "goinfer_testhooks"}, TestTags: []string{"gpu", "goinfer_testhooks"},
			TestArgs: []string{"-short"}, Device: true,
			Lint: []lintSpec{
				{"build", []string{"gpu", "goinfer_testhooks"}, "", "affected", "gpu-darwin: go build -tags 'gpu goinfer_testhooks' ./gpu/..."},
				{"vet", []string{"gpu", "goinfer_testhooks"}, "", "affected", "gpu-darwin: go vet -tags 'gpu goinfer_testhooks' ./gpu/..."},
				// CI runs this one on linux. WebGPU is cgo, and cgo does not cross-compile from
				// here (CGO_ENABLED=0 leaves webgpu's types undefined, measured), so this is the
				// host's analysis of the same packages.
				{"staticcheck", []string{"gpu", "goinfer_testhooks"}, "", "changed", "gpu: staticcheck -tags 'gpu goinfer_testhooks' ./gpu/... (host target)"},
			},
		},
		{
			Name: "cuda", Dir: "cuda", Work: []string{".", "cuda"},
			ListTags: []string{"cuda", "goinfer_testhooks"}, TestTags: []string{"cuda", "goinfer_testhooks"},
			TestArgs: []string{"-short"}, Env: map[string]string{"CGO_ENABLED": "0"},
			// Measured 2026-09-28 on darwin/arm64: `CGO_ENABLED=0 go build -tags 'cuda
			// goinfer_testhooks' ./cuda/...` fails (undefined: gpu.MappedHostBuffer, gpu.Graph,
			// gpu.Event — aikit/gpu's CUDA surface is linux-only), while the same vet for
			// GOOS=linux GOARCH=amd64 passes. So it is linted cross-target and its tests are NOT
			// RUN off linux.
			NativeGOOS: "linux", CrossTarget: "linux/amd64", Device: true,
			Lint: []lintSpec{
				{"build", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "affected", "cuda: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./cuda/..."},
				{"vet", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "affected", "cuda: go vet -tags 'cuda goinfer_testhooks' ./cuda/..."},
				{"staticcheck", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "changed", "cuda: staticcheck -tags 'cuda goinfer_testhooks' ./cuda/..."},
			},
		},
		{
			Name: "metal", Dir: "metal", Work: []string{".", "metal"},
			ListTags: testhooks, TestTags: testhooks,
			NativeGOOS: "darwin", CrossTarget: "darwin/arm64", Device: true,
			Lint: []lintSpec{
				{"build", nil, "", "affected", "metal-darwin: go build ./metal/..."},
				{"vet", nil, "", "affected", "metal-darwin: go vet ./metal/..."},
				{"vet", testhooks, "", "affected", "metal-darwin: go vet -tags goinfer_testhooks ./metal/..."},
				// Not a CI step. Run because a local staticcheck is the only one this module gets;
				// clean at 2026-09-28, so a red here is new.
				{"staticcheck", testhooks, "", "changed", "(not in CI) staticcheck -tags goinfer_testhooks ./metal/..."},
			},
		},
		{
			Name: "demo/agent", Dir: "demo/agent", Work: []string{".", "gpu", "demo/agent"},
			Lint: []lintSpec{
				{"build", nil, "", "affected", "gpu: go build ./demo/agent/..."},
				{"vet", nil, "", "affected", "(not in CI) go vet ./demo/agent/..."},
			},
		},
	}
	if err := qc.finish(); err != nil {
		return nil, err
	}
	qc.GoldenPkg = qc.Modules[0].Path + "/decoder"
	return qc, nil
}

// finish reads each module's path and the go version from the go.mod files, so neither is typed
// twice.
func (qc *quickConfig) finish() error {
	for _, m := range qc.Modules {
		b, err := os.ReadFile(filepath.Join(qc.Root, m.Dir, "go.mod"))
		if err != nil {
			return fmt.Errorf("module %s: %w", m.Name, err)
		}
		for _, ln := range strings.Split(string(b), "\n") {
			f := strings.Fields(ln)
			if len(f) == 2 && f[0] == "module" {
				m.Path = f[1]
			}
			if len(f) == 2 && f[0] == "go" && m.Dir == "." {
				qc.GoVersion = f[1]
			}
		}
		if m.Path == "" {
			return fmt.Errorf("module %s: no module line in go.mod", m.Name)
		}
	}
	if qc.GoVersion == "" {
		qc.GoVersion = strings.TrimPrefix(runtime.Version(), "go")
	}
	return nil
}

// moduleOf returns the module a repo-relative path belongs to: the longest module dir containing it.
func (qc *quickConfig) moduleOf(rel string) *quickModule {
	var best *quickModule
	for _, m := range qc.Modules {
		if m.Dir == "." || rel == m.Dir || strings.HasPrefix(rel, m.Dir+"/") {
			if best == nil || len(m.Dir) > len(best.Dir) || best.Dir == "." {
				best = m
			}
		}
	}
	return best
}

func (qc *quickConfig) modDir(m *quickModule) string { return filepath.Join(qc.Root, m.Dir) }

// workFile writes (if its content changed) the workspace file module m resolves in and returns its
// path, or "off". An EPHEMERAL workspace per module is what CI does (`go work init . ./gpu`); the
// developer's own go.work unions root+gpu+metal, which MVS-merges requirements CI never merges.
func (qc *quickConfig) workFile(m *quickModule) (string, error) {
	if len(m.Work) == 0 {
		return "off", nil
	}
	var b strings.Builder
	fmt.Fprintf(&b, "go %s\n\nuse (\n", qc.GoVersion)
	for _, d := range m.Work {
		fmt.Fprintf(&b, "\t%s\n", filepath.Join(qc.Root, d))
	}
	b.WriteString(")\n")
	if err := os.MkdirAll(qc.StateDir, 0o755); err != nil {
		return "", err
	}
	p := filepath.Join(qc.StateDir, sanitize(m.Name)+".go.work")
	if old, err := os.ReadFile(p); err != nil || string(old) != b.String() {
		if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
			return "", err
		}
	}
	return p, nil
}

// cmdEnv is the environment for a go command about module m, built for target ("" = this host).
// overlay, when set, is passed to every go command (the equivalence tests mutate through it, so no
// real file is ever edited).
//
// forTest builds a TEST cell's environment, and differs in two ways, both measured on this Mac:
//
//   - GOWORK is left to auto-discovery wherever the repo's own go.work (or its absence, for the
//     root) already resolves the module the way its CI job does. An explicit GOWORK is inherited by
//     every `go` a test spawns: gpu's TestParentSelfSkipsWhenNoSubtestRan runs `go test` in a scratch
//     module and fails with "setup failed" under an explicit workspace, and decoder's
//     TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams runs `go doc`, which under GOWORK=off hits
//     the stale vendor/ below. Both pass under a hand-typed `go test`, so the harness must not
//     be what turns them red.
//   - GOINFER_HEAVY_TESTS is removed: a shell that exported it would turn the day loop into the
//     90-minute heavy tier.
func (qc *quickConfig) cmdEnv(m *quickModule, target, overlay string, forTest bool) ([]string, error) {
	set := map[string]string{}
	auto := forTest && qc.repoWorkCovers(m)
	moduleMode := false
	if auto {
		moduleMode = !fileExists(filepath.Join(qc.Root, "go.work"))
	} else {
		work, err := qc.workFile(m)
		if err != nil {
			return nil, err
		}
		set["GOWORK"] = work
		moduleMode = work == "off"
	}
	for k, v := range m.Env {
		set[k] = v
	}
	if target != "" && target != runtime.GOOS+"/"+runtime.GOARCH {
		goos, goarch, _ := strings.Cut(target, "/")
		set["GOOS"], set["GOARCH"], set["CGO_ENABLED"] = goos, goarch, "0"
	}
	var flags []string
	if v := os.Getenv("GOFLAGS"); v != "" {
		flags = append(flags, v)
	}
	if moduleMode && fileExists(filepath.Join(qc.Root, m.Dir, "vendor", "modules.txt")) {
		// A gitignored, stale vendor/ (this Mac's root has one from 2026-09-20: aikit v1.46.0
		// against go.mod's v1.50.1) puts module-mode builds in vendor mode and fails them with
		// "inconsistent vendoring". CI's checkout has no vendor/, so -mod=readonly is what CI runs.
		flags = append(flags, "-mod=readonly")
	}
	if overlay != "" {
		flags = append(flags, "-overlay="+overlay)
	}
	if len(flags) > 0 {
		set["GOFLAGS"] = strings.Join(flags, " ")
	}
	env := make([]string, 0, len(os.Environ())+len(set))
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if _, ours := set[k]; ours || k == "GOWORK" || (forTest && k == "GOINFER_HEAVY_TESTS") {
			continue
		}
		env = append(env, kv)
	}
	for k, v := range set {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	return env, nil
}

// repoWorkCovers reports whether plain auto-discovery resolves module m with every workspace
// member its CI job uses: the repo's go.work lists them all (the root also qualifies when there is
// no go.work at all — module mode is exactly CI's GOWORK=off). The developer's go.work is gitignored
// and unions root+gpu+metal; CI's per-job workspaces are subsets of that with the same requirements.
func (qc *quickConfig) repoWorkCovers(m *quickModule) bool {
	b, err := os.ReadFile(filepath.Join(qc.Root, "go.work"))
	if err != nil {
		return len(m.Work) == 0
	}
	uses := map[string]bool{}
	inUse := false
	for _, ln := range strings.Split(string(b), "\n") {
		f := strings.Fields(strings.TrimSpace(ln))
		switch {
		case len(f) == 0:
		case f[0] == "use" && len(f) > 1 && f[1] == "(":
			inUse = true
		case f[0] == "use" && len(f) > 1:
			uses[filepath.Clean(f[1])] = true
		case inUse && f[0] == ")":
			inUse = false
		case inUse:
			uses[filepath.Clean(f[0])] = true
		}
	}
	need := m.Work
	if len(need) == 0 {
		need = []string{"."}
	}
	for _, d := range need {
		if !uses[filepath.Clean(d)] {
			return false
		}
	}
	return true
}

// ---- the graph ----

type goListPkg struct {
	ImportPath string
	Dir        string
	ForTest    string
	Module     *struct{ Path, Dir string }
	Deps       []string

	GoFiles, CgoFiles, IgnoredGoFiles, TestGoFiles, XTestGoFiles []string
	SFiles, CFiles, HFiles, CXXFiles, MFiles, SysoFiles          []string
	EmbedFiles, TestEmbedFiles, XTestEmbedFiles                  []string

	Error *struct{ Err string }
}

const goListFields = "ImportPath,Dir,ForTest,Module,Deps,GoFiles,CgoFiles,IgnoredGoFiles,TestGoFiles,XTestGoFiles," +
	"SFiles,CFiles,HFiles,CXXFiles,MFiles,SysoFiles,EmbedFiles,TestEmbedFiles,XTestEmbedFiles,Error"

// qPkg is one package of the five modules.
type qPkg struct {
	Path string
	Dir  string // absolute
	Mod  *quickModule
	Deps map[string]bool // non-test transitive deps, from the owning module's listing
	// Files are this package's own Go files (non-test and test), absolute.
	Files     []string
	TestFiles []string
}

// testRoot is one test binary: `go test <Pkg>` in module Mod.
type testRoot struct {
	Mod  *quickModule
	Pkg  string
	Dir  string
	Deps map[string]bool // every package the test binary links, the package itself included
	p    *qPkg

	scanned  bool
	src      string   // the package's Go sources, concatenated, for the "names it" scan
	srcs     []string // the same, per file (parallel to p.Files)
	tests    []string // top-level Test functions
	walkers  []string // the ones that walk the parent tree (split into their own cell)
	walkSite string   // where the package enumerates a directory ("" = nowhere)
	lits     []string // every string literal in the package's sources, unquoted
	litSet   map[string]bool
}

type fileOwner struct {
	Pkg  string
	Test bool // compiled only into the package's own test binary
}

type quickGraph struct {
	pkgs     map[string]*qPkg
	roots    []*testRoot
	owners   map[string][]fileOwner // absolute file -> packages compiling it
	dirPkgs  map[string][]string    // absolute dir -> packages in it
	warnings []string
}

func stripVariant(ip string) string {
	if i := strings.Index(ip, " ["); i >= 0 {
		return ip[:i]
	}
	return ip
}

// loadGraph runs `go list -e -deps -test -json` once per module, with that module's tags,
// workspace and target, and merges the five listings into one graph.
func loadGraph(qc *quickConfig, overlay string) (*quickGraph, error) {
	g := &quickGraph{pkgs: map[string]*qPkg{}, owners: map[string][]fileOwner{}, dirPkgs: map[string][]string{}}
	local := map[string]*quickModule{}
	for _, m := range qc.Modules {
		local[m.Path] = m
	}
	type rootEntry struct {
		m    *quickModule
		base string
		deps []string
	}
	var rootEntries []rootEntry
	for _, m := range qc.Modules {
		env, err := qc.cmdEnv(m, m.listTarget(), overlay, false)
		if err != nil {
			return nil, err
		}
		args := []string{"list", "-e", "-deps", "-test", "-json=" + goListFields}
		if len(m.ListTags) > 0 {
			args = append(args, "-tags", strings.Join(m.ListTags, " "))
		}
		args = append(args, "./...")
		cmd := exec.Command("go", args...)
		cmd.Dir, cmd.Env = qc.modDir(m), env
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("go list in %s (%s): %v\n%s", m.Name, m.Dir, err, strings.TrimSpace(stderr.String()))
		}
		dec := json.NewDecoder(bytes.NewReader(out))
		for {
			var p goListPkg
			if err := dec.Decode(&p); err == io.EOF {
				break
			} else if err != nil {
				return nil, fmt.Errorf("go list in %s: decoding: %w", m.Name, err)
			}
			base := stripVariant(p.ImportPath)
			if p.Module == nil || local[p.Module.Path] == nil {
				continue
			}
			if strings.HasSuffix(base, ".test") && p.ForTest == "" && !strings.Contains(p.ImportPath, " [") {
				pkg := strings.TrimSuffix(base, ".test")
				if local[p.Module.Path] == m {
					rootEntries = append(rootEntries, rootEntry{m, pkg, p.Deps})
				}
				continue
			}
			// The plain variant, from the listing of the module that OWNS it: its file lists and
			// non-test deps are then the ones its own tags produce.
			if p.ImportPath != base || p.ForTest != "" || local[p.Module.Path] != m {
				continue
			}
			if p.Error != nil {
				g.warnings = append(g.warnings, fmt.Sprintf("%s: %s", base, strings.TrimSpace(p.Error.Err)))
			}
			q := &qPkg{Path: base, Dir: p.Dir, Mod: m, Deps: map[string]bool{}}
			for _, d := range p.Deps {
				q.Deps[stripVariant(d)] = true
			}
			add := func(names []string, test bool) {
				for _, n := range names {
					abs := filepath.Join(p.Dir, n)
					t := test || strings.HasSuffix(n, "_test.go")
					g.owners[abs] = append(g.owners[abs], fileOwner{base, t})
					if strings.HasSuffix(n, ".go") {
						if t {
							q.TestFiles = append(q.TestFiles, abs)
						}
						q.Files = append(q.Files, abs)
					}
				}
			}
			add(p.GoFiles, false)
			add(p.CgoFiles, false)
			add(p.IgnoredGoFiles, false) // compiled under some other tag or GOOS: conservative
			for _, fl := range [][]string{p.SFiles, p.CFiles, p.HFiles, p.CXXFiles, p.MFiles, p.SysoFiles, p.EmbedFiles} {
				add(fl, false)
			}
			add(p.TestGoFiles, true)
			add(p.XTestGoFiles, true)
			add(p.TestEmbedFiles, true)
			add(p.XTestEmbedFiles, true)
			g.pkgs[base] = q
			g.dirPkgs[p.Dir] = append(g.dirPkgs[p.Dir], base)
		}
	}
	for _, re := range rootEntries {
		q := g.pkgs[re.base]
		if q == nil {
			continue
		}
		r := &testRoot{Mod: re.m, Pkg: re.base, Dir: q.Dir, Deps: map[string]bool{re.base: true}, p: q}
		for _, d := range re.deps {
			r.Deps[stripVariant(d)] = true
		}
		g.roots = append(g.roots, r)
	}
	sort.Slice(g.roots, func(i, j int) bool { return g.roots[i].Pkg < g.roots[j].Pkg })
	return g, nil
}

// ---- static scan of a package's sources ----

// enumerators are the calls through which a test lists a directory rather than naming a file.
// Package-qualified, so a method that happens to be called ReadDir on some unrelated type does not
// count; Readdir/Readdirnames exist only on *os.File, so those match on any receiver.
var enumerators = map[string]map[string]bool{
	"filepath": {"WalkDir": true, "Walk": true, "Glob": true},
	"fs":       {"WalkDir": true, "Glob": true, "ReadDir": true},
	"os":       {"ReadDir": true, "DirFS": true},
	"ioutil":   {"ReadDir": true},
}

// scan parses the package once: its source text (for "names it"), its string literals (for
// crossModuleReader), its top-level tests, and which of them enumerate directories. The split that
// uses the last is a CACHE optimisation and never a correctness input — both halves of a split
// package run, and each half's cache is keyed on what that half actually read — so a heuristic
// miss costs time, not coverage.
func (r *testRoot) scan() {
	if r.scanned {
		return
	}
	r.scanned = true
	var src strings.Builder
	fset := token.NewFileSet()
	type fn struct {
		direct bool   // enumerates any directory
		site   string //
		tree   bool   // enumerates the PARENT tree: the repo, not its own package or testdata
		calls  []string
	}
	funcs := map[string]*fn{}
	var tests []string
	r.srcs = make([]string, len(r.p.Files))
	type parsed struct {
		file   *ast.File
		isTest bool
	}
	var files []parsed
	isParent := func(lit *ast.BasicLit) bool { return lit.Value == `".."` || lit.Value == `"../.."` }
	// Package-level names bound to the parent: decoder/assets.go's `const repoRoot = ".."`, which
	// every tree walk in decoder passes by name.
	parentNames := map[string]bool{}
	for i, f := range r.p.Files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		r.srcs[i] = string(b)
		src.Write(b)
		src.WriteByte('\n')
		file, err := parser.ParseFile(fset, f, b, parser.SkipObjectResolution)
		if err != nil {
			continue
		}
		files = append(files, parsed{file, strings.HasSuffix(f, "_test.go")})
		for _, d := range file.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, sp := range gd.Specs {
				vs, ok := sp.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for k, v := range vs.Values {
					if lit, ok := v.(*ast.BasicLit); ok && isParent(lit) && k < len(vs.Names) {
						parentNames[vs.Names[k].Name] = true
					}
				}
			}
		}
	}
	// reachesParent: does this expression mention the parent — a ".." literal, or a name bound to one?
	reachesParent := func(n ast.Node) bool {
		hit := false
		ast.Inspect(n, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BasicLit:
				hit = hit || isParent(x)
			case *ast.Ident:
				hit = hit || parentNames[x.Name]
			}
			return !hit
		})
		return hit
	}
	for _, pf := range files {
		file, isTest := pf.file, pf.isTest
		ast.Inspect(file, func(n ast.Node) bool {
			if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if v, err := strconv.Unquote(lit.Value); err == nil && len(v) < 512 {
					r.lits = append(r.lits, v)
				}
			}
			return true
		})
		for _, d := range file.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			name := fd.Name.Name
			if fd.Recv != nil {
				name = "(method)" + name // methods take no part in the call graph below
			}
			e := &fn{}
			// The function reaches the parent itself: root := ".." / filepath.Abs("..") / repoRoot.
			parentLit := reachesParent(fd.Body)
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch fun := call.Fun.(type) {
				case *ast.Ident:
					e.calls = append(e.calls, fun.Name)
				case *ast.SelectorExpr:
					x, _ := fun.X.(*ast.Ident)
					hit := fun.Sel.Name == "Readdir" || fun.Sel.Name == "Readdirnames" ||
						(x != nil && enumerators[x.Name][fun.Sel.Name])
					// `git ls-files` / `git grep` read the tree as surely as a WalkDir.
					if !hit && x != nil && x.Name == "exec" && strings.HasPrefix(fun.Sel.Name, "Command") {
						for _, a := range call.Args {
							if lit, ok := a.(*ast.BasicLit); ok && lit.Kind == token.STRING && lit.Value == `"git"` {
								hit = true
							}
						}
					}
					if hit && !e.direct {
						e.direct = true
						pos := fset.Position(call.Pos())
						e.site = fmt.Sprintf("%s:%d", filepath.Base(pos.Filename), pos.Line)
					}
					if hit && len(call.Args) > 0 {
						// The walked root reaches the parent directly (WalkDir(repoRoot, …),
						// Glob(filepath.Join("..", "*.go"))), or is a local the function bound to it.
						_, local := call.Args[0].(*ast.Ident)
						e.tree = e.tree || reachesParent(call.Args[0]) || (local && parentLit)
					}
				}
				return true
			})
			if prev, dup := funcs[name]; dup { // same name under two build tags: union them
				prev.direct = prev.direct || e.direct
				prev.tree = prev.tree || e.tree
				if prev.site == "" {
					prev.site = e.site
				}
				prev.calls = append(prev.calls, e.calls...)
			} else {
				funcs[name] = e
			}
			if isTest && fd.Recv == nil && isTestFunc(fd.Name.Name) {
				tests = append(tests, fd.Name.Name)
			}
		}
	}
	// Fixed point: a function walks if it enumerates directly or calls a package-local one that does.
	for changed := true; changed; {
		changed = false
		for _, e := range funcs {
			for _, c := range e.calls {
				callee := funcs[c]
				if callee == nil {
					continue
				}
				if callee.direct && !e.direct {
					e.direct, e.site, changed = true, callee.site, true
				}
				if callee.tree && !e.tree {
					e.tree, changed = true, true
				}
			}
		}
	}
	r.src = src.String()
	sort.Strings(tests)
	r.tests = dedupe(tests)
	for _, t := range r.tests {
		// Only the tree walkers split off. A fixture helper that globs ../testdata is called by
		// hundreds of decoder tests (measured: counting every enumerator put 337 of them in the
		// "small" cell), and those re-run only when a fixture moves, which the rest cell's cache
		// already handles.
		if e := funcs[t]; e != nil && e.tree {
			r.walkers = append(r.walkers, t)
		}
	}
	for _, e := range funcs {
		if e.direct && (r.walkSite == "" || e.site < r.walkSite) {
			r.walkSite = e.site
		}
	}
}

func isTestFunc(name string) bool {
	if !strings.HasPrefix(name, "Test") || name == "TestMain" {
		return false
	}
	rest := name[len("Test"):]
	return rest == "" || !(rest[0] >= 'a' && rest[0] <= 'z')
}

func dedupe(xs []string) []string {
	var out []string
	for i, x := range xs {
		if i == 0 || x != xs[i-1] {
			out = append(out, x)
		}
	}
	return out
}

// names reports which of the package's source files contains name (a file's base name), or "".
func (r *testRoot) names(name string) string {
	r.scan()
	if len(name) < 4 || !strings.Contains(r.src, name) {
		return ""
	}
	for i, b := range r.srcs {
		if strings.Contains(b, name) {
			return filepath.Base(r.p.Files[i])
		}
	}
	return "?"
}

// needle is what a source must contain to "name" a changed file: its base name for data (fixtures
// are referenced by name — "config.json", "parity_manifest.json"), and parent/base for a Go file,
// whose base name alone ("main.go", "model.go") is mentioned in half the tree's comments.
func needle(rel string) string {
	if strings.HasSuffix(rel, ".go") {
		if d := filepath.Base(filepath.Dir(rel)); d != "." {
			return d + "/" + filepath.Base(rel)
		}
	}
	return filepath.Base(rel)
}

// crossModuleReader reports why the test cache cannot be trusted to see a change to rel on this
// root's behalf, or "" when it can. cmd/go re-checks a test's recorded inputs only INSIDE the
// module root ("Do not recheck files outside the module, GOPATH, or GOROOT root" — Go's
// computeTestInputsID), so a metal test that read ../docs/x.md replays a stale result after x.md
// changes. Where the read is determinable, the root is forced to re-run:
//
//   - its source names the file (see needle);
//   - a "../…" string literal, resolved against the package directory, IS the file or one of its
//     directories (metal's "../docs/audit-metal-2026-09-12.md" forces on that file only, not on
//     every docs/ edit);
//   - it has a bare ".." literal and every component of the file's path as a literal
//     (filepath.Join("..", "testdata", "llama-tiny") plus "config.json"). Requiring every
//     component, not just the top directory, is what stops metal's one
//     Join("..", "docs", "measurements", …, "tickets.jsonl") from forcing it on every docs/ edit.
//
// A read by a path assembled at run time outside the module root — including a walk of the whole
// parent from a computed root — stays the cache's blind spot, exactly as it is for a hand-typed
// `go test`; that is the risk TE7(b) accepted for day loops. At 2026-09-28 no submodule test does
// that: cuda enumerates only its own directory, metal only ../testdata (both caught above).
func (r *testRoot) crossModuleReader(root, rel string) string {
	modRel := r.Mod.Dir
	if modRel == "." || rel == modRel || strings.HasPrefix(rel, modRel+"/") {
		return ""
	}
	r.scan()
	if n := needle(rel); r.names(n) != "" {
		return "names " + n + " outside its module root"
	}
	abs := filepath.Join(root, filepath.FromSlash(rel))
	for _, lit := range r.lits {
		if !strings.HasPrefix(lit, "../") {
			continue
		}
		p := lit
		if i := strings.IndexAny(p, "*?["); i >= 0 { // a glob reads the directory it starts in
			p = p[:i]
		}
		res := filepath.Clean(filepath.Join(r.Dir, filepath.FromSlash(p)))
		if res == abs || strings.HasPrefix(abs, res+string(filepath.Separator)) {
			return "reads " + strconv.Quote(lit) + " outside its module root"
		}
	}
	if r.hasLit("..") || r.hasLit("../..") {
		all := true
		for _, c := range strings.Split(rel, "/") {
			all = all && r.hasLit(c)
		}
		if all {
			return `joins ".." with every component of ` + rel
		}
	}
	return ""
}

func (r *testRoot) hasLit(s string) bool {
	if r.litSet == nil {
		r.litSet = map[string]bool{}
		for _, l := range r.lits {
			r.litSet[l] = true
		}
	}
	return r.litSet[s]
}

// ---- changed files ----

// changedFiles is the diff the selection is computed from: commits since base, staged and
// unstaged edits (`git diff <base>` compares the base commit to the WORKING TREE, so it carries
// all three), deletions (a deleted file's package still has to re-run), and untracked files that
// are not ignored. --no-renames lists a rename as its delete and its add, so both ends select.
func changedFiles(root, base string) (files []string, baseDesc string, err error) {
	git := func(args ...string) (string, error) {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			return "", fmt.Errorf("git %s: %v %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
		}
		return string(out), nil
	}
	if base == "" {
		mb, mbErr := git("merge-base", "HEAD", "origin/main")
		if mbErr != nil {
			return nil, "", fmt.Errorf("no base: %v (pass -base REF)", mbErr)
		}
		base = strings.TrimSpace(mb)
		baseDesc = "merge-base(HEAD, origin/main) = " + shortSHA(base)
	} else {
		rev, revErr := git("rev-parse", "--verify", base+"^{commit}")
		if revErr != nil {
			return nil, "", revErr
		}
		baseDesc = base + " = " + shortSHA(strings.TrimSpace(rev))
	}
	diff, err := git("diff", "--name-only", "--no-renames", "-z", base, "--")
	if err != nil {
		return nil, "", err
	}
	untracked, err := git("ls-files", "--others", "--exclude-standard", "-z")
	if err != nil {
		return nil, "", err
	}
	seen := map[string]bool{}
	for _, s := range append(strings.Split(diff, "\x00"), strings.Split(untracked, "\x00")...) {
		if s = strings.TrimSpace(s); s != "" && !seen[s] {
			seen[s] = true
			files = append(files, filepath.ToSlash(s))
		}
	}
	sort.Strings(files)
	return files, baseDesc, nil
}

func shortSHA(s string) string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

// ---- selection ----

type quickSelection struct {
	Changed []string
	// Affected: roots whose test binary compiles a changed file. Observed: roots whose tests can
	// read a changed file as data. Force: roots whose read the test cache cannot see. Each value
	// is the list of reasons, printed.
	Affected map[*testRoot][]string
	Observed map[*testRoot][]string
	Force    map[*testRoot][]string
	Rest     []*testRoot // cache-checked: no link found, and never dropped

	ChangedPkgs map[string][]string   // package -> the changed files it compiles
	ModFiles    map[*quickModule]bool // a changed go.mod/go.sum: lint all of this module
	TouchedMods map[*quickModule]bool // a changed .go file lives here (the gofmt scope)
	Goldens     []string              // why the forward goldens run; empty = not triggered
	Unread      []string              // non-Go files no Go source names (conservative: module-wide)
}

// toolchainInput reports whether a file in a package directory is an input to its compile.
func toolchainInput(name string) bool {
	switch filepath.Ext(name) {
	case ".go", ".s", ".S", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".m", ".syso":
		return true
	}
	return false
}

func addReason(m map[*testRoot][]string, r *testRoot, why string) {
	for _, w := range m[r] {
		if w == why {
			return
		}
	}
	m[r] = append(m[r], why)
}

type affectedHit struct {
	root *testRoot
	why  string
}

// affectedBy is the AFFECTED tier for one changed file: the test binaries that compile it. It is
// the one piece of the selection the equivalence property (TestQuick_neverDropsAGoListDependent)
// checks file by file against an independent `go list -deps -test` per test binary, so it stays a
// function of its own. modFile is set when rel is a module's go.mod/go.sum.
func (g *quickGraph) affectedBy(qc *quickConfig, rel string) (owners []fileOwner, modFile *quickModule, hits []affectedHit) {
	abs := filepath.Join(qc.Root, filepath.FromSlash(rel))
	mod := qc.moduleOf(rel)
	base := filepath.Base(rel)
	// A module file changes resolution for every module whose workspace includes it.
	if (base == "go.mod" || base == "go.sum") && filepath.Dir(filepath.FromSlash(rel)) == filepath.FromSlash(mod.Dir) {
		for _, r := range g.roots {
			if r.Mod == mod || slicesContains(r.Mod.Work, mod.Dir) {
				hits = append(hits, affectedHit{r, "module file " + rel})
			}
		}
		return nil, mod, hits
	}
	owners = g.owners[abs]
	if len(owners) == 0 && toolchainInput(base) {
		// A deleted file (no longer listed) or one go list did not report: its directory's
		// package, by name — a _test.go file belongs to the package's own test binary only.
		for _, p := range g.dirPkgs[filepath.Dir(abs)] {
			owners = append(owners, fileOwner{p, strings.HasSuffix(base, "_test.go")})
		}
	}
	for _, o := range owners {
		for _, r := range g.roots {
			switch {
			case r.Pkg == o.Pkg:
				hits = append(hits, affectedHit{r, "compiles " + rel})
			case !o.Test && r.Deps[o.Pkg]:
				hits = append(hits, affectedHit{r, "imports " + shortPkg(qc, o.Pkg) + " (" + rel + ")"})
			}
		}
	}
	return owners, nil, hits
}

// selectFor computes the three tiers for a set of changed files (repo-relative).
func (g *quickGraph) selectFor(qc *quickConfig, changed []string, goldenFiles map[string]string) *quickSelection {
	s := &quickSelection{
		Changed:  changed,
		Affected: map[*testRoot][]string{}, Observed: map[*testRoot][]string{}, Force: map[*testRoot][]string{},
		ChangedPkgs: map[string][]string{}, ModFiles: map[*quickModule]bool{}, TouchedMods: map[*quickModule]bool{},
	}
	for _, rel := range changed {
		mod := qc.moduleOf(rel)
		base := filepath.Base(rel)
		if why, ok := goldenFiles[rel]; ok {
			s.Goldens = append(s.Goldens, rel+" "+why)
		}

		if strings.HasSuffix(base, ".go") {
			s.TouchedMods[mod] = true // CI's gofmt -l . covers testdata .go files too
		}
		owners, modFile, hits := g.affectedBy(qc, rel)
		if modFile != nil {
			s.ModFiles[modFile] = true
		}
		for _, o := range owners {
			s.ChangedPkgs[o.Pkg] = appendUnique(s.ChangedPkgs[o.Pkg], rel)
		}
		for _, h := range hits {
			addReason(s.Affected, h.root, h.why)
		}
		if modFile != nil {
			continue
		}

		// Every file — Go files included, which the tree-walking census tests read as data — can
		// have readers that do not import it.
		named := false
		for _, r := range g.roots {
			if where := r.names(needle(rel)); where != "" {
				named = true
				addReason(s.Observed, r, "names "+needle(rel)+" ("+where+")")
			} else if r.walkSite != "" {
				addReason(s.Observed, r, "enumerates directories ("+r.walkSite+")")
			}
			if why := r.crossModuleReader(qc.Root, rel); why != "" {
				addReason(s.Force, r, why+": "+rel)
			}
		}
		if len(owners) == 0 && !named {
			// Undeterminable: nothing embeds it and no Go source names it. Conservatively its
			// whole module — sound for same-module readers, because those replay only if the
			// cache saw them read it.
			s.Unread = append(s.Unread, rel)
			for _, r := range g.roots {
				if r.Mod == mod && s.Affected[r] == nil {
					addReason(s.Observed, r, "conservative: no Go source names "+base)
				}
			}
		}
	}
	for r := range s.Affected {
		delete(s.Observed, r)
	}
	for _, r := range g.roots {
		if s.Affected[r] == nil && s.Observed[r] == nil {
			s.Rest = append(s.Rest, r)
		}
	}
	return s
}

func appendUnique(xs []string, x string) []string {
	for _, y := range xs {
		if y == x {
			return xs
		}
	}
	return append(xs, x)
}

func slicesContains(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// goldenFiles reads the parity manifest's shared sets — plus each family's own non-test forward
// files — into rel path -> "∈ shared set core". A change to any of them is a change the forward
// goldens are the numeric proof for; the manifest is where the repo already says which files
// those are, so it is read rather than restated.
func goldenFiles(qc *quickConfig) (map[string]string, error) {
	out := map[string]string{}
	if qc.Manifest == "" {
		return out, nil
	}
	b, err := os.ReadFile(filepath.Join(qc.Root, qc.Manifest))
	if os.IsNotExist(err) {
		return out, nil
	} else if err != nil {
		return nil, err
	}
	var m struct {
		SharedSets map[string][]string `json:"shared_sets"`
		Families   map[string]struct {
			Own []string `json:"own"`
		} `json:"families"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("%s: %w", qc.Manifest, err)
	}
	for set, files := range m.SharedSets {
		for _, f := range files {
			if prev, ok := out[f]; ok {
				out[f] = prev + ", " + set
			} else {
				out[f] = "∈ shared set " + set
			}
		}
	}
	for fam, e := range m.Families {
		for _, f := range e.Own {
			if strings.HasSuffix(f, ".go") && !strings.HasSuffix(f, "_test.go") {
				if _, ok := out[f]; !ok {
					out[f] = "∈ " + fam + "'s own forward files"
				}
			}
		}
	}
	return out, nil
}

// ---- cells ----

// quickCell is one `go test` invocation: a root, or one part of a split root.
type quickCell struct {
	Name   string
	Root   *testRoot
	Run    string
	Skip   string
	Count1 bool
	Golden bool
	Tier   string // affected | observer | cache-checked
	Why    string
	Heavy  bool
	NotRun string // non-empty: cannot run on this host, and why
}

// minSplitRest is how many non-walking tests a package needs before its walkers get a cell of
// their own. The split exists for decoder (hundreds of tests, ~5 min, and three census tests that
// read the whole tree); for a ten-test package the second process costs more than it saves.
const minSplitRest = 30

// anchored builds ^(a|b|c)$ for exact top-level test names.
func anchored(names []string) string {
	q := make([]string, len(names))
	for i, n := range names {
		q[i] = regexp.QuoteMeta(n)
	}
	return "^(" + strings.Join(q, "|") + ")$"
}

// buildCells turns a selection into test invocations. A root is SPLIT when that lets most of it
// stay cached: its directory-enumerating tests (which re-run whenever anything in the tree moves)
// go in a cell of their own, and the forward goldens go in theirs when a shared set is touched.
// The halves' patterns are complementary by construction — the main cell -skips exactly what the
// others -run — so a split can never lose a test.
func buildCells(qc *quickConfig, g *quickGraph, s *quickSelection, count1, heavyGoldens bool) []*quickCell {
	var cells []*quickCell
	for _, r := range g.roots {
		tier, why := "cache-checked", "no import or data link found; the test cache decides"
		switch {
		case s.Affected[r] != nil:
			tier, why = "affected", strings.Join(s.Affected[r], "; ")
		case s.Observed[r] != nil:
			tier, why = "observer", strings.Join(s.Observed[r], "; ")
		}
		force := count1 && tier != "cache-checked"
		if len(s.Force[r]) > 0 {
			force = true
		}
		notRun := ""
		if !r.Mod.native() {
			notRun = fmt.Sprintf("module %s builds only on %s (this host is %s/%s)", r.Mod.Name, r.Mod.NativeGOOS, runtime.GOOS, runtime.GOARCH)
		}
		if notRun != "" { // reported, never split: nothing of it runs here
			cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg), Root: r, Tier: tier, Why: why, NotRun: notRun})
			continue
		}
		r.scan()
		golden := r.Pkg == qc.GoldenPkg && len(s.Goldens) > 0
		var skips []string
		if golden {
			cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg) + " [goldens]", Root: r, Run: qc.GoldenRun,
				Count1: force, Golden: true, Tier: tier, Why: strings.Join(s.Goldens, "; "), Heavy: heavyGoldens, NotRun: notRun})
			skips = append(skips, qc.GoldenRun)
		}
		goldenRE := regexp.MustCompile(qc.GoldenRun)
		var walkers []string
		for _, w := range r.walkers {
			if !golden || !goldenRE.MatchString(w) {
				walkers = append(walkers, w)
			}
		}
		// A package small enough to re-run whole is not worth a second process.
		if len(walkers) > 0 && len(r.tests)-len(walkers) >= minSplitRest {
			cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg) + " [walkers]", Root: r, Run: anchored(walkers),
				Count1: force, Tier: tier, Why: why, NotRun: notRun})
			skips = append(skips, anchored(walkers))
		}
		cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg), Root: r, Skip: strings.Join(skips, "|"),
			Count1: force, Tier: tier, Why: why, NotRun: notRun})
	}
	return cells
}

// shortPkg drops the root module path for display: decoder, not github.com/…/goinfer/decoder.
func shortPkg(qc *quickConfig, pkg string) string {
	root := qc.Modules[0].Path
	if pkg == root {
		return "."
	}
	return strings.TrimPrefix(pkg, root+"/")
}

// ---- lint scope ----

// lintScope returns the packages of module m a lint step covers. "changed": packages compiling a
// changed file. "affected": those plus every package of m whose build or test imports one — a
// changed signature breaks the importer, and a test-less importer (a cmd) has no test to fail.
func lintScope(g *quickGraph, s *quickSelection, m *quickModule, scope string) []string {
	set := map[string]bool{}
	for p, q := range g.pkgs {
		if q.Mod != m {
			continue
		}
		if s.ModFiles[m] {
			set[p] = true
			continue
		}
		if _, ok := s.ChangedPkgs[p]; ok {
			set[p] = true
			continue
		}
		if scope != "affected" {
			continue
		}
		for c := range s.ChangedPkgs {
			if q.Deps[c] {
				set[p] = true
				break
			}
		}
	}
	if scope == "affected" {
		for r, why := range s.Affected {
			if r.Mod == m && len(why) > 0 {
				set[r.Pkg] = true
			}
		}
	}
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// quoteTags renders tags as CI writes them.
func quoteTags(tags []string) string {
	switch len(tags) {
	case 0:
		return ""
	case 1:
		return " -tags " + tags[0]
	}
	return " -tags '" + strings.Join(tags, " ") + "'"
}
