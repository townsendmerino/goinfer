package gatequick

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
	"github.com/townsendmerino/goinfer/internal/guard/paritykey"
)

// `gate quick`'s SELECTION: changed files -> their
// packages -> every test binary, in any of the five modules, that can observe them.
//
// THREE TIERS, because "imports the change" is not the only way a test here observes one:
//
//   - AFFECTED: the test binary compiles a changed file (the file's package is in its
//     `go list -deps -test` closure, or it embeds the file). Its binary is new, so it re-runs. This is the
//     tier the equivalence property pins (quick_test.go): it never drops a package
//     `go list -deps -test` says depends on the change.
//   - OBSERVER: the tests can READ the change as data. Some decoder tests open files across the whole tree
//     (metal/, cuda/, gpu/, internal/, docs/), so an edit to a metal .go file is observable by decoder's
//     tests although decoder imports nothing in metal. An import-graph selection alone would report GREEN
//     on a change CI then fails.
//   - CACHE-CHECKED: every other test binary. A test can open a path it builds at run time, which no static
//     scan sees, so nothing is dropped: these run WITHOUT -count=1 and Go's test cache, which records every
//     file, directory listing and env var a test binary read, replays each one whose recorded inputs did
//     not change. That is sound for reads inside the test's own module root; the cache's one blind spot
//     (cmd/go does not re-check files OUTSIDE the module root) is closed where determinable by forcing
//     -count=1 on a cross-module reader (see crossModuleReader).

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
	// Slow is a -run pattern for the module's long kernel tests. They run when the diff touches the module's own
	// directory, or under -full; a change that only reaches the module from outside defers them (deferCells).
	Slow string
	// CheckpointAlone names real-checkpoint loaders that each take a [checkpoint <name>] cell of their own instead of sharing
	// the package's [checkpoint] cell, so every checkpoint cell fits the cell budget (quickFullTimeout).
	CheckpointAlone []string
	Lint            []lintSpec
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
	Manifest  string // the parity records' directory, repo-relative, read through paritykey.ManifestJSON ("" = none)
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
		Manifest:  paritykey.RecordsDir,
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
			// The longest real-checkpoint test, alone in its cell: docs/measurements/decoder-test-growth-2026-11/README.md.
			CheckpointAlone: []string{"TestPrefillChunked_bitIdenticalReal0p5B"},
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
			Name: "gpu", Dir: "backend/webgpu", Work: []string{".", "backend/webgpu"},
			ListTags: []string{"gpu", "goinfer_testhooks"}, TestTags: []string{"gpu", "goinfer_testhooks"},
			TestArgs: []string{"-short"}, Device: true,
			// The first two were measured on the Mac; the last two on Vulkan (150 device creations each, no model, built with
			// the gpu tag alone, so gate gpu's webgpu-suite runs them): docs/measurements/integration-harness-2026-10/gate-quick-slow-cuda-2026-10-13.md.
			Slow: `^(TestGumbelDeviceAgreesWithHost|TestDecodeRunnerW4A8_geometries|TestContextClose_finalizerRace|TestContextClose_lateWrapperClose)$`,
			Lint: []lintSpec{
				{"build", []string{"gpu", "goinfer_testhooks"}, "", "affected", "gpu-darwin: go build -tags 'gpu goinfer_testhooks' ./gpu/..."},
				{"vet", []string{"gpu", "goinfer_testhooks"}, "", "affected", "gpu-darwin: go vet -tags 'gpu goinfer_testhooks' ./gpu/..."},
				// CI runs this one on linux. WebGPU is cgo, and cgo does not cross-compile from here (CGO_ENABLED=0
				// leaves webgpu's types undefined), so this is the host's analysis of the same packages.
				{"staticcheck", []string{"gpu", "goinfer_testhooks"}, "", "changed", "gpu: staticcheck -tags 'gpu goinfer_testhooks' ./gpu/... (host target)"},
			},
		},
		{
			Name: "cuda", Dir: "backend/cuda", Work: []string{".", "backend/cuda"},
			ListTags: []string{"cuda", "goinfer_testhooks"}, TestTags: []string{"cuda", "goinfer_testhooks"},
			TestArgs: []string{"-short"}, Env: map[string]string{"CGO_ENABLED": "0"},
			// `CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./cuda/...` fails on darwin/arm64 (aikit/gpu's
			// CUDA surface is linux-only) while the same vet for GOOS=linux GOARCH=amd64 passes. So it is linted
			// cross-target and its tests are NOT RUN off linux.
			NativeGOOS: "linux", CrossTarget: "linux/amd64", Device: true,
			// Kernel sweeps that load no model through the decoder and carry no goinfer_testhooks constraint, so gate
			// gpu's cuda-suite runs them. Measured over the 19 recorded gate quick runs on nobara-pc: the one such
			// test of 4 s or more (docs/measurements/integration-harness-2026-10/gate-quick-slow-cuda-2026-10-13.md).
			Slow: `^(TestTowerBase_attention)$`,
			Lint: []lintSpec{
				{"build", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "affected", "cuda: CGO_ENABLED=0 go build -tags 'cuda goinfer_testhooks' ./cuda/..."},
				{"vet", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "affected", "cuda: go vet -tags 'cuda goinfer_testhooks' ./cuda/..."},
				{"staticcheck", []string{"cuda", "goinfer_testhooks"}, "linux/amd64", "changed", "cuda: staticcheck -tags 'cuda goinfer_testhooks' ./cuda/..."},
			},
		},
		{
			Name: "metal", Dir: "backend/metal", Work: []string{".", "backend/metal"},
			ListTags: testhooks, TestTags: testhooks,
			NativeGOOS: "darwin", CrossTarget: "darwin/arm64", Device: true,
			// Long tests left in package metal. Each is untagged, so gate gpu's Metal suite runs it. The kernel sweeps
			// that need nothing from decoder live in metal/kernels, where the test cache alone keeps them off a decoder diff.
			Slow: `^(TestWeightAlias_closeCycles|TestProfileKernels)$`,
			Lint: []lintSpec{
				{"build", nil, "darwin/arm64", "affected", "gpu-darwin (metal part): go build ./metal/..."},
				{"vet", nil, "darwin/arm64", "affected", "gpu-darwin (metal part): go vet ./metal/..."},
				{"vet", testhooks, "darwin/arm64", "affected", "gpu-darwin (metal part): go vet -tags goinfer_testhooks ./metal/..."},
				// Not a CI step. Run because a local staticcheck is the only one this module gets.
				{"staticcheck", testhooks, "darwin/arm64", "changed", "gpu-darwin (metal part): staticcheck -tags goinfer_testhooks ./metal/..."},
			},
		},
		{
			Name: "apps/sdktest", Dir: "apps/sdktest", Work: []string{".", "apps/sdktest"},
			Lint: []lintSpec{
				{"build", nil, "", "affected", "sdktest: go build ./apps/sdktest/..."},
				{"vet", nil, "", "affected", "sdktest: go vet ./apps/sdktest/..."},
			},
		},
		{
			Name: "demo/agent", Dir: "demo/agent", Work: []string{".", "backend/webgpu", "demo/agent"},
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
		for ln := range strings.SplitSeq(string(b), "\n") {
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
	p := filepath.Join(qc.StateDir, gatecore.Sanitize(m.Name)+".go.work")
	if old, err := os.ReadFile(p); err != nil || string(old) != b.String() {
		if err := os.WriteFile(p, []byte(b.String()), 0o644); err != nil {
			return "", err
		}
	}
	return p, nil
}

// cmdEnv is the environment for a go command about module m, built for target ("" = this host).
// overlay, when set, is passed to every go command (the equivalence tests mutate through it, so no real
// file is ever edited).
//
// forTest builds a TEST cell's environment, and differs in two ways:
//
//   - GOWORK is left to auto-discovery wherever the repo's own go.work (or its absence, for the root)
//     already resolves the module the way its CI job does. An explicit GOWORK is inherited by every `go` a
//     test spawns, which fails a test that runs `go doc` (decoder's
//     TestGoDoc_listsEveryFieldOfOptionsAndSamplingParams, which under GOWORK=off hits a stale vendor/) or
//     `go test` in a scratch module without scrubbing it. A module with no repo go.work gets the explicit
//     GOWORK all the same, so a test that spawns `go` in a scratch module sets GOWORK=off itself.
//     The harness must not be what turns red a test that passes by hand.
//   - GOINFER_HEAVY_TESTS is removed: a shell that exported it would turn the day loop into the heavy tier.
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
	maps.Copy(set, m.Env)
	if target != "" && target != runtime.GOOS+"/"+runtime.GOARCH {
		goos, goarch, _ := strings.Cut(target, "/")
		set["GOOS"], set["GOARCH"], set["CGO_ENABLED"] = goos, goarch, "0"
	}
	var flags []string
	if v := os.Getenv("GOFLAGS"); v != "" {
		flags = append(flags, v)
	}
	if moduleMode && fileExists(filepath.Join(qc.Root, m.Dir, "vendor", "modules.txt")) {
		// A gitignored, stale vendor/ puts module-mode builds in vendor mode and fails them with "inconsistent
		// vendoring". CI's checkout has no vendor/, so -mod=readonly is what CI runs.
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
	slices.Sort(env)
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
	for ln := range strings.SplitSeq(string(b), "\n") {
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
			return "", fmt.Errorf("git %s: %w %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
		}
		return string(out), nil
	}
	if base == "" {
		mb, mbErr := git("merge-base", "HEAD", "origin/main")
		if mbErr != nil {
			return nil, "", fmt.Errorf("no base: %w (pass -base REF)", mbErr)
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
	slices.Sort(files)
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
	if slices.Contains(m[r], why) {
		return
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
			if r.Mod == mod || slices.Contains(r.Mod.Work, mod.Dir) {
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
	if slices.Contains(xs, x) {
		return xs
	}
	return append(xs, x)
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
	b, err := paritykey.ManifestJSON(paritykey.Dir(qc.Root))
	if err != nil {
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
