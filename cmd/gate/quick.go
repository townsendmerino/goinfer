package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// `gate quick` — the day loop's one command (TE7(a)+(c), docs/tasks/task-test-efficiency-2026-09.md):
// gofmt, vet (the tagged variants CI runs, for the touched modules), the pinned staticcheck, the
// tests that can observe the diff — one `go test` per package, in parallel, each to its own log —
// and, when a parity shared set is touched, the tiny forward goldens. Then PASS / SKIP / FAIL,
// counted separately because a skip is not a pass, the wall time, and the verdict.
//
// THE CONSTRAINT IT IS BUILT UNDER: a faster check that stops detecting is worse than a slow one.
// So nothing is dropped on a guess. Tests whose binary compiles the change re-run; tests that can
// read it as data are named; every other test binary still runs, and Go's test cache — which knows
// what each one actually read — replays the ones whose inputs did not move. quick_test.go holds the
// equivalence proofs, and each was shown to go red when the selection was broken on purpose.

type quickOpts struct {
	Base         string
	Files        []string // override the diff: "what if exactly these changed"
	Count1       bool
	Jobs         int
	Timeout      string
	LogDir       string
	HeavyGoldens bool
	Overlay      string // a `go build -overlay` file; every go command gets it (the equivalence tests use it)
	SelectOnly   bool
	NoLint       bool
	IgnoreLock   bool
	Verbose      bool
	Progress     io.Writer // heartbeat + per-job lines; nil = os.Stderr
	TimingLock   string    // repo-relative lock script; "" = do not check
	// Keep, when set, runs only the cells it accepts. A test hook: the real-tree equivalence
	// tests run one cell, not the tree. The CLI never sets it — quick never drops a cell.
	Keep func(*quickCell) bool
}

func runQuick(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate quick", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var o quickOpts
	var files string
	fs.StringVar(&o.Base, "base", "", "diff against REF (default: merge-base of HEAD and origin/main)")
	fs.StringVar(&files, "files", "", "comma-separated repo-relative files to treat as the change, instead of the git diff")
	fs.BoolVar(&o.Count1, "count1", false, "force -count=1 on the affected and observer cells (cache-checked ones still consult the cache)")
	fs.IntVar(&o.Jobs, "j", max(1, runtime.NumCPU()/2), "jobs run at once (device test cells additionally one at a time)")
	fs.StringVar(&o.Timeout, "timeout", "20m", "per test cell `go test -timeout`")
	fs.StringVar(&o.LogDir, "logdir", os.TempDir(), "directory the run directory of per-cell logs is created in")
	fs.BoolVar(&o.HeavyGoldens, "heavy-goldens", false, "run the forward goldens with GOINFER_HEAVY_TESTS=1 (the int8 goldens join, ~70 s more)")
	fs.StringVar(&o.Overlay, "overlay", "", "pass -overlay=FILE to every go command (test the tree as if files were edited, without editing them)")
	fs.BoolVar(&o.SelectOnly, "select", false, "print the selection and exit")
	fs.BoolVar(&o.NoLint, "no-lint", false, "skip gofmt / vet / build / staticcheck")
	fs.BoolVar(&o.IgnoreLock, "ignore-timing-lock", false, "run although a timed run holds the timing lock (it will contaminate that run)")
	fs.BoolVar(&o.Verbose, "v", false, "print every selection reason and every skip")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if files != "" {
		for f := range strings.SplitSeq(files, ",") {
			if f = strings.TrimSpace(f); f != "" {
				o.Files = append(o.Files, filepath.ToSlash(f))
			}
		}
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate quick: %v\n", err)
		return 2
	}
	qc, err := realQuickConfig(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate quick: %v\n", err)
		return 2
	}
	o.TimingLock = "scripts/timing_lock.py"
	return quickMain(qc, o, w)
}

// quickMain returns the exit code: 0 green, 1 red, 2 refused.
func quickMain(qc *quickConfig, o quickOpts, w io.Writer) int {
	t0 := time.Now()
	progress := o.Progress
	if progress == nil {
		progress = os.Stderr
	}
	refuse := func(format string, a ...any) int {
		fmt.Fprintf(w, "%s== quick ==%s\n  %sREFUSED: %s%s\n", bold, off, amber, fmt.Sprintf(format, a...), off)
		return 2
	}
	if _, err := exec.LookPath("go"); err != nil {
		return refuse("go toolchain not on PATH: %v", err)
	}

	// TE9: one timed run per box. quick is not timed, but it loads every core, so it must not run
	// under a run that is — two runs on one box measure each other, and nothing downstream can tell.
	lockNote := "not checked"
	if o.TimingLock != "" {
		if _, err := os.Stat(filepath.Join(qc.Root, o.TimingLock)); err == nil {
			cmd := exec.Command("python3", o.TimingLock, "status")
			cmd.Dir = qc.Root
			out, err := cmd.CombinedOutput()
			msg := strings.TrimSpace(string(out))
			switch rc := exitCode(err); {
			case err == nil:
				lockNote = "free"
			case rc == 1 && !o.IgnoreLock:
				return refuse("a timed run holds the timing lock — quick would contaminate it (-ignore-timing-lock overrides):\n    %s", msg)
			case rc == 1:
				lockNote = "HELD, ignored by -ignore-timing-lock: " + msg
			default:
				lockNote = fmt.Sprintf("not checked (%v: %s)", err, msg)
			}
		}
	}

	changed, baseDesc := o.Files, "-files (no git diff)"
	if len(changed) == 0 {
		var err error
		changed, baseDesc, err = changedFiles(qc.Root, o.Base)
		if err != nil {
			return refuse("%v", err)
		}
	}
	golden, err := goldenFiles(qc)
	if err != nil {
		return refuse("parity manifest: %v", err)
	}
	g, err := loadGraph(qc, o.Overlay)
	if err != nil {
		return refuse("%v", err)
	}
	sel := g.selectFor(qc, changed, golden)
	cells := buildCells(qc, g, sel, o.Count1, o.HeavyGoldens)

	rundir := ""
	if !o.SelectOnly {
		if err := os.MkdirAll(o.LogDir, 0o755); err != nil {
			return refuse("logdir: %v", err)
		}
		if rundir, err = os.MkdirTemp(o.LogDir, "gate-quick-"+t0.Format("20060102-150405")+"-"); err != nil {
			return refuse("run dir: %v", err)
		}
	}
	count := "cached: an unchanged test binary with unchanged inputs replays (TE7(b))"
	if o.Count1 {
		count = "-count=1 forced on affected + observer cells; cache-checked cells still replay"
	}
	prov := gatherProvenanceIn(qc.Root, [][2]string{
		{"base", baseDesc},
		{"changed", fmt.Sprintf("%d file(s)", len(changed))},
		{"count", count},
		{"jobs", fmt.Sprintf("%d at once (device cells one at a time; checkpoint-loading cells alone)", o.Jobs)},
		{"timing lock", lockNote},
		{"run dir", orDash(rundir)},
		{"started", t0.Format("15:04:05 MST")},
	})
	fmt.Fprintf(w, "%s== quick provenance ==%s\n", bold, off)
	prov.write(w)
	printSelection(w, qc, g, sel, cells, o.Verbose)
	if o.SelectOnly {
		return 0
	}
	if len(changed) == 0 {
		fmt.Fprintf(w, "\n  nothing changed against the base: every cell is cache-checked\n")
	}

	hist := loadDurations(qc.StateDir)
	var jobs []*quickJob
	if !o.NoLint {
		jobs = append(jobs, lintJobs(qc, g, sel, o.Overlay, rundir)...)
	}
	var notRun []*quickCell
	var tests []*quickJob
	for _, c := range cells {
		if o.Keep != nil && !o.Keep(c) {
			continue
		}
		if c.NotRun != "" {
			notRun = append(notRun, c)
			continue
		}
		jb, err := testJob(qc, c, o)
		if err != nil {
			return refuse("%v", err)
		}
		jb.expected = hist[c.Name]
		tests = append(tests, jb)
	}
	// Longest first; a cell never run before goes by its test count, which ranks decoder first.
	sort.SliceStable(tests, func(i, j int) bool {
		a, b := tests[i], tests[j]
		if a.expected != b.expected {
			return a.expected > b.expected
		}
		return len(a.Cell.Root.tests) > len(b.Cell.Root.tests)
	})
	jobs = append(jobs, tests...)

	beat := 30 * time.Second
	if v := os.Getenv("GOINFER_GATE_HEARTBEAT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			beat = d
		}
	}
	fmt.Fprintf(progress, "== quick: %d job(s), started %s — progress every %s ==\n", len(jobs), t0.Format("15:04:05"), beat)
	runJobs(jobs, max(1, o.Jobs), rundir, progress, beat, t0)
	saveDurations(qc.StateDir, hist, jobs)
	return reportQuick(w, qc, sel, jobs, notRun, t0, o.Verbose)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// gatherProvenanceIn is gatherProvenance for a tree that may not be the process's cwd.
func gatherProvenanceIn(root string, fields [][2]string) provenance {
	p := provenance{Commit: "?", Date: time.Now().UTC().Format(time.RFC3339), Fields: fields}
	git := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, _ := cmd.Output()
		return strings.TrimSpace(string(out))
	}
	if c := git("rev-parse", "--short", "HEAD"); c != "" {
		p.Commit = c
	}
	p.Dirty = git("status", "--porcelain") != ""
	if out, err := exec.Command("uname", "-sm").Output(); err == nil {
		p.Host = strings.TrimSpace(string(out))
	}
	return p
}

func testJob(qc *quickConfig, c *quickCell, o quickOpts) (*quickJob, error) {
	m := c.Root.Mod
	env, err := qc.cmdEnv(m, "", o.Overlay, true)
	if err != nil {
		return nil, err
	}
	timeout := o.Timeout
	if c.Heavy {
		env = append(env, "GOINFER_HEAVY_TESTS=1")
		timeout = "60m"
	}
	args := []string{"go", "test", "-json"}
	if c.Count1 {
		args = append(args, "-count=1")
	}
	if len(m.TestTags) > 0 {
		args = append(args, "-tags", strings.Join(m.TestTags, " "))
	}
	args = append(args, "-timeout", timeout)
	if c.Run != "" {
		args = append(args, "-run", c.Run)
	}
	if c.Skip != "" {
		args = append(args, "-skip", c.Skip)
	}
	args = append(args, m.TestArgs...)
	args = append(args, c.Root.Pkg)
	return &quickJob{Name: c.Name, Kind: "test", Dir: qc.modDir(m), Env: env, Args: args, Device: m.Device, Mem: c.Mem, Cell: c}, nil
}

// lintJobs builds gofmt (per touched module), the staticcheck canary, and each module's vet /
// build / staticcheck steps over the packages the diff reaches.
func lintJobs(qc *quickConfig, g *quickGraph, s *quickSelection, overlay, rundir string) []*quickJob {
	var jobs []*quickJob
	gofmt := "gofmt"
	if out, err := exec.Command("go", "env", "GOROOT").Output(); err == nil {
		if p := filepath.Join(strings.TrimSpace(string(out)), "bin", "gofmt"); fileExists(p) {
			gofmt = p // the toolchain's own gofmt, so its version is the compiler's
		}
	}
	for _, m := range qc.Modules {
		if !s.TouchedMods[m] {
			continue
		}
		files := moduleGoFiles(qc, m)
		jobs = append(jobs, &quickJob{Name: "gofmt " + m.Name, Kind: "lint", Dir: qc.Root,
			Args: append([]string{gofmt, "-l"}, files...), QuietOK: true,
			Detail: fmt.Sprintf("%d files", len(files)), Mirrors: "lint: gofmt -l ."})
	}

	sc, scErr := staticcheckBinary(qc)
	canary := false
	for _, m := range qc.Modules {
		for _, spec := range m.Lint {
			pkgs := lintScope(g, s, m, spec.Scope)
			if len(pkgs) == 0 {
				continue
			}
			env, err := qc.cmdEnv(m, spec.Target, overlay, false)
			name := spec.Tool + " " + m.Name + quoteTags(spec.Tags)
			if spec.Target != "" && spec.Target != runtime.GOOS+"/"+runtime.GOARCH {
				name += " (" + spec.Target + ")"
			}
			jb := &quickJob{Name: name, Kind: "lint", Dir: qc.modDir(m), Env: env, Mirrors: spec.Mirrors,
				Detail: fmt.Sprintf("%d pkgs", len(pkgs))}
			if err != nil {
				jb.Err = err
			}
			var tool []string
			filterTmpl := `{{if or .GoFiles .CgoFiles .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}`
			switch spec.Tool {
			case "build":
				tool = []string{"go", "build", "-o", os.DevNull}
				filterTmpl = `{{if or .GoFiles .CgoFiles}}{{.ImportPath}}{{end}}` // a test-only package does not build
			case "vet":
				tool = []string{"go", "vet"}
			case "staticcheck":
				tool = []string{sc}
				if scErr != nil {
					jb.Err = scErr
				}
				canary = true
			}
			if len(spec.Tags) > 0 {
				tool = append(tool, "-tags", strings.Join(spec.Tags, " "))
			}
			filter := []string{"go", "list", "-e"}
			if len(spec.Tags) > 0 {
				filter = append(filter, "-tags", strings.Join(spec.Tags, " "))
			}
			jb.Args = tool
			jb.Filter = append(append(filter, "-f", filterTmpl), pkgs...)
			jobs = append(jobs, jb)
		}
	}
	if canary {
		// PROVE THE GATE CAN GO RED. An empty staticcheck result is indistinguishable from one that
		// never analysed anything: a version-skewed binary prints only an internal error about
		// importing internal/cpu and checks nothing (CLAUDE.md; a U1000 held CI red for three pushes).
		// So every run shows the binary finding the defect CI once shipped, in a four-line package.
		dir := filepath.Join(rundir, "staticcheck-canary")
		_ = os.MkdirAll(dir, 0o755)
		_ = os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module canary\n\ngo 1.21\n"), 0o644)
		_ = os.WriteFile(filepath.Join(dir, "canary.go"), []byte("package canary\n\ntype s struct{ deadField int }\n\nvar _ = s{}\n"), 0o644)
		env := append(os.Environ(), "GOWORK=off", "GOFLAGS=")
		jb := &quickJob{Name: "staticcheck canary", Kind: "lint", Dir: dir, Env: env, Args: []string{sc, "./..."},
			Canary: "field deadField is unused (U1000)", Mirrors: "must print U1000, or no staticcheck result above means anything"}
		if scErr != nil {
			jb.Err = scErr
		}
		jobs = append([]*quickJob{jb}, jobs...)
	}
	for _, jb := range jobs {
		if jb.Err != nil {
			jb.Args = nil // run() reports the error without starting anything
		}
	}
	return jobs
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// moduleGoFiles is the module's tracked and untracked-unignored .go files, without the modules
// nested inside it: `gofmt -l .` over what CI's checkout would hold (it has no gitignored vendor/).
func moduleGoFiles(qc *quickConfig, m *quickModule) []string {
	cmd := exec.Command("git", "ls-files", "-co", "--exclude-standard", "-z", "--", m.Dir)
	cmd.Dir = qc.Root
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var files []string
	for f := range strings.SplitSeq(string(out), "\x00") {
		if !strings.HasSuffix(f, ".go") || qc.moduleOf(f) != m || !fileExists(filepath.Join(qc.Root, f)) {
			continue
		}
		files = append(files, f)
	}
	return files
}

// staticcheckBinary finds the pinned staticcheck: CI installs it where `go install` puts it and
// calls it by name, so GOPATH/bin first. Its version must be the release ci.yml pins — a
// version-skewed staticcheck analyses nothing and says so only in an internal error.
func staticcheckBinary(qc *quickConfig) (string, error) {
	bin := ""
	if out, err := exec.Command("go", "env", "GOPATH").Output(); err == nil {
		gp, _, _ := strings.Cut(strings.TrimSpace(string(out)), string(os.PathListSeparator))
		if p := filepath.Join(gp, "bin", "staticcheck"); fileExists(p) {
			bin = p
		}
	}
	if bin == "" {
		p, err := exec.LookPath("staticcheck")
		if err != nil {
			return "staticcheck", fmt.Errorf("staticcheck not installed: %s", staticcheckInstall)
		}
		bin = p
	}
	out, err := exec.Command(bin, "-version").CombinedOutput()
	if err != nil {
		return bin, fmt.Errorf("%s -version: %v", bin, err)
	}
	want := pinnedStaticcheck(qc.Root)
	if !strings.Contains(string(out), want) {
		return bin, fmt.Errorf("%s is %q; CI pins %q — %s",
			bin, strings.TrimSpace(string(out)), strings.TrimSpace(want), staticcheckInstall)
	}
	return bin, nil
}

// staticcheckInstall builds the staticcheck CI pins (.github/actions/staticcheck's go.mod: v0.8.1 against
// golang.org/x/tools v0.51.0, since no release reads Go 1.27.2's export data) where `go install` would put it.
const staticcheckInstall = `GOWORK=off go build -C .github/actions/staticcheck -o "$(go env GOPATH)/bin/staticcheck" honnef.co/go/tools/cmd/staticcheck`

var staticcheckPinRE = regexp.MustCompile(`(?s)uses: \./\.github/actions/staticcheck\s+with:\s+version: "([0-9.]+)"`)

// pinnedStaticcheck reads the release tag CI pins (ci.yml's staticcheck action input, e.g. "2026.2.1",
// which `staticcheck -version` prints as "staticcheck 2026.2.1 (0.8.1)"). Without a ci.yml it falls
// back to the version CLAUDE.md names.
func pinnedStaticcheck(root string) string {
	if b, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci.yml")); err == nil {
		if m := staticcheckPinRE.FindSubmatch(b); m != nil {
			return "staticcheck " + string(m[1]) + " "
		}
	}
	return "(0.8.1)"
}

// ---- report ----

func printSelection(w io.Writer, qc *quickConfig, g *quickGraph, s *quickSelection, cells []*quickCell, verbose bool) {
	fmt.Fprintf(w, "\n%s== selection ==%s\n", bold, off)
	if len(s.Changed) == 0 {
		fmt.Fprintf(w, "  changed: (none)\n")
	} else {
		fmt.Fprintf(w, "  changed:\n")
		for i, f := range s.Changed {
			if i >= 25 && !verbose {
				fmt.Fprintf(w, "    … %d more (-v lists them)\n", len(s.Changed)-i)
				break
			}
			fmt.Fprintf(w, "    %s\n", f)
		}
	}
	list := func(title string, m map[*testRoot][]string) {
		var roots []*testRoot
		for r := range m {
			roots = append(roots, r)
		}
		sort.Slice(roots, func(i, j int) bool { return roots[i].Pkg < roots[j].Pkg })
		fmt.Fprintf(w, "  %s (%d):\n", title, len(roots))
		for _, r := range roots {
			why := m[r]
			shown := why
			if !verbose && len(why) > 2 {
				shown = append(append([]string(nil), why[:2]...), fmt.Sprintf("+%d more", len(why)-2))
			}
			fmt.Fprintf(w, "    %-28s %s\n", shortPkg(qc, r.Pkg), strings.Join(shown, "; "))
		}
	}
	list("AFFECTED — the test binary compiles the change, so it re-runs", s.Affected)
	list("OBSERVERS — can read the change as data; the test cache re-runs them if they did", s.Observed)
	var rest []string
	for _, r := range s.Rest {
		rest = append(rest, shortPkg(qc, r.Pkg))
	}
	fmt.Fprintf(w, "  CACHE-CHECKED (%d) — no link found; kept, and replayed by the test cache unless an input changed:\n    %s\n",
		len(rest), wrapWords(rest, 96, "    "))
	if len(s.Force) > 0 {
		list("FORCED -count=1 — read outside their module root, which the test cache does not re-check", s.Force)
	}
	if len(s.Unread) > 0 {
		fmt.Fprintf(w, "  undeterminable (no Go source names them; their module is selected conservatively): %s\n",
			strings.Join(s.Unread, " "))
	}
	if len(s.Goldens) > 0 {
		fmt.Fprintf(w, "  %sFORWARD GOLDENS%s run (%s -run %q): %s\n", bold, off, shortPkg(qc, qc.GoldenPkg), qc.GoldenRun, strings.Join(s.Goldens, "; "))
		fmt.Fprintf(w, "    note: a shared-set edit also re-stales TestParityManifest_fresh until deps_hash is refreshed\n"+
			"    (bash scripts/refresh_parity_hashes.sh for a non-numeric change) or the family is re-validated.\n")
	} else {
		fmt.Fprintf(w, "  forward goldens: not triggered (no parity shared-set file changed)\n")
	}
	var split, nr []string
	for _, c := range cells {
		if strings.HasSuffix(c.Name, "[walkers]") || strings.HasSuffix(c.Name, "[checkpoint]") {
			n := strings.Count(c.Run, "|") + 1
			split = append(split, fmt.Sprintf("%s (%d of %d tests)", c.Name, n, len(c.Root.tests)))
			if verbose {
				split[len(split)-1] += " " + c.Run
			}
		} else if strings.HasSuffix(c.Name, "]") {
			split = append(split, c.Name)
		}
		if c.NotRun != "" {
			nr = append(nr, c.Name+" — "+c.NotRun)
		}
	}
	if len(split) > 0 {
		fmt.Fprintf(w, "  split cells (so the rest of the package can stay cached, or run beside others): %s\n", strings.Join(split, ", "))
	}
	var mem []string
	for _, c := range cells {
		if c.Mem && c.NotRun == "" {
			mem = append(mem, c.Name)
		}
	}
	if len(mem) > 0 {
		fmt.Fprintf(w, "  run ALONE, nothing beside them (they load a real checkpoint): %s\n", strings.Join(mem, ", "))
		if verbose {
			for _, r := range g.roots {
				if !r.Mod.native() {
					continue
				}
				for _, t := range r.loaders {
					fmt.Fprintf(w, "    %s %s: %s\n", shortPkg(qc, r.Pkg), t, r.loadWhy[t])
				}
			}
		}
	}
	for _, n := range nr {
		fmt.Fprintf(w, "  %sNOT RUNNABLE HERE%s: %s\n", amber, off, n)
	}
	for i, wmsg := range g.warnings {
		if i >= 5 && !verbose {
			fmt.Fprintf(w, "  … %d more go list error(s)\n", len(g.warnings)-i)
			break
		}
		fmt.Fprintf(w, "  %sgo list%s: %s\n", amber, off, trunc(wmsg, 160))
	}
}

func wrapWords(words []string, width int, indent string) string {
	var b strings.Builder
	col := 0
	for i, wd := range words {
		if i > 0 {
			if col+1+len(wd) > width {
				b.WriteString("\n" + indent)
				col = 0
			} else {
				b.WriteByte(' ')
				col++
			}
		}
		b.WriteString(wd)
		col += len(wd)
	}
	return b.String()
}

func reportQuick(w io.Writer, qc *quickConfig, s *quickSelection, jobs []*quickJob, notRun []*quickCell, t0 time.Time, verbose bool) int {
	var lint, tests []*quickJob
	for _, jb := range jobs {
		if jb.Kind == "lint" {
			lint = append(lint, jb)
		} else {
			tests = append(tests, jb)
		}
	}
	failures := 0
	if len(lint) > 0 {
		fmt.Fprintf(w, "\n%s== lint ==%s\n", bold, off)
		for _, jb := range lint {
			st, col := "PASS", green
			if !jb.ok() {
				st, col = "FAIL", red
				failures++
			}
			fmt.Fprintf(w, "  %s%-4s%s  %-52s %-9s %6s   %s\n", col, st, off, jb.Name, jb.Detail, jb.dur().Round(100*time.Millisecond), jb.Mirrors)
			if !jb.ok() {
				msg := jb.Out
				if jb.Err != nil {
					msg = jb.Err.Error() + "\n" + msg
				}
				if jb.QuietOK {
					msg = "unformatted:\n" + msg
				}
				if jb.Canary != "" {
					msg = "the canary's U1000 did NOT appear — this staticcheck cannot be trusted to find anything:\n" + msg
				}
				for i, ln := range strings.Split(strings.TrimRight(msg, "\n"), "\n") {
					if i >= 15 {
						fmt.Fprintf(w, "        … (see %s)\n", jb.LogPath)
						break
					}
					fmt.Fprintf(w, "        %s\n", ln)
				}
			}
		}
	}

	fmt.Fprintf(w, "\n%s== tests ==%s\n", bold, off)
	var pass, skip, fail, ran, cached, allSkip int
	var goldenJobs []*quickJob
	var vacuousAffected []string
	skipBuckets := map[string]int{}
	var skipRows []skipRow
	sorted := append([]*quickJob(nil), tests...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	for _, jb := range sorted {
		pass, skip, fail = pass+jb.Pass, skip+jb.Skip, fail+jb.Fail
		if jb.Hidden || jb.Err != nil {
			fail++
		}
		if jb.Cached {
			cached++
		} else {
			ran++
		}
		st := statusWord(jb)
		col := green
		switch st {
		case "FAIL":
			col = red
			failures++
		case "ALLSKIP", "NOTESTS":
			col = amber
			allSkip++
			if jb.Cell.Tier == "affected" {
				vacuousAffected = append(vacuousAffected, jb.Name)
			}
		}
		if jb.Cell.Golden {
			goldenJobs = append(goldenJobs, jb)
		}
		fmt.Fprintf(w, "  %s%-7s%s %-34s %-13s %7s  %4d pass %4d skip %3d fail\n", col, st, off, jb.Name, jb.Cell.Tier,
			jb.dur().Round(100*time.Millisecond), jb.Pass, jb.Skip, jb.Fail)
		if jb.res != nil {
			for _, r := range collectSkips(jb.res, true) {
				skipBuckets[r.Bucket]++
				skipRows = append(skipRows, r)
			}
		}
	}
	for _, c := range notRun {
		fmt.Fprintf(w, "  %s%-7s%s %-34s %s\n", amber, "NOT RUN", off, c.Name, c.NotRun)
	}

	// Each failing cell: its source-located lines, or — for a failure with no per-test FAIL (a
	// build error, a panic, a timeout) — the tail of its log, which is the only place it shows.
	for _, jb := range sorted {
		if jb.ok() {
			continue
		}
		fmt.Fprintf(w, "\n  %s%s%s — log: %s\n", red, jb.Name, off, jb.LogPath)
		if jb.Err != nil {
			fmt.Fprintf(w, "    did not run: %v\n", jb.Err)
		}
		if jb.res != nil && jb.Fail > 0 {
			writeFailures(w, jb.res, true)
		}
		if jb.Hidden {
			fmt.Fprintf(w, "    go test exited %d with 0 counted failures (build error / panic / timeout); log tail:\n", jb.RC)
			for _, ln := range tailLines(jb.LogPath, 15) {
				fmt.Fprintf(w, "      %s\n", ln)
			}
		}
	}

	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  tests    PASSED %d   SKIPPED %d   FAILED %d   (top-level tests; a skip is not a pass)\n", pass, skip, fail)
	fmt.Fprintf(w, "  cells    %d: %d ran, %d replayed from the test cache, %d not runnable on this host\n",
		len(tests)+len(notRun), ran, cached, len(notRun))
	if len(skipBuckets) > 0 {
		var parts []string
		for _, b := range bucketOrder {
			if n := skipBuckets[b]; n > 0 {
				parts = append(parts, fmt.Sprintf("%s %d", b, n))
			}
		}
		fmt.Fprintf(w, "  skips    %s\n", strings.Join(parts, ", "))
		if verbose {
			for _, r := range skipRows {
				fmt.Fprintf(w, "    %s  %s — %s\n", r.Pkg, r.Test, trunc(r.Reason, 100))
			}
		}
	}
	lintFail := 0
	for _, jb := range lint {
		if !jb.ok() {
			lintFail++
		}
	}
	if len(lint) > 0 {
		fmt.Fprintf(w, "  lint     %d step(s): %d pass, %d fail\n", len(lint), len(lint)-lintFail, lintFail)
	} else {
		fmt.Fprintf(w, "  lint     not run\n")
	}
	if len(s.Goldens) > 0 {
		gp, gs, gf := 0, 0, 0
		for _, jb := range goldenJobs {
			gp, gs, gf = gp+jb.Pass, gs+jb.Skip, gf+jb.Fail
		}
		fmt.Fprintf(w, "  goldens  %d passed, %d skipped, %d failed\n", gp, gs, gf)
		// ZERO GOLDENS RAN is not a green: a shared-set change with no numeric proof is exactly
		// what scripts/refresh_parity_hashes.sh refuses on, for the same reason.
		if gp == 0 && gf == 0 && len(goldenJobs) > 0 {
			fmt.Fprintf(w, "  %sgoldens VACUOUS — a shared set changed and no forward golden ran (fixtures absent?)%s\n", red, off)
			failures++
		}
	} else {
		fmt.Fprintf(w, "  goldens  not triggered\n")
	}
	end := time.Now()
	fmt.Fprintf(w, "  wall     %s (%s → %s)\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"), end.Format("15:04:05"))
	if len(notRun) > 0 {
		fmt.Fprintf(w, "  %sNOT COVERED on this host%s: %d cell(s) of module(s) that do not build here — run quick on a box that builds them.\n",
			amber, off, len(notRun))
	}
	if len(vacuousAffected) > 0 {
		fmt.Fprintf(w, "  %sUNVERIFIED%s: affected cell(s) in which every test skipped: %s\n", amber, off, strings.Join(vacuousAffected, ", "))
	}
	if failures > 0 {
		fmt.Fprintf(w, "  %sRED — %d failing step(s)/cell(s)%s\n", red, failures, off)
		return 1
	}
	fmt.Fprintf(w, "  %sGREEN — %d tests passed (%d skipped)%s\n", green, pass, skip, off)
	return 0
}

func tailLines(path string, n int) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return []string{fmt.Sprintf("(no log: %v)", err)}
	}
	lines := strings.Split(strings.TrimRight(string(bytes.TrimSpace(b)), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}
