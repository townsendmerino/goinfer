package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

// `gate identity <old-rev> <new-rev>`: inherit validation by identity (TE6(b),
// docs/tasks/task-test-efficiency-2026-09.md).
//
// For a change meant to be numerically neutral, the cheapest complete proof is byte-identical logits
// against the last validated build on the family's parity prompt: build ONE small dumper at each revision,
// dump full logits (prefill + N greedy steps, raw little-endian float32) for each family's parity
// prompt(s), and compare bytes.
//
//   - Two temporary `git worktree`s, detached, outside the repo; removed afterwards (also on SIGINT).
//     The main working tree is never touched. The dumper is written into each worktree and built
//     there with -trimpath, so both builds compile the same dumper against their own revision.
//   - DETERMINISM FIRST: the NEW build runs every cell twice, in separate processes, and must be
//     byte-identical to itself. A cell that is not is compared against the old build in tolerance
//     mode (identityTolFactor × its own run-to-run max |diff|, argmax and tokens exact) and reported
//     WITHIN TOLERANCE, never IDENTICAL. On the CPU that nondeterminism is TE6's kill criterion.
//   - A SKIP IS NOT A PASS: a family whose asset is missing, whose load fails in both builds, or whose
//     GPU cell fell back to the CPU is NOT RUN, listed separately with the reason.
//   - BOTH SIDES SEE ONE ENVIRONMENT. Every GOINFER_* variable is removed from the dumpers' env: a
//     diagnostic given to one side only may not reach everything it claims to, and the dumps then differ
//     from byte 1 for a reason that is not the change. The only thing that differs between the two runs is
//     the code; the decode path each side reports is printed, and a path difference is flagged.
//   - The parity manifest is NOT written: what an "identity-inherited" row would record is printed,
//     and whether to make it a manifest method is the owner's decision.

//go:embed identity_dumper.go.tmpl
var identityDumperTmpl string

type identityOpts struct {
	Repo        string // the git repository the revs resolve in
	FixtureRoot string // the tree the tiny fixtures are read from (gitignored ones exist only in the real checkout)
	OldRev      string
	NewRev      string
	Backend     string
	Families    []string
	Only        []string // asset names (identityAsset.Name) to keep; nil = all
	Assets      string   // tiny | real
	Quants      []string // nil = defaultIdentityQuants
	Steps       int
	Long        bool // -assets real: add a third prompt of about 2,500 tokens
	ModelsDir   string
	LogDir      string
	Keep        bool
	Record      string        // -record: write eligible families' PARITY_ROW lines here (identity_record.go); with -giw bytes, one JSON line per cell
	Giw         string        // "" | bytes | logits: compare through each revision's own transcode (identity_giw.go)
	Configs     []giwConfig   // -giw bytes: the transcode configurations; nil = defaultGiwConfigs
	Timeout     time.Duration // per dumper process
	Progress    io.Writer
	// DumperSource replaces the embedded template (the equivalence tests inject a nondeterministic
	// dumper through it). The CLI never sets it.
	DumperSource string
}

func runIdentity(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate identity", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var o identityOpts
	var fams, quants, only, configs string
	fs.StringVar(&o.Backend, "backend", "cpu", "cpu | metal | webgpu | cuda")
	fs.StringVar(&fams, "families", "", "comma-separated parity-manifest families (default: all)")
	fs.StringVar(&o.Assets, "assets", "tiny", "tiny (the forward goldens' fixtures) | real (small checkpoints under ~/models)")
	fs.StringVar(&quants, "quant", "", "comma-separated quants (default: cpu f32,int8int8,int4; metal and cuda int4,int8int8; webgpu int4; -assets real int4)")
	fs.StringVar(&only, "only", "", "comma-separated asset names to keep (e.g. coder-0.5b-q4km)")
	fs.BoolVar(&o.Long, "long", false, "-assets real: add a third prompt of about 2,500 tokens, past the CPU prefill's 2,048-row chunk")
	fs.IntVar(&o.Steps, "steps", 0, "tokens generated per prompt; logits are dumped for each (default: tiny 8, real 32)")
	fs.StringVar(&o.LogDir, "logdir", os.TempDir(), "where the run directory (worktrees, builds, dumps, logs) is created")
	fs.BoolVar(&o.Keep, "keep", false, "keep the dumps, binaries and logs (worktrees are always removed)")
	fs.StringVar(&o.Record, "record", "", "write PARITY_ROW lines for families whose T3 validation this run inherits (TE6(b)); merge them with the decoder's TestParityManifest_merge")
	fs.DurationVar(&o.Timeout, "timeout", 15*time.Minute, "per dumper process, and per transcode with -giw")
	fs.StringVar(&o.Giw, "giw", "", "compare through each revision's OWN transcode of the source: bytes (the two .giw files' sha256) | logits (the dumper on each side's own .giw)")
	replay := fs.String("replay", "", "grade the run record a -keep run wrote ("+identityRunRecordFile+" in its run directory) instead of running")
	fs.StringVar(&configs, "configs", "", "-giw bytes: comma-separated quant:target[:e4h] (default: tiny int4:metal:e4h,int4:metal,int4:cpu,int8int8:cpu,int4:cpu-arm64; real int4:cpu and int4 for this box's GPU)")
	// Positionals may sit before, between or after the flags.
	var pos []string
	rest := argv
	for {
		if err := fs.Parse(rest); err != nil {
			return 2
		}
		if fs.NArg() == 0 {
			break
		}
		pos = append(pos, fs.Arg(0))
		rest = fs.Args()[1:]
	}
	if *replay != "" {
		return replayIdentity(w, *replay)
	}
	if len(pos) != 2 {
		fmt.Fprintln(os.Stderr, "usage: gate identity <old-rev> <new-rev> [-backend cpu|metal|webgpu|cuda] [-families a,b] [-assets tiny|real] [-quant q,…] [-steps N] [-keep] [-giw bytes|logits [-configs q:target[:e4h],…]]")
		return 2
	}
	o.OldRev, o.NewRev = pos[0], pos[1]
	o.Families, o.Quants, o.Only = splitList(fams), splitList(quants), splitList(only)
	switch o.Giw {
	case "", "bytes":
	case "logits":
		if o.Record != "" {
			fmt.Fprintln(os.Stderr, "gate identity: -record does not apply to -giw logits (a transcoded sidecar's logits inherit no parity row)")
			return 2
		}
	default:
		fmt.Fprintf(os.Stderr, "gate identity: -giw %q: want bytes or logits\n", o.Giw)
		return 2
	}
	if configs != "" && o.Giw != "bytes" {
		fmt.Fprintln(os.Stderr, "gate identity: -configs applies to -giw bytes only")
		return 2
	}
	var cerr error
	if o.Configs, cerr = parseGiwConfigs(configs); cerr != nil {
		fmt.Fprintf(os.Stderr, "gate identity: %v\n", cerr)
		return 2
	}
	switch o.Backend {
	case "cpu", "metal", "webgpu", "cuda":
	default:
		fmt.Fprintf(os.Stderr, "gate identity: -backend %q: want cpu, metal, webgpu or cuda\n", o.Backend)
		return 2
	}
	if o.Assets != "tiny" && o.Assets != "real" {
		fmt.Fprintf(os.Stderr, "gate identity: -assets %q: want tiny or real\n", o.Assets)
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate identity: %v\n", err)
		return 2
	}
	o.Repo, o.FixtureRoot = root, root
	o.ModelsDir = env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))
	return identityMain(o, w)
}

func splitList(s string) []string {
	var out []string
	for f := range strings.SplitSeq(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// identitySide is one revision's worktree and dumper build.
type identitySide struct {
	Label string // old | new
	Rev   string // as given
	SHA   string
	Short string
	Title string
	WT    string
	Bin   string
	// Prequant is this revision's cmd/prequant, built when -giw is set.
	Prequant string
	BuildT   time.Duration
	Err      error
	Log      string
}

// identityRun is one execution of a dumper over all cells: new#1, old, new#2.
type identityRun struct {
	Name string // new1 | old | new2
	Side *identitySide
}

// identityMain returns 0 when every family that ran is IDENTICAL (or WITHIN TOLERANCE on a
// non-deterministic GPU backend), 1 when any family is DIFFERENT or the CPU is non-deterministic,
// and 2 when it refused (bad rev, a build failed, or nothing at all could run).
func identityMain(o identityOpts, w io.Writer) int {
	// What identity prints before the verdict is kept in the run record, so a replay prints the whole report.
	var prelude strings.Builder
	out := w
	w = io.MultiWriter(out, &prelude)
	t0 := time.Now()
	progress := o.Progress
	if progress == nil {
		progress = os.Stderr
	}
	refuse := func(format string, a ...any) int {
		fmt.Fprintf(w, "%s== identity ==%s\n  %sREFUSED: %s%s\n", bold, off, amber, fmt.Sprintf(format, a...), off)
		return 2
	}
	for _, bin := range []string{"go", "git"} {
		if _, err := exec.LookPath(bin); err != nil {
			return refuse("%s not on PATH", bin)
		}
	}
	if o.Steps <= 0 {
		o.Steps = 8
		if o.Assets == "real" {
			o.Steps = 32
		}
	}
	if o.Timeout <= 0 {
		o.Timeout = 15 * time.Minute
	}
	if o.Backend == "metal" && runtime.GOOS != "darwin" {
		return refuse("-backend metal needs darwin (this is %s)", runtime.GOOS)
	}
	if o.Backend == "cuda" && runtime.GOOS != "linux" && runtime.GOOS != "windows" {
		return refuse("-backend cuda needs linux or windows (this is %s)", runtime.GOOS)
	}
	if o.Giw == "bytes" {
		return identityGiwBytes(o, w)
	}
	old := &identitySide{Label: "old", Rev: o.OldRev}
	nw := &identitySide{Label: "new", Rev: o.NewRev}
	if why := resolveIdentitySides(o, old, nw); why != "" {
		return refuse("%s", why)
	}

	fams, err := readManifestFamilies(o.FixtureRoot)
	if err != nil {
		return refuse("%v", err)
	}
	cells, selected, err := planIdentityCells(&o, fams)
	if err != nil {
		return refuse("%v", err)
	}
	runnable := 0
	for _, c := range cells {
		if o.Giw == "logits" && c.NotRun == "" && strings.HasSuffix(c.Path, ".giw") {
			c.NotRun = "already a .giw: -giw logits transcodes a source at each revision"
		}
		if c.NotRun == "" {
			runnable++
		}
	}

	wd, logs, cleanup, err := identityRunDir(o, t0, old, nw, "")
	if err != nil {
		return refuse("%v", err)
	}
	defer cleanup()

	scrubbed := identityScrubbedVars()
	prov := gatherProvenanceIn(o.Repo, [][2]string{
		{"old", fmt.Sprintf("%s %s", old.Short, trunc(old.Title, 70))},
		{"new", fmt.Sprintf("%s %s", nw.Short, trunc(nw.Title, 70))},
		{"backend", o.Backend},
		{"mode", identityModeDesc(o)},
		{"assets", identityAssetsDesc(o)},
		{"quants", strings.Join(orDefaultQuants(o), ",")},
		{"steps", fmt.Sprintf("%d per prompt (the first dumped step is the prefill's logits)", o.Steps)},
		{"env", fmt.Sprintf("GOINFER_* removed from both sides' dumpers (%s) — the code is the only difference", scrubbedDesc(scrubbed))},
		{"build", "go build -trimpath; " + identityBuildDesc(o.Backend)},
		{"run dir", wd},
		{"started", t0.Format("15:04:05 MST")},
	})
	// The verdict is about two COMMITS; a dirty main tree does not change what either build compiles
	// (only the fixtures are read from it, the same bytes for both sides).
	prov.Dirty = false
	prov.Fields = append(prov.Fields, [2]string{"tree", "the revs are commits, built in temporary worktrees; the main tree supplies only the fixtures, identically to both"})
	fmt.Fprintf(w, "%s== identity provenance ==%s\n", bold, off)
	prov.write(w)
	if runnable == 0 {
		fmt.Fprintf(w, "\n")
		printNotRun(w, cells)
		return refuse("no selected cell has an asset on this host — nothing would be compared (a skip is not a pass)")
	}

	// ---- worktrees + builds (both sides in parallel) ----
	fmt.Fprintf(progress, "== identity: %d cell(s) over %d family(ies), started %s — building %s and %s ==\n",
		runnable, len(selected), t0.Format("15:04:05"), old.Short, nw.Short)
	tb := time.Now()
	if failed := buildIdentitySides(o, wd, old, nw); failed != nil {
		fmt.Fprintf(w, "\n  %s%s build (%s) failed%s: %v\n", red, failed.Label, failed.Short, off, failed.Err)
		for _, ln := range tailLines(failed.Log, 20) {
			fmt.Fprintf(w, "    %s\n", ln)
		}
		return refuse("the dumper did not build at %s %s — see above (an old rev may predate the public API it uses)", failed.Label, failed.Short)
	}
	buildWall := time.Since(tb)
	fmt.Fprintf(progress, "   built both in %s (old %s, new %s)\n", buildWall.Round(time.Second), old.BuildT.Round(time.Second), nw.BuildT.Round(time.Second))

	// ---- runs: per family, new#1 → old → new#2 ----
	byFam := map[string][]*identityCell{}
	var famOrder []string
	for _, c := range cells {
		if c.NotRun != "" {
			continue
		}
		if _, ok := byFam[c.Family]; !ok {
			famOrder = append(famOrder, c.Family)
		}
		byFam[c.Family] = append(byFam[c.Family], c)
	}
	runs := []identityRun{{"new1", nw}, {"old", old}, {"new2", nw}}
	crash := map[string]string{} // run/family -> why the process ended badly
	tr := time.Now()
	stopBeat := identityHeartbeat(progress, t0)
	for i, f := range famOrder {
		identityBeatState.Store("state", fmt.Sprintf("running %s (%d/%d families, %d done)", f, i+1, len(famOrder), i))
		ft := time.Now()
		var parts []string
		for _, r := range runs {
			rt := time.Now()
			var paths map[string]string
			var giwDirs []string
			if o.Giw == "logits" {
				paths, giwDirs = giwTranscodeCells(o, wd, r, f, byFam[f])
			}
			why := runIdentityDumper(o, wd, r, f, byFam[f], paths)
			if !o.Keep {
				for _, d := range giwDirs {
					_ = os.RemoveAll(d)
				}
			}
			if why != "" {
				crash[r.Name+"/"+f] = why
			}
			st := fmt.Sprintf("%s %s", r.Name, time.Since(rt).Round(100*time.Millisecond))
			if why != "" {
				st += " (" + trunc(why, 60) + ")"
			}
			parts = append(parts, st)
		}
		fmt.Fprintf(progress, "   [%d/%d] %-18s %d cell(s): %s — %s total, elapsed %s (started %s)\n", i+1, len(famOrder), f,
			len(byFam[f]), strings.Join(parts, ", "), time.Since(ft).Round(100*time.Millisecond), time.Since(t0).Round(time.Second), t0.Format("15:04:05"))
	}
	stopBeat()
	runWall := time.Since(tr)

	// ---- judge ----
	var outs []cellOutcome
	missing := map[string]string{} // run/cell -> why that dump was not a result, for the run record
	for _, c := range cells {
		if c.NotRun != "" {
			outs = append(outs, judgeCell(c, o.Backend, nil, nil, nil))
			continue
		}
		dir := func(run string) string { return filepath.Join(wd, "dumps", run, c.Family) }
		n1 := readDump(dir("new1"), c.ID, crash["new1/"+c.Family])
		od := readDump(dir("old"), c.ID, crash["old/"+c.Family])
		n2 := readDump(dir("new2"), c.ID, crash["new2/"+c.Family])
		for run, d := range map[string]*dump{"new1": n1, "old": od, "new2": n2} {
			if d.Missing != "" {
				missing[run+"/"+c.ID] = d.Missing
			}
		}
		outs = append(outs, judgeCell(c, o.Backend, n1, n2, od))
	}
	head, _ := gitOut(o.Repo, "rev-parse", "HEAD")
	in := identityGradeInputs{T0: t0, BuildWall: buildWall, RunWall: runWall, Logs: logs,
		Host: strings.ReplaceAll(gatherProvenanceIn(o.Repo, nil).Host, " ", "/"), Head: strings.TrimSpace(head),
		Arch: identityRecordArch(), Date: time.Now().Format("2006-01-02")}
	var rowsErr string
	writeRows := func(rows []string) error {
		err := os.WriteFile(o.Record, []byte(strings.Join(rows, "\n")+"\n"), 0o644)
		if err != nil {
			rowsErr = err.Error()
		}
		return err
	}
	in.End = time.Now()
	w = out
	rc := gradeIdentity(w, o, old, nw, fams, selected, outs, in, writeRows)
	writeIdentityRunRecord(os.Stderr, wd, prelude.String(), o, old, nw, fams, selected, cells, crash, missing, in, rowsErr, rc)
	return rc
}

// resolveIdentitySides fills each side's SHA, short SHA and title from its rev; a non-empty return is why it refused.
func resolveIdentitySides(o identityOpts, sides ...*identitySide) string {
	for _, s := range sides {
		sha, err := gitOut(o.Repo, "rev-parse", "--verify", "--quiet", s.Rev+"^{commit}")
		if err != nil || sha == "" {
			return fmt.Sprintf("%s rev %q does not resolve to a commit in %s", s.Label, s.Rev, o.Repo)
		}
		s.SHA = sha
		s.Short, _ = gitOut(o.Repo, "rev-parse", "--short", sha)
		s.Title, _ = gitOut(o.Repo, "log", "-1", "--format=%s", sha)
	}
	return ""
}

// identityRunDir creates the run directory and its logs/, and returns the cleanup that removes the sides' worktrees
// (always) and the run's bulky output (unless -keep). The cleanup also runs on SIGINT and SIGTERM, which a deferred call
// would not; calling it twice is safe. extra names one more subdirectory to remove.
func identityRunDir(o identityOpts, t0 time.Time, old, nw *identitySide, extra string) (wd, logs string, cleanup func(), err error) {
	if err := os.MkdirAll(o.LogDir, 0o755); err != nil {
		return "", "", nil, fmt.Errorf("logdir: %v", err)
	}
	wd, err = os.MkdirTemp(o.LogDir, "gate-identity-"+t0.Format("20060102-150405")+"-")
	if err != nil {
		return "", "", nil, fmt.Errorf("run dir: %v", err)
	}
	if r, err := filepath.EvalSymlinks(wd); err == nil {
		wd = r
	}
	logs = filepath.Join(wd, "logs")
	_ = os.MkdirAll(logs, 0o755)
	var once sync.Once
	remove := func() {
		once.Do(func() {
			for _, s := range []*identitySide{old, nw} {
				if s.WT == "" {
					continue
				}
				if _, err := gitOut(o.Repo, "worktree", "remove", "--force", s.WT); err != nil {
					_ = os.RemoveAll(s.WT)
				}
			}
			_, _ = gitOut(o.Repo, "worktree", "prune")
			if !o.Keep {
				for _, sub := range []string{"dumps", "bin", "spec", "giw", extra} {
					if sub != "" {
						_ = os.RemoveAll(filepath.Join(wd, sub))
					}
				}
			}
		})
	}
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigs:
			fmt.Fprintln(os.Stderr, "gate identity: interrupted — removing the worktrees")
			remove()
			os.Exit(130)
		case <-done:
		}
	}()
	return wd, logs, func() { remove(); close(done); signal.Stop(sigs) }, nil
}

// buildIdentitySides builds both sides in parallel and returns the first that failed, or nil.
func buildIdentitySides(o identityOpts, wd string, old, nw *identitySide) *identitySide {
	var wg sync.WaitGroup
	for _, s := range []*identitySide{old, nw} {
		wg.Add(1)
		go func(s *identitySide) {
			defer wg.Done()
			s.Err = buildIdentitySide(o, wd, s)
		}(s)
	}
	wg.Wait()
	for _, s := range []*identitySide{old, nw} {
		if s.Err != nil {
			return s
		}
	}
	return nil
}

func orDefaultQuants(o identityOpts) []string {
	if len(o.Quants) > 0 {
		return o.Quants
	}
	return defaultIdentityQuants(o.Backend, o.Assets)
}

func scrubbedDesc(xs []string) string {
	if len(xs) == 0 {
		return "none were set"
	}
	return strings.Join(xs, ", ")
}

func identityAssetsDesc(o identityOpts) string {
	if o.Assets == "real" {
		return "real — checkpoints under " + o.ModelsDir + ", L1's two prose prompts tokenized by this build"
	}
	return "tiny — the forward goldens' fixtures under " + o.FixtureRoot + ", each golden's prompt ids (+ a longer cycled prompt where the context allows)"
}

func identityModeDesc(o identityOpts) string {
	if o.Giw == "logits" {
		return "giw logits: each revision's own cmd/prequant transcodes the cell's source for target " + giwTargetForBackend(o.Backend) + "; the dumper loads that file"
	}
	return "logits from each cell's own checkpoint"
}

func identityBuildDesc(backend string) string {
	switch backend {
	case "metal":
		return "dumper in metal/ (registers the backend), ephemeral go.work: . ./metal"
	case "webgpu":
		return "dumper in gpu/ with -tags gpu, ephemeral go.work: . ./gpu"
	case "cuda":
		return "dumper in cuda/ with -tags cuda, ephemeral go.work: . ./cuda"
	}
	return "dumper in the root module, GOWORK=off -mod=readonly (CI's root build)"
}

// identityScrubbedVars names the GOINFER_* variables set in this process, which the dumpers do not see.
func identityScrubbedVars() []string {
	var out []string
	for _, kv := range os.Environ() {
		if k, _, _ := strings.Cut(kv, "="); strings.HasPrefix(k, "GOINFER_") {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// identityEnv is the environment every build and dumper runs in: this process's, minus GOWORK,
// GOFLAGS and every GOINFER_* variable, plus set.
func identityEnv(set map[string]string) []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		if k == "GOWORK" || k == "GOFLAGS" || strings.HasPrefix(k, "GOINFER_") {
			continue
		}
		if _, ours := set[k]; ours {
			continue
		}
		env = append(env, kv)
	}
	for k, v := range set {
		env = append(env, k+"="+v)
	}
	sort.Strings(env)
	return env
}

func gitOut(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// buildIdentitySide checks out s's revision in a detached worktree and builds the dumper in it.
func buildIdentitySide(o identityOpts, wd string, s *identitySide) error {
	t0 := time.Now()
	defer func() { s.BuildT = time.Since(t0) }()
	s.Log = filepath.Join(wd, "logs", "build-"+s.Label+".log")
	s.WT = filepath.Join(wd, "wt-"+s.Label)
	if _, err := gitOut(o.Repo, "worktree", "add", "--detach", "--quiet", s.WT, s.SHA); err != nil {
		s.WT = ""
		return err
	}
	if o.Giw != "" {
		if err := buildIdentityPrequant(o, wd, s); err != nil {
			return err
		}
		if o.Giw == "bytes" {
			return nil // no dumper: the files themselves are compared
		}
	}
	modDir, register, tags := s.WT, "", ""
	set := map[string]string{}
	switch o.Backend {
	case "metal":
		modDir, register = filepath.Join(s.WT, "metal"), `_ "github.com/townsendmerino/goinfer/metal"`
	case "webgpu":
		modDir, register, tags = filepath.Join(s.WT, "gpu"), `_ "github.com/townsendmerino/goinfer/gpu"`, "gpu"
	case "cuda":
		modDir, register, tags = filepath.Join(s.WT, "cuda"), `_ "github.com/townsendmerino/goinfer/cuda"`, "cuda"
	}
	if o.Backend == "cpu" {
		set["GOWORK"], set["GOFLAGS"] = "off", "-mod=readonly"
	} else {
		gov := "1.21"
		if b, err := os.ReadFile(filepath.Join(s.WT, "go.mod")); err == nil {
			for ln := range strings.SplitSeq(string(b), "\n") {
				if f := strings.Fields(ln); len(f) == 2 && f[0] == "go" {
					gov = f[1]
				}
			}
		}
		work := filepath.Join(s.WT, "go.work")
		body := fmt.Sprintf("go %s\n\nuse (\n\t.\n\t./%s\n)\n", gov, filepath.Base(modDir))
		if err := os.WriteFile(work, []byte(body), 0o644); err != nil {
			return err
		}
		set["GOWORK"] = work
	}
	src := o.DumperSource
	if src == "" {
		src = identityDumperTmpl
	}
	src = strings.Replace(src, "//REGISTER", register, 1)
	dir := filepath.Join(modDir, "cmd", "zzgateidentity")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o644); err != nil {
		return err
	}
	_ = os.MkdirAll(filepath.Join(wd, "bin"), 0o755)
	s.Bin = filepath.Join(wd, "bin", "dump-"+s.Label)
	args := []string{"build", "-trimpath", "-o", s.Bin}
	if tags != "" {
		args = append(args, "-tags", tags)
	}
	args = append(args, "./cmd/zzgateidentity")
	cmd := exec.Command("go", args...)
	cmd.Dir = modDir
	cmd.Env = identityEnv(set)
	out, err := cmd.CombinedOutput()
	_ = os.WriteFile(s.Log, append([]byte("go "+strings.Join(args, " ")+"\n"), out...), 0o644)
	if err != nil {
		return fmt.Errorf("go build: %v", err)
	}
	return nil
}

// runIdentityDumper runs one side's dumper over one family's cells, into dumps/<run>/<family>/. It
// returns "" on a clean exit, else why the process did not finish (its cells then have no result
// and are judged on that). A non-nil paths maps a cell's ID to the checkpoint this run loads for it in
// place of the cell's own, and a cell it does not name is left out.
func runIdentityDumper(o identityOpts, wd string, r identityRun, family string, cells []*identityCell, paths map[string]string) string {
	out := filepath.Join(wd, "dumps", r.Name, family)
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err.Error()
	}
	type cellSpec struct {
		ID         string  `json:"id"`
		Path       string  `json:"path"`
		Quant      string  `json:"quant"`
		Prompts    [][]int `json:"prompts"`
		DirectLoad bool    `json:"direct_load,omitempty"`
	}
	spec := struct {
		Backend string     `json:"backend"`
		Steps   int        `json:"steps"`
		Out     string     `json:"out"`
		Cells   []cellSpec `json:"cells"`
	}{Backend: o.Backend, Steps: o.Steps, Out: out}
	for _, c := range cells {
		path := c.Path
		if paths != nil { // -giw logits: this side's own transcode of the cell; a cell it could not transcode has its failure written already
			var ok bool
			if path, ok = paths[c.ID]; !ok {
				continue
			}
		}
		spec.Cells = append(spec.Cells, cellSpec{c.ID, path, c.Quant, c.Prompts, c.DirectLoad})
	}
	if len(spec.Cells) == 0 {
		return ""
	}
	b, _ := json.Marshal(spec)
	sp := filepath.Join(wd, "spec", r.Name+"-"+sanitize(family)+".json")
	_ = os.MkdirAll(filepath.Dir(sp), 0o755)
	if err := os.WriteFile(sp, b, 0o644); err != nil {
		return err.Error()
	}
	logPath := filepath.Join(wd, "logs", r.Name+"-"+sanitize(family)+".log")
	logf, err := os.Create(logPath)
	if err != nil {
		return err.Error()
	}
	defer logf.Close()
	ctx, cancel := context.WithTimeout(context.Background(), o.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, r.Side.Bin, sp)
	cmd.Dir = wd
	cmd.Env = identityEnv(nil)
	cmd.Stdout, cmd.Stderr = logf, logf
	err = cmd.Run()
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		return fmt.Sprintf("timed out after %s (log %s)", o.Timeout, logPath)
	case err != nil:
		tail := tailLines(logPath, 3)
		return fmt.Sprintf("dumper exited: %v; %s (log %s)", err, strings.Join(tail, " | "), logPath)
	}
	return ""
}

// identityBeatState is what the heartbeat reports between family lines.
var identityBeatState sync.Map

func identityHeartbeat(w io.Writer, t0 time.Time) (stop func()) {
	every := 30 * time.Second
	if v := os.Getenv("GOINFER_GATE_HEARTBEAT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			every = d
		}
	}
	if every <= 0 {
		return func() {}
	}
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				st, _ := identityBeatState.Load("state")
				if st == nil {
					st = "first family running"
				}
				fmt.Fprintf(w, "   ... identity: %s elapsed (started %s), %v\n", time.Since(t0).Round(time.Second), t0.Format("15:04:05"), st)
			}
		}
	}()
	return func() { close(done); <-finished }
}

// ---- report ----

type identityFamily struct {
	Name     string
	Verdict  string
	Cells    []cellOutcome
	Ran      int
	NotRun   int
	NonDet   int
	DetCheck int
}

func readDump(dir, id, crash string) *dump {
	b, err := os.ReadFile(filepath.Join(dir, id+".json"))
	if err != nil {
		why := "no result written"
		if crash != "" {
			why += " — " + crash
		}
		return &dump{Missing: why}
	}
	var m dumpMeta
	if err := json.Unmarshal(b, &m); err != nil {
		return &dump{Missing: "unreadable result: " + err.Error()}
	}
	d := &dump{Meta: &m}
	bin, err := os.ReadFile(filepath.Join(dir, id+".bin"))
	if m.OK {
		if err != nil {
			return &dump{Missing: "the result's logits are unreadable: " + err.Error()}
		}
		if want := dumpBinSize(&m); len(bin) != want {
			return &dump{Missing: fmt.Sprintf("the result's logits are %d bytes, its prompts' steps call for %d", len(bin), want)}
		}
	}
	d.Bin = bin
	return d
}
