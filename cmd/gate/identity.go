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
	ModelsDir   string
	LogDir      string
	Keep        bool
	Record      string        // -record: write eligible families' PARITY_ROW lines here (identity_record.go)
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
	var fams, quants, only string
	fs.StringVar(&o.Backend, "backend", "cpu", "cpu | metal | webgpu")
	fs.StringVar(&fams, "families", "", "comma-separated parity-manifest families (default: all)")
	fs.StringVar(&o.Assets, "assets", "tiny", "tiny (the forward goldens' fixtures) | real (small checkpoints under ~/models)")
	fs.StringVar(&quants, "quant", "", "comma-separated quants (default: cpu f32,int8int8,int4; metal int4,int8int8; webgpu int4; -assets real int4)")
	fs.StringVar(&only, "only", "", "comma-separated asset names to keep (e.g. coder-0.5b-q4km)")
	fs.IntVar(&o.Steps, "steps", 0, "tokens generated per prompt; logits are dumped for each (default: tiny 8, real 32)")
	fs.StringVar(&o.LogDir, "logdir", os.TempDir(), "where the run directory (worktrees, builds, dumps, logs) is created")
	fs.BoolVar(&o.Keep, "keep", false, "keep the dumps, binaries and logs (worktrees are always removed)")
	fs.StringVar(&o.Record, "record", "", "write PARITY_ROW lines for families whose T3 validation this run inherits (TE6(b)); merge them with the decoder's TestParityManifest_merge")
	fs.DurationVar(&o.Timeout, "timeout", 15*time.Minute, "per dumper process")
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
	if len(pos) != 2 {
		fmt.Fprintln(os.Stderr, "usage: gate identity <old-rev> <new-rev> [-backend cpu|metal|webgpu] [-families a,b] [-assets tiny|real] [-quant q,…] [-steps N] [-keep]")
		return 2
	}
	o.OldRev, o.NewRev = pos[0], pos[1]
	o.Families, o.Quants, o.Only = splitList(fams), splitList(quants), splitList(only)
	switch o.Backend {
	case "cpu", "metal", "webgpu":
	default:
		fmt.Fprintf(os.Stderr, "gate identity: -backend %q: want cpu, metal or webgpu\n", o.Backend)
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
	Label  string // old | new
	Rev    string // as given
	SHA    string
	Short  string
	Title  string
	WT     string
	Bin    string
	BuildT time.Duration
	Err    error
	Log    string
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
	old := &identitySide{Label: "old", Rev: o.OldRev}
	nw := &identitySide{Label: "new", Rev: o.NewRev}
	for _, s := range []*identitySide{old, nw} {
		sha, err := gitOut(o.Repo, "rev-parse", "--verify", "--quiet", s.Rev+"^{commit}")
		if err != nil || sha == "" {
			return refuse("%s rev %q does not resolve to a commit in %s", s.Label, s.Rev, o.Repo)
		}
		s.SHA = sha
		s.Short, _ = gitOut(o.Repo, "rev-parse", "--short", sha)
		s.Title, _ = gitOut(o.Repo, "log", "-1", "--format=%s", sha)
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
		if c.NotRun == "" {
			runnable++
		}
	}

	if err := os.MkdirAll(o.LogDir, 0o755); err != nil {
		return refuse("logdir: %v", err)
	}
	wd, err := os.MkdirTemp(o.LogDir, "gate-identity-"+t0.Format("20060102-150405")+"-")
	if err != nil {
		return refuse("run dir: %v", err)
	}
	if r, err := filepath.EvalSymlinks(wd); err == nil {
		wd = r
	}
	logs := filepath.Join(wd, "logs")
	_ = os.MkdirAll(logs, 0o755)

	// Cleanup: worktrees always; the rest unless -keep. A deferred call does not run on a signal, so
	// the handler is explicit (the same shape as gate mutation's restore).
	var once sync.Once
	cleanup := func() {
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
				for _, sub := range []string{"dumps", "bin", "spec"} {
					_ = os.RemoveAll(filepath.Join(wd, sub))
				}
			}
		})
	}
	defer cleanup()
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		select {
		case <-sigs:
			fmt.Fprintln(os.Stderr, "gate identity: interrupted — removing the worktrees")
			cleanup()
			os.Exit(130)
		case <-done:
		}
	}()
	defer func() { close(done); signal.Stop(sigs) }()

	scrubbed := identityScrubbedVars()
	prov := gatherProvenanceIn(o.Repo, [][2]string{
		{"old", fmt.Sprintf("%s %s", old.Short, trunc(old.Title, 70))},
		{"new", fmt.Sprintf("%s %s", nw.Short, trunc(nw.Title, 70))},
		{"backend", o.Backend},
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
			fmt.Fprintf(w, "\n  %s%s build (%s) failed%s: %v\n", red, s.Label, s.Short, off, s.Err)
			for _, ln := range tailLines(s.Log, 20) {
				fmt.Fprintf(w, "    %s\n", ln)
			}
			return refuse("the dumper did not build at %s %s — see above (an old rev may predate the public API it uses)", s.Label, s.Short)
		}
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
			why := runIdentityDumper(o, wd, r, f, byFam[f])
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
	for _, c := range cells {
		if c.NotRun != "" {
			outs = append(outs, judgeCell(c, o.Backend, nil, nil, nil))
			continue
		}
		dir := func(run string) string { return filepath.Join(wd, "dumps", run, c.Family) }
		n1 := readDump(dir("new1"), c.ID, crash["new1/"+c.Family])
		od := readDump(dir("old"), c.ID, crash["old/"+c.Family])
		n2 := readDump(dir("new2"), c.ID, crash["new2/"+c.Family])
		outs = append(outs, judgeCell(c, o.Backend, n1, n2, od))
	}
	return reportIdentity(w, o, old, nw, fams, selected, outs, t0, buildWall, runWall, logs)
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

func identityBuildDesc(backend string) string {
	switch backend {
	case "metal":
		return "dumper in metal/ (registers the backend), ephemeral go.work: . ./metal"
	case "webgpu":
		return "dumper in gpu/ with -tags gpu, ephemeral go.work: . ./gpu"
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
	modDir, register, tags := s.WT, "", ""
	set := map[string]string{}
	switch o.Backend {
	case "metal":
		modDir, register = filepath.Join(s.WT, "metal"), `_ "github.com/townsendmerino/goinfer/metal"`
	case "webgpu":
		modDir, register, tags = filepath.Join(s.WT, "gpu"), `_ "github.com/townsendmerino/goinfer/gpu"`, "gpu"
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
// and are judged on that).
func runIdentityDumper(o identityOpts, wd string, r identityRun, family string, cells []*identityCell) string {
	out := filepath.Join(wd, "dumps", r.Name, family)
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err.Error()
	}
	type cellSpec struct {
		ID      string  `json:"id"`
		Path    string  `json:"path"`
		Quant   string  `json:"quant"`
		Prompts [][]int `json:"prompts"`
	}
	spec := struct {
		Backend string     `json:"backend"`
		Steps   int        `json:"steps"`
		Out     string     `json:"out"`
		Cells   []cellSpec `json:"cells"`
	}{Backend: o.Backend, Steps: o.Steps, Out: out}
	for _, c := range cells {
		spec.Cells = append(spec.Cells, cellSpec{c.ID, c.Path, c.Quant, c.Prompts})
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

func summarizeFamilies(selected []string, outs []cellOutcome) []*identityFamily {
	by := map[string]*identityFamily{}
	for _, f := range selected {
		by[f] = &identityFamily{Name: f}
	}
	for _, c := range outs {
		f := by[c.Cell.Family]
		f.Cells = append(f.Cells, c)
		if c.Verdict == vNotRun {
			f.NotRun++
		} else {
			f.Ran++
		}
		if c.DetChecked {
			f.DetCheck++
			if !c.Deterministic {
				f.NonDet++
			}
		}
	}
	var out []*identityFamily
	for _, name := range selected {
		f := by[name]
		f.Verdict = vNotRun
		if f.Ran > 0 {
			f.Verdict = vIdentical
		}
		for _, c := range f.Cells {
			switch c.Verdict {
			case vDifferent:
				f.Verdict = vDifferent
			case vTolerant:
				if f.Verdict == vIdentical {
					f.Verdict = vTolerant
				}
			}
		}
		out = append(out, f)
	}
	return out
}

func reportIdentity(w io.Writer, o identityOpts, old, nw *identitySide, manifest map[string]manifestFamily, selected []string,
	outs []cellOutcome, t0 time.Time, buildWall, runWall time.Duration, logs string) int {
	fams := summarizeFamilies(selected, outs)

	fmt.Fprintf(w, "\n%s== determinism: the new build (%s) run twice, separate processes, on %s ==%s\n", bold, nw.Short, o.Backend, off)
	det, nondet, unchecked := 0, 0, 0
	for _, c := range outs {
		switch {
		case c.Verdict == vNotRun:
		case !c.DetChecked:
			unchecked++
		case c.Deterministic:
			det++
		default:
			nondet++
			fmt.Fprintf(w, "  %sNONDETERMINISTIC%s %-44s %s\n", red, off, c.Cell.ID, c.DetDiff.describe())
		}
	}
	fmt.Fprintf(w, "  %d cell(s) byte-identical to themselves, %d not, %d unchecked (second run failed)\n", det, nondet, unchecked)
	if nondet > 0 {
		if o.Backend == "cpu" {
			fmt.Fprintf(w, "  %sKILL CRITERION (TE6(b)): run-to-run nondeterminism on the CPU — impossible under the bit-identical discipline, and a finding in its own right.%s\n", red, off)
		} else {
			fmt.Fprintf(w, "  %s%s is not bitwise deterministic on %d cell(s): those are compared in TOLERANCE MODE (max |diff| <= %.0fx the cell's own run-to-run max |diff|, argmax and tokens exact) and reported WITHIN TOLERANCE, never IDENTICAL.%s\n",
				amber, o.Backend, nondet, identityTolFactor, off)
		}
	}

	fmt.Fprintf(w, "\n%s== identity: old %s vs new %s, per family ==%s\n", bold, old.Short, nw.Short, off)
	nIdent, nTol, nDiff, nNR := 0, 0, 0, 0
	for _, f := range fams {
		col := green
		switch f.Verdict {
		case vDifferent:
			col = red
			nDiff++
		case vTolerant:
			col = amber
			nTol++
		case vNotRun:
			col = amber
			nNR++
		default:
			nIdent++
		}
		cov := fmt.Sprintf("%d/%d cell(s) ran", f.Ran, f.Ran+f.NotRun)
		switch f.Verdict {
		case vIdentical:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s validation inheritable from %s\n", col, f.Verdict, off, f.Name, cov, old.Short)
		case vTolerant:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s not byte-identical (nondeterministic backend); tolerance-inheritable only if the owner accepts it\n", col, f.Verdict, off, f.Name, cov)
		case vDifferent:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %-16s goes to its reference gate\n", col, f.Verdict, off, f.Name, cov)
		case vNotRun:
			fmt.Fprintf(w, "  %s%-16s%s %-18s %s\n", col, f.Verdict, off, f.Name, "nothing ran — see NOT RUN")
			continue
		}
		for _, c := range f.Cells {
			if c.Verdict == vNotRun {
				continue
			}
			line := fmt.Sprintf("      %-10s %-38s %s", c.Verdict, c.Cell.Asset+" @ "+c.Cell.Quant, orDash(c.NewPath))
			if c.Verdict != vIdentical || strings.TrimSpace(c.Detail) != "" {
				line += " — " + strings.TrimSpace(c.Detail)
			}
			fmt.Fprintln(w, line)
		}
	}

	printNotRun(w, cellsOf(outs))

	fmt.Fprintf(w, "\n%s== would record (the parity manifest is NOT written by this item) ==%s\n", bold, off)
	host := strings.ReplaceAll(gatherProvenanceIn(o.Repo, nil).Host, " ", "/")
	anyRec := false
	var notValidated []string
	for _, f := range fams {
		if f.Verdict != vIdentical {
			continue
		}
		anyRec = true
		mf := manifest[f.Name]
		va := mf.ValidatedAt
		if va == "" {
			va = "never"
		}
		if mf.ValidatedAt == "" || !strings.HasPrefix(old.SHA, mf.ValidatedAt) {
			notValidated = append(notValidated, f.Name+"@"+va)
		}
		fmt.Fprintf(w, "  %-18s method \"identity-inherited from %s\" at %s — %s, %s, %s assets, %d cell(s); today: %s @ %s (%s)\n",
			f.Name, old.Short, nw.Short, o.Backend, host, o.Assets, f.Ran, mf.Method, va, mf.Machine)
	}
	if !anyRec {
		fmt.Fprintf(w, "  nothing — no family came back IDENTICAL\n")
	}
	if len(notValidated) > 0 {
		fmt.Fprintf(w, "  %sNOTE%s old %s is not the validated_at of %d of these: identity inherits only what %s itself carries —\n"+
			"       chain through the revs that were checked, or pass the family's validated_at as <old-rev>:\n       %s\n",
			amber, off, old.Short, len(notValidated), old.Short, wrapWords(notValidated, 96, "       "))
	}
	fmt.Fprintf(w, "  scope: this machine (%s) and backend (%s) only — identity is per machine and per backend (the CPU reference is bit-identical within an arch, not across).\n", host, o.Backend)

	if o.Record != "" {
		head, _ := gitOut(o.Repo, "rev-parse", "HEAD")
		var rows []string
		fmt.Fprintf(w, "\n%s== inheritance (-record %s) ==%s\n", bold, o.Record, off)
		for _, f := range fams {
			if f.Verdict == vNotRun {
				continue
			}
			in := identityInheritance(o, old, nw, strings.TrimSpace(head), identityRecordArch(), host, manifest[f.Name], *f)
			if in.Row != "" {
				rows = append(rows, in.Row)
				fmt.Fprintf(w, "  %sINHERITS%s %-18s\n", green, off, f.Name)
				continue
			}
			fmt.Fprintf(w, "  not      %-18s %s\n", f.Name, strings.Join(in.Reasons, "; "))
		}
		if len(rows) == 0 {
			fmt.Fprintf(w, "  nothing eligible; %s not written\n", o.Record)
		} else if err := os.WriteFile(o.Record, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
			fmt.Fprintf(w, "  %swrite %s: %v%s\n", red, o.Record, err, off)
		} else {
			fmt.Fprintf(w, "  wrote %d row(s) to %s. Before merging, check each row's checkpoint against its original reference, then:\n"+
				"    GOINFER_MANIFEST_MACHINE=<the row's machine name> go test ./decoder/ -run TestParityManifest_merge -merge-rows %s\n",
				len(rows), o.Record, o.Record)
		}
	}

	end := time.Now()
	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  families %d: %d IDENTICAL, %d WITHIN TOLERANCE, %d DIFFERENT, %d NOT RUN (a skip is not a pass)\n", len(fams), nIdent, nTol, nDiff, nNR)
	fmt.Fprintf(w, "  wall     %s (%s → %s): builds %s, runs %s\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"), end.Format("15:04:05"),
		buildWall.Round(time.Second), runWall.Round(time.Second))
	keep := "removed (pass -keep to retain dumps)"
	if o.Keep {
		keep = "kept"
	}
	fmt.Fprintf(w, "  logs     %s (dumps/binaries %s; worktrees removed)\n", logs, keep)
	switch {
	case nondet > 0 && o.Backend == "cpu":
		fmt.Fprintf(w, "  %sRED — the CPU is not deterministic run to run on %d cell(s)%s\n", red, nondet, off)
		return 1
	case nDiff > 0:
		fmt.Fprintf(w, "  %sRED — %d family(ies) DIFFERENT: they, and only they, go to their reference gates%s\n", red, nDiff, off)
		return 1
	case nIdent+nTol == 0:
		fmt.Fprintf(w, "  %sREFUSED — no family ran%s\n", amber, off)
		return 2
	}
	fmt.Fprintf(w, "  %sGREEN — every family that ran is %s%s\n", green, map[bool]string{true: "IDENTICAL or WITHIN TOLERANCE", false: "IDENTICAL"}[nTol > 0], off)
	return 0
}

func cellsOf(outs []cellOutcome) []*identityCell {
	var cs []*identityCell
	for _, c := range outs {
		if c.Verdict == vNotRun {
			cc := *c.Cell
			cc.NotRun = c.Detail
			cs = append(cs, &cc)
		}
	}
	return cs
}

func printNotRun(w io.Writer, cells []*identityCell) {
	var nr []*identityCell
	for _, c := range cells {
		if c.NotRun != "" {
			nr = append(nr, c)
		}
	}
	if len(nr) == 0 {
		return
	}
	fmt.Fprintf(w, "\n%s== NOT RUN (%d cell(s)) — reported, never counted as a pass ==%s\n", bold, len(nr), off)
	// Collapse the per-quant copies of one reason.
	type key struct{ fam, asset, why string }
	seen := map[key][]string{}
	var order []key
	for _, c := range nr {
		k := key{c.Family, c.Asset, c.NotRun}
		if _, ok := seen[k]; !ok {
			order = append(order, k)
		}
		seen[k] = append(seen[k], c.Quant)
	}
	for _, k := range order {
		fmt.Fprintf(w, "  %s%-8s%s %-18s %-22s [%s] %s\n", amber, vNotRun, off, k.fam, k.asset, strings.Join(seen[k], ","), trunc(k.why, 150))
	}
}
