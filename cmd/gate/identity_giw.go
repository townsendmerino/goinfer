package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// `gate identity -giw bytes|logits <old-rev> <new-rev>`: identity through each revision's OWN transcode
// (docs/tasks/task-giw-v16-transcode-2026-10.md, T0).
//
// Each revision builds its own cmd/prequant in its worktree and transcodes the shared source into a directory that
// belongs to that revision and run.
//
//   - bytes: the two files' sha256 must be equal. For a change to the transcode that is meant to write the same bytes.
//   - logits: the existing dumper runs on each side's own file. For a format change: different bytes, the same numbers.
//
// identity's rules hold. The new build transcodes every cell twice and must agree with itself. A source that neither
// build transcodes is NOT RUN, listed with the reason, never counted as a pass; one that only one build transcodes is
// DIFFERENT. Both sides run in identityEnv.

// giwConfig is one transcode configuration, spelled quant:target[:e4h].
type giwConfig struct {
	Quant     string // prequant's -quant ("f32" is passed as "")
	Target    string // prequant's -target
	EmbedInt4 bool   // prequant's -embed-int4
}

func (c giwConfig) String() string {
	s := c.Quant + ":" + c.Target
	if c.EmbedInt4 {
		s += ":e4h"
	}
	return s
}

func parseGiwConfigs(s string) ([]giwConfig, error) {
	var out []giwConfig
	for _, item := range splitList(s) {
		f := strings.Split(item, ":")
		if len(f) < 2 || len(f) > 3 || f[0] == "" || f[1] == "" || (len(f) == 3 && f[2] != "e4h") {
			return nil, fmt.Errorf("-configs %q: want quant:target or quant:target:e4h (e.g. int4:metal:e4h)", item)
		}
		out = append(out, giwConfig{Quant: f[0], Target: f[1], EmbedInt4: len(f) == 3})
	}
	return out, nil
}

// defaultGiwConfigs is G-T1's matrix: by day on the tiny fixtures, G-DS1's configurations (int4 for Metal with and
// without the int4 embedding, int4 for this box's CPU, int8int8) plus int4 for cpu-arm64; on a real checkpoint, int4
// for this box's CPU and its GPU.
func defaultGiwConfigs(assets string) []giwConfig {
	if assets == "real" {
		gpu := "cuda"
		if runtime.GOOS == "darwin" {
			gpu = "metal"
		}
		return []giwConfig{{Quant: "int4", Target: "cpu"}, {Quant: "int4", Target: gpu}}
	}
	return []giwConfig{
		{Quant: "int4", Target: "metal", EmbedInt4: true},
		{Quant: "int4", Target: "metal"},
		{Quant: "int4", Target: "cpu"},
		{Quant: "int8int8", Target: "cpu"},
		{Quant: "int4", Target: "cpu-arm64"},
	}
}

// giwTargetForBackend is the target a load on backend reads: what -giw logits transcodes for.
func giwTargetForBackend(backend string) string {
	if backend == "cpu" {
		return "cpu" // prequant resolves it to this box's arch
	}
	return backend
}

// giwTinyCapBytes keeps a real checkpoint that lives under testdata out of the by-day matrix (a variable for the tests).
var giwTinyCapBytes int64 = 1 << 30

// giwSource is one checkpoint the bytes mode transcodes.
type giwSource struct {
	Name   string // unique, as the report prints it
	Path   string // absolute
	NotRun string
}

// planGiwSources lists the bytes mode's sources. Tiny: every directory holding a .safetensors file directly under
// testdata/ and decoder/testdata/ (G-DS1's walk; whether a directory is a model the transcode builds is decided by
// running it, and one neither build accepts is NOT RUN). Real: the -assets real table's GGUFs and directories.
func planGiwSources(o identityOpts) ([]giwSource, error) {
	keep := func(name string) bool { return len(o.Only) == 0 || slicesContains(o.Only, name) }
	var out []giwSource
	if o.Assets == "real" {
		seen := map[string]bool{}
		for _, a := range identityReal {
			if strings.HasSuffix(a.Path, ".giw") || seen[a.Path] || !keep(a.Name) {
				continue
			}
			if len(o.Families) > 0 && !slicesContains(o.Families, a.Family) {
				continue
			}
			seen[a.Path] = true
			s := giwSource{Name: a.Name, Path: filepath.Join(o.ModelsDir, a.Path)}
			if err := identityPathAllowed(s.Path); err != nil {
				return nil, err
			}
			s.NotRun = assetPresent(s.Path)
			out = append(out, s)
		}
		return out, nil
	}
	for _, sub := range []string{"testdata", filepath.Join("decoder", "testdata")} {
		ents, err := os.ReadDir(filepath.Join(o.FixtureRoot, sub))
		if err != nil {
			return nil, err
		}
		for _, e := range ents {
			dir := filepath.Join(o.FixtureRoot, sub, e.Name())
			if fi, err := os.Stat(dir); err != nil || !fi.IsDir() { // Stat: a fixture may be a symlink to a directory
				continue
			}
			if m, _ := filepath.Glob(filepath.Join(dir, "*.safetensors")); len(m) == 0 {
				continue
			}
			name := e.Name()
			if sub != "testdata" {
				name = "decoder/" + name
			}
			if !keep(name) {
				continue
			}
			s := giwSource{Name: name, Path: dir}
			if err := identityPathAllowed(dir); err != nil {
				return nil, err
			}
			if n := dirBytes(dir); n > giwTinyCapBytes {
				s.NotRun = fmt.Sprintf("%.1f GB: over the tiny matrix's %d GB cap (a real checkpoint under testdata)", float64(n)/1e9, giwTinyCapBytes>>30)
			}
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// dirBytes is the size of the regular files directly in dir, symlinks followed.
func dirBytes(dir string) int64 {
	ents, _ := os.ReadDir(dir)
	var n int64
	for _, e := range ents {
		if fi, err := os.Stat(filepath.Join(dir, e.Name())); err == nil && fi.Mode().IsRegular() {
			n += fi.Size()
		}
	}
	return n
}

// giwResult is one transcode by one build.
type giwResult struct {
	OK   bool
	SHA  string // sha256 of the written file
	Size int64
	Err  string // why it did not transcode
	Line string // prequant's own "wrote … in …: stream …, self-check …" line, when it prints one
	Wall time.Duration
}

// buildIdentityPrequant builds cmd/prequant at s's revision, in the root module of its worktree.
func buildIdentityPrequant(o identityOpts, wd string, s *identitySide) error {
	_ = os.MkdirAll(filepath.Join(wd, "bin"), 0o755)
	s.Prequant = filepath.Join(wd, "bin", "prequant-"+s.Label)
	args := []string{"build", "-trimpath", "-o", s.Prequant, "./cmd/prequant"}
	cmd := exec.Command("go", args...)
	cmd.Dir = s.WT
	cmd.Env = identityEnv(map[string]string{"GOWORK": "off", "GOFLAGS": "-mod=readonly"})
	out, err := cmd.CombinedOutput()
	log := filepath.Join(wd, "logs", "build-prequant-"+s.Label+".log")
	_ = os.WriteFile(log, append([]byte("go "+strings.Join(args, " ")+"\n"), out...), 0o644)
	if err != nil {
		s.Log = log
		return fmt.Errorf("go build ./cmd/prequant: %v", err)
	}
	return nil
}

// runGiwTranscode transcodes src with one side's prequant into outDir/s.giw and returns what was written. The file's
// name is the same on both sides; a bundle records its source's base name, never its own.
func runGiwTranscode(o identityOpts, bin, src, outDir string, c giwConfig, logPath string) (giwResult, string) {
	var r giwResult
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		r.Err = err.Error()
		return r, ""
	}
	out := filepath.Join(outDir, "s.giw")
	quant := c.Quant
	if quant == "f32" {
		quant = ""
	}
	args := []string{"-o", out, "-quant", quant, "-target", c.Target}
	if c.EmbedInt4 {
		args = append(args, "-embed-int4")
	}
	args = append(args, src)
	ctx, cancel := context.WithTimeout(context.Background(), o.Timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Env = identityEnv(nil)
	t0 := time.Now()
	b, err := cmd.CombinedOutput()
	r.Wall = time.Since(t0)
	_ = os.MkdirAll(filepath.Dir(logPath), 0o755)
	_ = os.WriteFile(logPath, append([]byte(filepath.Base(bin)+" "+strings.Join(args, " ")+"\n"), b...), 0o644)
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	last := strings.TrimSpace(lines[len(lines)-1])
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		r.Err = fmt.Sprintf("timed out after %s", o.Timeout)
		return r, out
	case err != nil:
		r.Err = strings.TrimPrefix(last, "prequant: ")
		if r.Err == "" {
			r.Err = err.Error()
		}
		return r, out
	}
	if strings.HasPrefix(last, "wrote ") {
		if i := strings.Index(last, " MB ("); i >= 0 {
			r.Line = last[strings.LastIndex(last[:i], " ")+1:] // from the size on: the path is this run's own
		}
	}
	sum, n, herr := sha256File(out)
	if herr != nil {
		r.Err = "transcode exited 0 and left no readable bundle: " + herr.Error()
		return r, out
	}
	r.OK, r.SHA, r.Size = true, sum, n
	return r, out
}

func sha256File(p string) (string, int64, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// judgeGiwBytes decides one cell from its three transcodes (new#1, new#2, old).
func judgeGiwBytes(n1, n2, old giwResult) (verdict, detail string) {
	switch {
	case !n1.OK && !old.OK:
		detail = "neither build transcodes it — new: " + n1.Err
		if old.Err != n1.Err {
			detail += " | old: " + old.Err
		}
		return vNotRun, detail
	case !n1.OK:
		return vDifferent, "the new build fails where the old transcodes: " + n1.Err
	case !old.OK:
		return vDifferent, "the old build fails where the new transcodes: " + old.Err
	case !n2.OK:
		return vDifferent, "the new build's second transcode failed: " + n2.Err
	case n1.SHA != n2.SHA:
		return vDifferent, fmt.Sprintf("NONDETERMINISTIC: the new build wrote %s… (%d B) then %s… (%d B) from the same source",
			n1.SHA[:16], n1.Size, n2.SHA[:16], n2.Size)
	case n1.SHA != old.SHA:
		return vDifferent, fmt.Sprintf("sha256 old %s… (%d B), new %s… (%d B)", old.SHA[:16], old.Size, n1.SHA[:16], n1.Size)
	}
	return vIdentical, ""
}

// giwCell is one (source, configuration) of the bytes mode, with its three results.
type giwCell struct {
	Src          giwSource
	Cfg          giwConfig
	N1, N2, Old  giwResult
	Verdict, Why string
}

// identityGiwBytes is `gate identity -giw bytes`. Exit codes are identityMain's.
func identityGiwBytes(o identityOpts, w io.Writer) int {
	t0 := time.Now()
	progress := o.Progress
	if progress == nil {
		progress = os.Stderr
	}
	refuse := func(format string, a ...any) int {
		fmt.Fprintf(w, "%s== identity (giw bytes) ==%s\n  %sREFUSED: %s%s\n", bold, off, amber, fmt.Sprintf(format, a...), off)
		return 2
	}
	old := &identitySide{Label: "old", Rev: o.OldRev}
	nw := &identitySide{Label: "new", Rev: o.NewRev}
	if why := resolveIdentitySides(o, old, nw); why != "" {
		return refuse("%s", why)
	}
	if o.Timeout <= 0 {
		o.Timeout = 15 * time.Minute
	}
	cfgs := o.Configs
	if len(cfgs) == 0 {
		cfgs = defaultGiwConfigs(o.Assets)
	}
	srcs, err := planGiwSources(o)
	if err != nil {
		return refuse("%v", err)
	}
	var cells []*giwCell
	runnable := 0
	for _, s := range srcs {
		for _, c := range cfgs {
			cells = append(cells, &giwCell{Src: s, Cfg: c})
		}
		if s.NotRun == "" {
			runnable++
		}
	}
	wd, logs, cleanup, err := identityRunDir(o, t0, old, nw, "giw")
	if err != nil {
		return refuse("%v", err)
	}
	defer cleanup()

	var cfgNames []string
	for _, c := range cfgs {
		cfgNames = append(cfgNames, c.String())
	}
	prov := gatherProvenanceIn(o.Repo, [][2]string{
		{"old", fmt.Sprintf("%s %s", old.Short, trunc(old.Title, 70))},
		{"new", fmt.Sprintf("%s %s", nw.Short, trunc(nw.Title, 70))},
		{"mode", "giw bytes: each revision's own cmd/prequant transcodes the source; the files' sha256 are compared"},
		{"assets", giwAssetsDesc(o)},
		{"configs", strings.Join(cfgNames, ", ") + " (quant:target[:e4h]; cpu is this box's arch)"},
		{"env", fmt.Sprintf("GOINFER_* removed from both sides' transcodes (%s) — the code is the only difference", scrubbedDesc(identityScrubbedVars()))},
		{"build", "go build -trimpath ./cmd/prequant in each worktree's root module, GOWORK=off -mod=readonly"},
		{"run dir", wd},
		{"started", t0.Format("15:04:05 MST")},
	})
	prov.Dirty = false
	prov.Fields = append(prov.Fields, [2]string{"tree", "the revs are commits, built in temporary worktrees; the main tree supplies only the fixtures, identically to both"})
	fmt.Fprintf(w, "%s== identity provenance ==%s\n", bold, off)
	prov.write(w)
	if runnable == 0 {
		return refuse("no source on this host — nothing would be compared (a skip is not a pass)")
	}

	fmt.Fprintf(progress, "== identity giw bytes: %d source(s) x %d config(s), started %s — building prequant at %s and %s ==\n",
		runnable, len(cfgs), t0.Format("15:04:05"), old.Short, nw.Short)
	tb := time.Now()
	if failed := buildIdentitySides(o, wd, old, nw); failed != nil {
		fmt.Fprintf(w, "\n  %s%s build (%s) failed%s: %v\n", red, failed.Label, failed.Short, off, failed.Err)
		for _, ln := range tailLines(failed.Log, 20) {
			fmt.Fprintf(w, "    %s\n", ln)
		}
		return refuse("cmd/prequant did not build at %s %s — see above", failed.Label, failed.Short)
	}
	buildWall := time.Since(tb)
	fmt.Fprintf(progress, "   built both in %s\n", buildWall.Round(time.Second))

	tr := time.Now()
	stopBeat := identityHeartbeat(progress, t0)
	done, bySrc := 0, map[string][]*giwCell{}
	for _, c := range cells {
		bySrc[c.Src.Name] = append(bySrc[c.Src.Name], c)
	}
	for i, s := range srcs {
		if s.NotRun != "" {
			for _, c := range bySrc[s.Name] {
				c.Verdict, c.Why = vNotRun, s.NotRun
			}
			continue
		}
		identityBeatState.Store("state", fmt.Sprintf("transcoding %s (%d/%d sources)", s.Name, i+1, len(srcs)))
		st := time.Now()
		for _, c := range bySrc[s.Name] {
			id := sanitize(s.Name + "__" + c.Cfg.String())
			for _, r := range []struct {
				name string
				side *identitySide
				dst  *giwResult
			}{{"new1", nw, &c.N1}, {"old", old, &c.Old}, {"new2", nw, &c.N2}} {
				dir := filepath.Join(wd, "giw", r.name, id)
				res, _ := runGiwTranscode(o, r.side.Prequant, s.Path, dir, c.Cfg, filepath.Join(logs, r.name+"-"+id+".log"))
				*r.dst = res
				if !o.Keep {
					_ = os.RemoveAll(dir) // hashed already: a real sidecar is gigabytes, and three of them need not coexist
				}
			}
			c.Verdict, c.Why = judgeGiwBytes(c.N1, c.N2, c.Old)
		}
		done++
		counts := map[string]int{}
		for _, c := range bySrc[s.Name] {
			counts[c.Verdict]++
		}
		fmt.Fprintf(progress, "   [%d/%d] %-34s %d identical, %d different, %d not run — %s, elapsed %s (started %s)\n", done, runnable, s.Name,
			counts[vIdentical], counts[vDifferent], counts[vNotRun], time.Since(st).Round(100*time.Millisecond), time.Since(t0).Round(time.Second), t0.Format("15:04:05"))
	}
	stopBeat()
	return reportGiwBytes(w, o, old, nw, srcs, cells, t0, buildWall, time.Since(tr), logs)
}

func giwAssetsDesc(o identityOpts) string {
	if o.Assets == "real" {
		return "real — the -assets real table's GGUFs and directories under " + o.ModelsDir
	}
	return "tiny — every directory with a .safetensors file under " + filepath.Join(o.FixtureRoot, "testdata") + " and decoder/testdata"
}

func reportGiwBytes(w io.Writer, o identityOpts, old, nw *identitySide, srcs []giwSource, cells []*giwCell, t0 time.Time, buildWall, runWall time.Duration, logs string) int {
	fmt.Fprintf(w, "\n%s== giw bytes: old %s vs new %s, per source ==%s\n", bold, old.Short, nw.Short, off)
	bySrc := map[string][]*giwCell{}
	for _, c := range cells {
		bySrc[c.Src.Name] = append(bySrc[c.Src.Name], c)
	}
	nI, nD, nNR := 0, 0, 0
	for _, s := range srcs {
		var ident, diff, nr []string
		for _, c := range bySrc[s.Name] {
			switch c.Verdict {
			case vIdentical:
				ident = append(ident, c.Cfg.String())
			case vDifferent:
				diff = append(diff, c.Cfg.String())
			default:
				nr = append(nr, c.Cfg.String())
			}
		}
		nI, nD, nNR = nI+len(ident), nD+len(diff), nNR+len(nr)
		switch {
		case len(diff) > 0:
			fmt.Fprintf(w, "  %s%-10s%s %-34s %d of %d cell(s) differ\n", red, vDifferent, off, s.Name, len(diff), len(bySrc[s.Name]))
			for _, c := range bySrc[s.Name] {
				if c.Verdict == vDifferent {
					fmt.Fprintf(w, "      %-18s %s\n", c.Cfg, c.Why)
				}
			}
		case len(ident) > 0:
			note := ""
			if len(nr) > 0 {
				note = fmt.Sprintf(" (%d not run: %s)", len(nr), strings.Join(nr, " "))
			}
			fmt.Fprintf(w, "  %s%-10s%s %-34s %d cell(s): %s%s\n", green, vIdentical, off, s.Name, len(ident), strings.Join(ident, " "), note)
		default:
			fmt.Fprintf(w, "  %s%-10s%s %-34s nothing ran — see NOT RUN\n", amber, vNotRun, off, s.Name)
		}
	}
	if nNR > 0 {
		fmt.Fprintf(w, "\n%s== NOT RUN (%d cell(s)) — reported, never counted as a pass ==%s\n", bold, nNR, off)
		type key struct{ src, why string }
		seen := map[key][]string{}
		var order []key
		for _, c := range cells {
			if c.Verdict != vNotRun {
				continue
			}
			k := key{c.Src.Name, c.Why}
			if _, ok := seen[k]; !ok {
				order = append(order, k)
			}
			seen[k] = append(seen[k], c.Cfg.String())
		}
		for _, k := range order {
			fmt.Fprintf(w, "  %s%-8s%s %-34s [%s] %s\n", amber, vNotRun, off, k.src, strings.Join(seen[k], " "), trunc(k.why, 170))
		}
	}
	end := time.Now()
	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  cells    %d: %d IDENTICAL, %d DIFFERENT, %d NOT RUN (a skip is not a pass)\n", len(cells), nI, nD, nNR)
	fmt.Fprintf(w, "  wall     %s (%s → %s): builds %s, transcodes %s (each cell: new, old, new again)\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"),
		end.Format("15:04:05"), buildWall.Round(time.Second), runWall.Round(time.Second))
	keep := "removed after hashing (pass -keep to retain them)"
	if o.Keep {
		keep = "kept"
	}
	fmt.Fprintf(w, "  logs     %s (bundles %s; worktrees removed)\n", logs, keep)
	if o.Record != "" {
		if err := writeGiwRecord(o.Record, old, nw, cells); err != nil {
			fmt.Fprintf(w, "  %swrite %s: %v%s\n", red, o.Record, err, off)
		} else {
			fmt.Fprintf(w, "  record   %s (one JSON line per cell: verdict, sha256, bytes, prequant's phase line)\n", o.Record)
		}
	}
	switch {
	case nD > 0:
		fmt.Fprintf(w, "  %sRED — %d cell(s) DIFFERENT%s\n", red, nD, off)
		return 1
	case nI == 0:
		fmt.Fprintf(w, "  %sREFUSED — no cell ran%s\n", amber, off)
		return 2
	}
	fmt.Fprintf(w, "  %sGREEN — every cell that ran is IDENTICAL%s\n", green, off)
	return 0
}

// writeGiwRecord writes one JSON line per cell for a measurement record.
func writeGiwRecord(path string, old, nw *identitySide, cells []*giwCell) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, c := range cells {
		row := map[string]any{"source": c.Src.Name, "config": c.Cfg.String(), "verdict": c.Verdict, "old": old.Short, "new": nw.Short}
		if c.Why != "" {
			row["why"] = c.Why
		}
		if c.Old.OK {
			row["old_sha256"], row["old_bytes"], row["old_line"] = c.Old.SHA, c.Old.Size, c.Old.Line
		}
		if c.N1.OK {
			row["new_sha256"], row["new_bytes"], row["new_line"] = c.N1.SHA, c.N1.Size, c.N1.Line
		}
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

// giwTranscodeCells is the logits mode's extra step before a dumper runs: this side transcodes each cell's source at
// the cell's quant for the backend's target, and the dumper is pointed at the result. A cell this side cannot transcode
// gets a failed result in the dump directory, which judgeCell reads as the dumper's own failure would be read. It
// returns each transcoded cell's .giw and the directories to remove afterwards.
func giwTranscodeCells(o identityOpts, wd string, r identityRun, family string, cells []*identityCell) (paths map[string]string, dirs []string) {
	paths = map[string]string{}
	dumpDir := filepath.Join(wd, "dumps", r.Name, family)
	_ = os.MkdirAll(dumpDir, 0o755)
	cfg := func(c *identityCell) giwConfig {
		return giwConfig{Quant: c.Quant, Target: giwTargetForBackend(o.Backend)}
	}
	for _, c := range cells {
		dir := filepath.Join(wd, "giw", r.Name, c.ID)
		res, out := runGiwTranscode(o, r.Side.Prequant, c.Path, dir, cfg(c), filepath.Join(wd, "logs", r.Name+"-giw-"+c.ID+".log"))
		dirs = append(dirs, dir)
		if !res.OK {
			b, _ := json.Marshal(dumpMeta{ID: c.ID, Err: "transcode (" + cfg(c).String() + "): " + res.Err})
			_ = os.WriteFile(filepath.Join(dumpDir, c.ID+".json"), b, 0o644)
			continue
		}
		paths[c.ID] = out
	}
	return paths, dirs
}
