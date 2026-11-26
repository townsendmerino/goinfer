package main

import (
	"fmt"
	"maps"
	"regexp"
	"runtime"
	"slices"
	"strings"
)

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
	// Mem: the cell loads a real checkpoint, so it runs with nothing beside it (runJobs). Its tests
	// would otherwise be refused by the fit guard, or pass only while the rest of the run happens to
	// leave enough memory free.
	Mem    bool
	NotRun string // non-empty: cannot run on this host, and why
	// Checkpoint: the cell is a package's real-checkpoint loaders, split out of it.
	Checkpoint bool
	// Deferred, when non-empty, is why the day loop leaves the cell to the pre-tag tier (deferCells). It is
	// reported and counted like NotRun, and -full runs it.
	Deferred string
}

// minSplitRest is how many non-walking tests a package needs before its walkers get a cell of their own.
// The split exists for decoder (hundreds of tests, and census tests that read the whole tree); for a
// ten-test package the second process costs more than it saves.
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
		loads := map[string]bool{}
		for _, l := range r.loaders {
			loads[l] = true
		}
		anyLoads := func(names []string) bool {
			for _, n := range names {
				if loads[n] {
					return true
				}
			}
			return false
		}
		if golden {
			gc := cells[len(cells)-1]
			for _, l := range r.loaders {
				gc.Mem = gc.Mem || goldenRE.MatchString(l)
			}
			gc.Mem = gc.Mem || heavyGoldens
		}
		var walkers []string
		for _, w := range r.walkers {
			if !golden || !goldenRE.MatchString(w) {
				walkers = append(walkers, w)
			}
		}
		// A package small enough to re-run whole is not worth a second process.
		split := map[string]bool{}
		if len(walkers) > 0 && len(r.tests)-len(walkers) >= minSplitRest {
			cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg) + " [walkers]", Root: r, Run: anchored(walkers),
				Count1: force, Tier: tier, Why: why, Mem: anyLoads(walkers), NotRun: notRun})
			skips = append(skips, anchored(walkers))
			for _, w := range walkers {
				split[w] = true
			}
		}
		// The checkpoint loaders get a cell of their own, so the run-alone rule holds only them and
		// not the rest of a five-minute package.
		var loaders []string
		for _, l := range r.loaders {
			if !split[l] && !(golden && goldenRE.MatchString(l)) {
				loaders = append(loaders, l)
			}
		}
		restMem := false
		if len(loaders) > 0 && len(r.tests)-len(split)-len(loaders) >= minSplitRest {
			shared := loaders[:0:0]
			for _, l := range loaders {
				if !slicesContains(r.Mod.CheckpointAlone, l) {
					shared = append(shared, l)
					continue
				}
				cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg) + " [checkpoint " + l + "]", Root: r, Run: anchored([]string{l}),
					Count1: force, Tier: tier, Why: why, Mem: true, NotRun: notRun, Checkpoint: true})
				skips = append(skips, anchored([]string{l}))
			}
			if len(shared) > 0 {
				cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg) + " [checkpoint]", Root: r, Run: anchored(shared),
					Count1: force, Tier: tier, Why: why, Mem: true, NotRun: notRun, Checkpoint: true})
				skips = append(skips, anchored(shared))
			}
		} else {
			restMem = len(loaders) > 0
		}
		cells = append(cells, &quickCell{Name: shortPkg(qc, r.Pkg), Root: r, Skip: strings.Join(skips, "|"),
			Count1: force, Tier: tier, Why: why, Mem: restMem, NotRun: notRun})
	}
	return cells
}

// The reasons deferCells gives. Each names where the cell still runs.
const (
	deferCheckpoint = "loads a real checkpoint: pre-tag tier (gate quick -full)"
	deferSlow       = "long kernel tests of a device module the diff does not touch: pre-tag tier (gate quick -full)"
)

// deferCells is the day loop's cut of what buildCells selected: it marks the cells left to the pre-tag tier and
// returns the list with them marked. Two rules, and nothing else is ever deferred:
//
//  1. A [checkpoint] cell: the tests that load a real checkpoint because one is on this machine. CI has no
//     checkpoints and never runs them.
//  2. In a device module the diff reaches only from outside (no changed file under its directory), the tests its
//     Slow pattern names, split into a [slow] cell. The cell they came from skips exactly what [slow] runs, so the
//     two are complementary and -full, which runs both, loses nothing.
//
// A cache-checked cell is left whole: the test cache decides it, and replays it when nothing it reads has moved.
func deferCells(qc *quickConfig, s *quickSelection, cells []*quickCell) []*quickCell {
	touched := map[*quickModule]bool{}
	for _, rel := range s.Changed {
		touched[qc.moduleOf(rel)] = true
	}
	var out []*quickCell
	for _, c := range cells {
		m := c.Root.Mod
		switch {
		case c.NotRun != "":
		case c.Checkpoint:
			c.Deferred = deferCheckpoint
		case m.Device && m.Slow != "" && !touched[m] && c.Tier != "cache-checked" && c.Run == "" && !c.Golden:
			re := regexp.MustCompile(m.Slow)
			var slow []string
			for _, t := range c.Root.tests {
				if re.MatchString(t) && !c.skips(t) {
					slow = append(slow, t)
				}
			}
			if len(slow) > 0 {
				sc := *c
				sc.Name, sc.Run, sc.Skip, sc.Deferred = c.Name+" [slow]", anchored(slow), "", deferSlow
				out = append(out, &sc)
				if c.Skip != "" {
					c.Skip += "|"
				}
				c.Skip += anchored(slow)
			}
		}
		out = append(out, c)
	}
	return out
}

// skips reports whether the cell's -skip pattern already excludes a top-level test.
func (c *quickCell) skips(test string) bool {
	return c.Skip != "" && regexp.MustCompile(c.Skip).MatchString(test)
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
	return slices.Sorted(maps.Keys(set))
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
