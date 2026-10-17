package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// heavyFamily is one row of the table the CUDA heavy tier groups its processes by. Each test goes to the group of the
// highest-ranked family whose pattern matches a string literal the test can reach, and a test that reaches none goes to
// heavySmallGroup. A Large family is one whose checkpoint alone is a large share of the process's memory: a test may reach
// at most one, so no process holds two. Basis: docs/code-notes/cmd-gate.md, "cudaHeavy: one model family per process".
type heavyFamily struct {
	Group string
	Large bool
	Rank  int
	re    *regexp.Regexp
}

// heavySmallGroup is the shared group: tiny fixtures and checkpoints of about 1.5B parameters or less.
const heavySmallGroup = "small"

// heavyFamilies is matched in order, and the first row whose pattern matches a literal classifies it, so a narrower
// pattern sits above the broader one it would otherwise lose to. A literal that names a checkpoint and matches no row is
// a loud failure (heavyPartition), so a new model gets a row here before its test can run in the gate.
var heavyFamilies = []heavyFamily{
	{Group: heavySmallGroup, Rank: 0, re: regexp.MustCompile(`(^|/)testdata/`)}, // committed or locally dropped fixtures
	{Group: heavySmallGroup, Rank: 0, re: regexp.MustCompile(`(?i)qwen3\.6-35b-a3b-vision`)},
	{Group: "gemma4-26b", Large: true, Rank: 100, re: regexp.MustCompile(`(?i)gemma-?4-26b`)},
	{Group: "qwen3.6-35b", Large: true, Rank: 99, re: regexp.MustCompile(`(?i)qwen3?\.?6-35b`)},
	{Group: "gpt-oss-20b", Large: true, Rank: 98, re: regexp.MustCompile(`(?i)gpt-oss-20b`)},
	{Group: "7b-8b", Large: true, Rank: 90, re: regexp.MustCompile(`(?i)qwen2\.5-7b|command-r7b|aya-expanse-8b`)},
	{Group: "nemotron-h", Large: true, Rank: 85, re: regexp.MustCompile(`(?i)nemotron-h-4b|nemotron-nano-9b|nvidia_NVIDIA-Nemotron`)},
	{Group: "gemma4-e4b", Large: true, Rank: 80, re: regexp.MustCompile(`(?i)gemma-4-e4b`)},
	{Group: "gemma3-4b", Rank: 50, re: regexp.MustCompile(`(?i)gemma-3-4b`)},
	{Group: "qwen3-4b", Rank: 49, re: regexp.MustCompile(`(?i)qwen3-4b`)},
	{Group: "qwen3-vl-2b", Rank: 48, re: regexp.MustCompile(`(?i)qwen3-vl-2b`)},
	{Group: "qwen25vl-3b", Rank: 47, re: regexp.MustCompile(`(?i)qwen25vl-3b|qwen2\.5-vl-3b`)},
	{Group: "glm-ocr", Rank: 46, re: regexp.MustCompile(`(?i)glm-ocr`)},
	{Group: "gemma2-2b", Rank: 44, re: regexp.MustCompile(`(?i)gemma-2-2b`)},
	{Group: "gemma4-e2b", Rank: 45, re: regexp.MustCompile(`(?i)gemma-4-e2b`)},
	{Group: heavySmallGroup, Rank: 1, re: regexp.MustCompile(
		`(?i)qwen2\.5-coder-(0\.5b|1\.5b)|qwen2\.5-0\.5b|gemma3-1b|llama-3\.2-1b|qwen3-1\.7b|qwen3\.5-0\.8b|qwen35vl|phi-?3-mini|clef-flash`)},
}

// modelishRe matches a literal that names a checkpoint by its shape: a path under a models directory, or a GGUF or .giw
// file name. It is applied only to literals without whitespace, which a message or a log line has and a path does not.
var modelishRe = regexp.MustCompile(`(^|/|\$)(models|MODELS)/[^/]|[A-Za-z0-9_-]\.(gguf|giw)$`)

// classifyLiteral returns the heavyFamilies row a literal matches, or nil.
func classifyLiteral(s string) *heavyFamily {
	for i := range heavyFamilies {
		if heavyFamilies[i].re.MatchString(s) {
			return &heavyFamilies[i]
		}
	}
	return nil
}

// heavyGroup is one process of the heavy tier's main group: the tests it owns and the -run pattern that selects exactly
// those.
type heavyGroup struct {
	Label string
	Tests []string
	Run   string
}

// heavyPartition assigns each listed test to one group. refs maps every top-level test found in source to the string
// literals it can reach (testReachableLiterals); modelEntries are the names in the models directory, so a literal naming
// one of them that no row classifies is caught even when its shape is not path-like. problems lists each test that fits
// no group, and the gate fails on any: a listed test missing from refs, a literal that names a checkpoint no row
// classifies, or a test that reaches two Large families.
func heavyPartition(listed []string, refs map[string][]string, modelEntries []string) (groups []heavyGroup, problems []string) {
	entries := map[string]bool{}
	for _, e := range modelEntries {
		entries[e] = true
	}
	byGroup := map[string][]string{}
	for _, t := range listed {
		lits, ok := refs[t]
		if !ok {
			problems = append(problems, fmt.Sprintf("%s: listed by `go test -list` but not found in cuda/*_test.go under the cuda and goinfer_testhooks tags", t))
			continue
		}
		var best *heavyFamily
		large := map[string]bool{}
		var unknown []string
		for _, l := range lits {
			f := classifyLiteral(l)
			if f == nil {
				if !strings.ContainsAny(l, " \t\n") && (modelishRe.MatchString(l) || namesEntry(l, entries)) {
					unknown = append(unknown, l)
				}
				continue
			}
			if f.Large {
				large[f.Group] = true
			}
			if best == nil || f.Rank > best.Rank {
				best = f
			}
		}
		if len(unknown) > 0 {
			problems = append(problems, fmt.Sprintf("%s: names a checkpoint no heavyFamilies row classifies: %s", t, strings.Join(unknown, ", ")))
			continue
		}
		if len(large) > 1 {
			problems = append(problems, fmt.Sprintf("%s: reaches %d large families (%s); a process may hold only one", t, len(large), strings.Join(sortedSet(large), ", ")))
			continue
		}
		g := heavySmallGroup
		if best != nil {
			g = best.Group
		}
		byGroup[g] = append(byGroup[g], t)
	}
	// Groups in table order, so the largest families run first and the shared group last.
	seen := map[string]bool{}
	var order []string
	for _, f := range heavyFamilies {
		if f.Group != heavySmallGroup && !seen[f.Group] {
			seen[f.Group] = true
			order = append(order, f.Group)
		}
	}
	order = append(order, heavySmallGroup)
	for _, label := range order {
		tests := byGroup[label]
		if len(tests) == 0 {
			continue
		}
		sort.Strings(tests)
		quoted := make([]string, len(tests))
		for i, n := range tests {
			quoted[i] = regexp.QuoteMeta(n)
		}
		groups = append(groups, heavyGroup{Label: label, Tests: tests, Run: "^(" + strings.Join(quoted, "|") + ")$"})
	}
	return groups, problems
}

// namesEntry reports whether any path segment of l is an entry of the models directory.
func namesEntry(l string, entries map[string]bool) bool {
	for seg := range strings.SplitSeq(l, "/") {
		if seg != "" && entries[seg] {
			return true
		}
	}
	return false
}

// testReachableLiterals parses dir's *_test.go files that build under tags on linux/amd64 and returns, for each top-level
// Test function, the string literals in its body and in every package-level declaration it reaches by name, transitively.
// A literal that is a key of the asset registry (assetsJSON, testdata/assets.json) also contributes that asset's candidate
// paths. Reaching by name over-approximates (a method is matched by name alone), which can only move a test to a larger
// family's group, never out of one.
func testReachableLiterals(dir string, tags []string, assetsJSON string) (map[string][]string, error) {
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH, ctx.BuildTags, ctx.CgoEnabled = "linux", "amd64", tags, false
	files, err := filepath.Glob(filepath.Join(dir, "*_test.go"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no *_test.go files in %s", dir)
	}
	sort.Strings(files)
	fset := token.NewFileSet()
	lits := map[string]map[string]bool{} // declaration name -> literals in it
	uses := map[string]map[string]bool{} // declaration name -> identifiers it mentions
	var tests []string
	add := func(name string, node ast.Node) {
		if lits[name] == nil {
			lits[name], uses[name] = map[string]bool{}, map[string]bool{}
		}
		ast.Inspect(node, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BasicLit:
				if n.Kind == token.STRING {
					if s, err := strconv.Unquote(n.Value); err == nil {
						lits[name][s] = true
					}
				}
			case *ast.Ident:
				uses[name][n.Name] = true
			case *ast.SelectorExpr:
				uses[name]["method:"+n.Sel.Name] = true
			}
			return true
		})
	}
	for _, f := range files {
		if ok, err := ctx.MatchFile(dir, filepath.Base(f)); err != nil || !ok {
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		for _, d := range af.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				name := d.Name.Name
				if d.Recv != nil {
					name = "method:" + name
				} else if strings.HasPrefix(name, "Test") {
					tests = append(tests, name)
				}
				add(name, d)
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.ValueSpec:
						for _, id := range s.Names {
							add(id.Name, s)
						}
					case *ast.TypeSpec:
						add(s.Name.Name, s)
					}
				}
			}
		}
	}
	assets, err := assetCandidates(assetsJSON)
	if err != nil {
		return nil, err
	}
	out := map[string][]string{}
	for _, t := range tests {
		seen := map[string]bool{t: true}
		todo := []string{t}
		got := map[string]bool{}
		for len(todo) > 0 {
			n := todo[len(todo)-1]
			todo = todo[:len(todo)-1]
			for l := range lits[n] {
				got[l] = true
				for _, c := range assets[l] {
					got[c] = true
				}
			}
			for u := range uses[n] {
				if _, isDecl := lits[u]; isDecl && !seen[u] {
					seen[u] = true
					todo = append(todo, u)
				}
			}
		}
		out[t] = sortedSet(got)
	}
	return out, nil
}

// assetCandidates reads the asset registry into env -> candidate paths. A missing file is an empty registry.
func assetCandidates(path string) (map[string][]string, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return map[string][]string{}, nil
	}
	if err != nil {
		return nil, err
	}
	var reg struct {
		Assets []struct {
			Env        string   `json:"env"`
			Candidates []string `json:"candidates"`
		} `json:"assets"`
	}
	if err := json.Unmarshal(b, &reg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	out := map[string][]string{}
	for _, a := range reg.Assets {
		out[a.Env] = a.Candidates
	}
	return out, nil
}

// modelsDirEntries lists the models directory's entry names, or nil when it cannot be read.
func modelsDirEntries(dir string) []string {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range ents {
		out = append(out, e.Name())
	}
	return out
}

// runCUDAHeavy is `gate cuda-heavy`: the heavy tier's main-group partition (-list), or chosen groups run alone with their
// test binary's memory sampled (-group). It runs no drain group or fresh-process test and writes no record; `gate gpu` is
// the gate. Exit 0 when every chosen group passed and every listed test fits a group, 1 otherwise, 2 on bad usage.
func runCUDAHeavy(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate cuda-heavy", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	list := fs.Bool("list", false, "print the partition, each group's tests included, and run nothing")
	only := fs.String("group", "", "comma-separated group labels to run (default: every group)")
	logDir := fs.String("logdir", os.TempDir(), "directory for raw go test -json logs")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(w, "gate cuda-heavy: %v\n", err)
		return 2
	}
	if err := os.Chdir(root); err != nil {
		fmt.Fprintf(w, "gate cuda-heavy: %v\n", err)
		return 2
	}
	// As runGPU does: an ambient drain-group variable would run the draining tests inside a main-group process.
	for _, v := range []string{"GOINFER_DRAIN_GROUP"} {
		if val := os.Getenv(v); val != "" {
			fmt.Fprintf(w, "note: %s=%s was set in the calling environment — UNSET (the main group sets its own).\n", v, val)
			os.Unsetenv(v)
		}
	}
	g := &gpuGate{w: w, logDir: *logDir, emitted: map[string]bool{}, models: env("GOINFER_GATE_MODELS", filepath.Join(home(), "models"))}
	listed, err := listCUDATests()
	if err != nil || len(listed) == 0 {
		fmt.Fprintf(w, "gate cuda-heavy: `go test -list` found no tests in ./cuda/ (%v)\n", err)
		return 1
	}
	groups, problems, err := partitionHeavy(listed, g.models)
	if err != nil {
		fmt.Fprintf(w, "gate cuda-heavy: %v\n", err)
		return 1
	}
	fmt.Fprintf(w, "%d test(s) in ./cuda/, %d main-group process(es):\n", len(listed), len(groups))
	for _, h := range groups {
		fmt.Fprintf(w, "  %-14s %d test(s)\n", h.Label, len(h.Tests))
		if *list {
			for _, t := range h.Tests {
				fmt.Fprintf(w, "      %s\n", t)
			}
		}
	}
	for _, p := range problems {
		fmt.Fprintf(w, "FITS NO GROUP: %s\n", p)
	}
	rc := 0
	if len(problems) > 0 {
		rc = 1
	}
	if *list {
		return rc
	}
	want := map[string]bool{}
	for l := range strings.SplitSeq(*only, ",") {
		if l = strings.TrimSpace(l); l != "" {
			want[l] = true
		}
	}
	for l := range want {
		if !slices.ContainsFunc(groups, func(h heavyGroup) bool { return h.Label == l }) {
			fmt.Fprintf(w, "gate cuda-heavy: no group %q\n", l)
			return 2
		}
	}
	for _, h := range groups {
		if len(want) > 0 && !want[h.Label] {
			continue
		}
		fmt.Fprintf(w, "\n== %s (%d test(s)) ==\n", h.Label, len(h.Tests))
		_, cr, out, peak := g.runHeavyGroup(h)
		g.reportPeak(h, cr, peak)
		fmt.Fprintf(w, "      pass %d  skip %d  fail %d  rc %d  log %s\n", cr.Pass, cr.Skip, cr.Fail, cr.RC, cr.LogPath)
		if cr.Fail > 0 || cr.RC != 0 {
			rc = 1
			g.detail(out, failTestRe)
		}
	}
	return rc
}
