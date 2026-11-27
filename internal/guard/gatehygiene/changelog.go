package gatehygiene

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

const changelogWaiverFile = "docs/changelog-waivers.txt"

type clSource interface {
	read(path string) ([]byte, error)
	list(dir string) ([]string, error)
}

type gitSource struct{ root, rev string }

func (g gitSource) read(path string) ([]byte, error) {
	b, err := g.showOnce(path)
	if err == nil {
		return b, nil
	}
	old, moved := gatecore.PreMovePath(path)
	if !moved || !gatecore.AbsentFromRev(err) {
		return nil, err
	}
	return g.showOnce(old)
}

func (g gitSource) showOnce(path string) ([]byte, error) {
	cmd := exec.Command("git", "show", g.rev+":"+path)
	cmd.Dir = g.root
	var stderr strings.Builder
	cmd.Stderr = &stderr
	b, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git show %s:%s: %v: %s", g.rev, path, err, strings.TrimSpace(stderr.String()))
	}
	return b, nil
}

func (g gitSource) list(dir string) ([]string, error) {
	names, err := g.lsTree(dir)
	if err == nil && len(names) > 0 {
		return names, nil
	}
	old, moved := gatecore.PreMovePath(dir)
	if !moved {
		return names, err
	}
	return g.lsTree(old)
}

func (g gitSource) lsTree(dir string) ([]string, error) {
	cmd := exec.Command("git", "ls-tree", "--name-only", g.rev, dir+"/")
	cmd.Dir = g.root
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var names []string
	for ln := range strings.FieldsSeq(string(out)) {
		names = append(names, filepath.Base(ln))
	}
	return names, nil
}

type dirSource struct{ root string }

func (d dirSource) read(path string) ([]byte, error) { return os.ReadFile(filepath.Join(d.root, path)) }

func (d dirSource) list(dir string) ([]string, error) {
	ents, err := os.ReadDir(filepath.Join(d.root, dir))
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range ents {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	return names, nil
}

var serveAppDirs = []string{"internal/app/serveapp", "internal/serveapp"}

func serveRoutes(src clSource) (map[string]bool, error) {
	dir := serveAppDirs[0]
	names, err := src.list(dir)
	if err != nil || len(names) == 0 {
		dir = serveAppDirs[1]
		if names, err = src.list(dir); err != nil {
			return nil, err
		}
	}
	routes := map[string]bool{}
	fset := token.NewFileSet()
	for _, n := range names {
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		b, err := src.read(dir + "/" + n)
		if err != nil {
			return nil, err
		}
		f, err := parser.ParseFile(fset, n, b, 0)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", n, err)
		}
		ast.Inspect(f, func(nd ast.Node) bool {
			call, ok := nd.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || (sel.Sel.Name != "HandleFunc" && sel.Sel.Name != "Handle") {
				return true
			}
			if lit, ok := call.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if s, err := strconv.Unquote(lit.Value); err == nil && strings.Contains(s, "/") {
					routes[NormRoute(s)] = true
				}
			}
			return true
		})
	}
	return routes, nil
}

func readClState(src clSource) (*ClState, error) {
	st := &ClState{Flags: map[string]bool{}, Families: map[string][]string{}}
	var err error
	if st.Routes, err = serveRoutes(src); err != nil {
		return nil, err
	}
	b, err := src.read("docs/flags.meta.json")
	if err != nil {
		return nil, err
	}
	var fm struct {
		Flags []struct {
			Name string `json:"name"`
		} `json:"flags"`
	}
	if err := json.Unmarshal(b, &fm); err != nil {
		return nil, fmt.Errorf("docs/flags.meta.json: %w", err)
	}
	for _, f := range fm.Flags {
		st.Flags[f.Name] = true
	}
	if b, err = src.read("docs/capability-matrix.json"); err != nil {
		return nil, err
	}
	var fams []struct {
		Name       string   `json:"name"`
		Display    string   `json:"display_name"`
		ModelTypes []string `json:"model_types"`
	}
	if err := json.Unmarshal(b, &fams); err != nil {
		return nil, fmt.Errorf("docs/capability-matrix.json: %w", err)
	}
	for _, f := range fams {
		st.Families[f.Name] = append([]string{f.Display}, f.ModelTypes...)
	}
	if b, err = src.read("go.mod"); err != nil {
		return nil, err
	}
	if m := regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit (v[^\s]+)`).FindSubmatch(b); m != nil {
		st.Pin = string(m[1])
	}
	return st, nil
}

func changelogFragments(src clSource) (string, error) {
	names, err := src.list(ChangelogFragmentDir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	slices.Sort(names)
	slices.Reverse(names)
	var b strings.Builder
	for _, n := range names {
		if n == "README.md" || !strings.HasSuffix(n, ".md") {
			continue
		}
		t, err := src.read(ChangelogFragmentDir + "/" + n)
		if err != nil {
			return "", err
		}
		b.Write(t)
	}
	return b.String(), nil
}

func loadChangelogWaivers(path string) (map[string]string, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	w := map[string]string{}
	for i, ln := range strings.Split(string(b), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		f := strings.Fields(ln)
		nameWords := 1
		if len(f) > 0 && f[0] == "route" {
			nameWords = 2
		}
		if len(f) < 1+nameWords+1 {
			return nil, fmt.Errorf("%s:%d: a waiver is `<kind> <name> <reason>` (a route's name is its method and path); the reason is mandatory", filepath.Base(path), i+1)
		}
		w[f[0]+" "+strings.Join(f[1:1+nameWords], " ")] = strings.Join(f[1+nameWords:], " ")
	}
	return w, nil
}

// NewestVersionTag returns the newest root-module v* tag reachable from HEAD other than skip.
func NewestVersionTag(root, skip string) (string, error) {
	out, err := gatecore.GitOut(root, "tag", "--merged", "HEAD", "--list", "v[0-9]*", "--sort=-v:refname")
	if err != nil {
		return "", err
	}
	for t := range strings.FieldsSeq(out) {
		if t != skip && !strings.Contains(t, "/") {
			return t, nil
		}
	}
	return "", fmt.Errorf("no v* tag reachable from HEAD")
}

// RunChangelog implements `gate changelog`.
func RunChangelog(args []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate changelog", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	base := fs.String("base", "", "the revision to compare with (default: the newest v* tag reachable from HEAD that is not -section)")
	section := fs.String("section", "Unreleased", "the CHANGELOG section that must name the changes: Unreleased, or a version such as v0.23.0")
	headRev := fs.String("head", "", "compare this revision instead of the working tree")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	root, err := gatecore.RepoRoot()
	if err != nil {
		fmt.Fprintln(w, err)
		return 2
	}
	if *base == "" {
		if *base, err = NewestVersionTag(root, *section); err != nil {
			fmt.Fprintf(w, "gate changelog: %v\n", err)
			return 2
		}
	}
	var headSrc clSource = dirSource{root}
	if *headRev != "" {
		headSrc = gitSource{root, *headRev}
	}
	doc, err := headSrc.read("CHANGELOG.md")
	if err != nil {
		fmt.Fprintf(w, "gate changelog: %v\n", err)
		return 2
	}
	sec, ok := ClSection(string(doc), *section)
	if !ok {
		fmt.Fprintf(w, "gate changelog: CHANGELOG.md has no `## [%s]` section\n", *section)
		return 2
	}
	var stray []string
	if *section == "Unreleased" {
		stray = unreleasedStray(sec)
		frags, err := changelogFragments(headSrc)
		if err != nil {
			fmt.Fprintf(w, "gate changelog: %v\n", err)
			return 2
		}
		sec += frags
	}
	waivers, err := loadChangelogWaivers(filepath.Join(root, changelogWaiverFile))
	if err != nil {
		fmt.Fprintf(w, "gate changelog: %v\n", err)
		return 2
	}
	bs, err := readClState(gitSource{root, *base})
	if err != nil {
		fmt.Fprintf(w, "gate changelog: reading %s: %v\n", *base, err)
		return 2
	}
	hs, err := readClState(headSrc)
	if err != nil {
		fmt.Fprintf(w, "gate changelog: reading the head tree: %v\n", err)
		return 2
	}
	if len(hs.Routes) < 10 || len(bs.Routes) < 10 {
		fmt.Fprintf(w, "gate changelog: the route scan found %d routes at the working tree and %d at %s; it is not reading serve's registrations\n", len(hs.Routes), len(bs.Routes), *base)
		return 2
	}
	used := map[string]bool{}
	waived := func(kind, name string) bool {
		if _, ok := waivers[kind+" "+name]; ok {
			used[kind+" "+name] = true
			return true
		}
		return false
	}
	owed := ChangelogOwed(bs, hs, sec, waived)
	if len(stray) > 0 {
		owed = append(owed, OwedItem{
			Kind: "entry", Name: strings.TrimSpace(stray[0]),
			Why: fmt.Sprintf("%d line(s) under CHANGELOG.md's [Unreleased], which holds only the pointer; move the entry into %s/", len(stray), ChangelogFragmentDir),
		})
	}
	fmt.Fprintf(w, "CHANGELOG [%s] against %s: %d routes, %d flags, %d families, aikit %s (was %s)\n", *section, *base, len(hs.Routes), len(hs.Flags), len(hs.Families), hs.Pin, bs.Pin)
	for _, k := range slices.Sorted(maps.Keys(used)) {
		fmt.Fprintf(w, "  waived %s: %s\n", k, waivers[k])
	}
	if len(owed) == 0 {
		fmt.Fprintln(w, "  every new route, flag, family and the aikit pin is named")
		return 0
	}
	for _, o := range owed {
		fmt.Fprintf(w, "  OWED %-6s %s (%s)\n", o.Kind, o.Name, o.Why)
	}
	return 1
}
