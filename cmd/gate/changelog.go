package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// `gate changelog`: the release notes say what changed. The docs already have checks that fail when a table disagrees with the code
// (the flags page, the support table, the env-var list); the CHANGELOG had none. Four kinds of change are user-visible, enumerable
// from the tree, and have been missed: an HTTP route, a serve flag, a model family, and the aikit pin. This compares the tree at a
// base tag with the working tree and lists each one the CHANGELOG section does not name.
//
// The rule is a whole-string match, deliberately weak: it catches the omission, which is what happens, and not a careless sentence,
// which no tool catches. A route is covered by its own path (any `{name}` placeholder reads as `{}`) or by a NEW route that is a
// path prefix of it at a segment boundary (a new family's sub-routes ride with the family). Something that is not release-note
// material is waived in docs/changelog-waivers.txt as `<kind> <name> <reason>`, the reason mandatory and printed each time.
//
// The Unreleased section is read with the fragments in changelog.d/, which hold its entries.
//
// Usage: gate changelog [-base vX.Y.Z] [-section Unreleased|vX.Y.Z]   exit 0 clean, 1 something owed, 2 refused.

const changelogWaiverFile = "docs/changelog-waivers.txt"

// clSource reads files at one revision of the repository, or from the working tree.
type clSource interface {
	read(path string) ([]byte, error)
	list(dir string) ([]string, error) // file names directly under dir
}

type gitSource struct{ root, rev string }

func (g gitSource) read(path string) ([]byte, error) {
	cmd := exec.Command("git", "show", g.rev+":"+path)
	cmd.Dir = g.root
	return cmd.Output()
}

func (g gitSource) list(dir string) ([]string, error) {
	cmd := exec.Command("git", "ls-tree", "--name-only", g.rev, dir+"/")
	cmd.Dir = g.root
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var names []string
	for _, ln := range strings.Fields(string(out)) {
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

// clState is what a tree says a release changes for a user.
type clState struct {
	Routes   map[string]bool // "METHOD /path" as registered, placeholders normalised to {}
	Flags    map[string]bool
	Families map[string][]string // capability-matrix name -> the other names it goes by (display name, model types)
	Pin      string              // the aikit version go.mod requires
}

var placeholderRE = regexp.MustCompile(`\{[^}]*\}`)

// normRoute reads `POST /v1/jobs/{id}/events` as `POST /v1/jobs/{}/events`.
func normRoute(r string) string { return placeholderRE.ReplaceAllString(r, "{}") }

// serveRoutes parses the non-test files of internal/serveapp and returns every route literal passed to a HandleFunc or Handle call.
func serveRoutes(src clSource) (map[string]bool, error) {
	names, err := src.list("internal/serveapp")
	if err != nil {
		return nil, err
	}
	routes := map[string]bool{}
	fset := token.NewFileSet()
	for _, n := range names {
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		b, err := src.read("internal/serveapp/" + n)
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
					routes[normRoute(s)] = true
				}
			}
			return true
		})
	}
	return routes, nil
}

func readClState(src clSource) (*clState, error) {
	st := &clState{Flags: map[string]bool{}, Families: map[string][]string{}}
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

// owedItem is one change the section does not name.
type owedItem struct{ Kind, Name, Why string }

// changelogOwed compares base with head and returns what the section sec leaves unmentioned and is not waived.
func changelogOwed(base, head *clState, sec string, waived func(kind, name string) bool) []owedItem {
	var owed []owedItem
	add := func(kind, name, why string) {
		if !waived(kind, name) {
			owed = append(owed, owedItem{kind, name, why})
		}
	}
	nsec := normRoute(sec)
	// routes: new at head
	var newRoutes []string
	for r := range head.Routes {
		if !base.Routes[r] {
			newRoutes = append(newRoutes, r)
		}
	}
	sort.Strings(newRoutes)
	pathOf := func(r string) string { _, p, _ := strings.Cut(r, " "); return p }
	for _, r := range newRoutes {
		p := pathOf(r)
		if strings.Contains(nsec, p) {
			continue
		}
		covered := false
		for _, o := range newRoutes {
			op := pathOf(o)
			if op != p && strings.HasPrefix(p, strings.TrimSuffix(op, "/")+"/") && strings.Contains(nsec, op) {
				covered = true
				break
			}
		}
		if !covered {
			add("route", r, "a new route no section line names")
		}
	}
	// flags: new or removed
	flagRE := func(n string) *regexp.Regexp {
		return regexp.MustCompile(`-{1,2}` + regexp.QuoteMeta(n) + `([^A-Za-z0-9_-]|$)`)
	}
	for _, n := range sortedKeysBool(head.Flags) {
		if !base.Flags[n] && !flagRE(n).MatchString(sec) {
			add("flag", n, "a new flag")
		}
	}
	for _, n := range sortedKeysBool(base.Flags) {
		if !head.Flags[n] && !flagRE(n).MatchString(sec) {
			add("flag", n, "a removed flag")
		}
	}
	// families
	var fams []string
	for n := range head.Families {
		fams = append(fams, n)
	}
	sort.Strings(fams)
	for _, n := range fams {
		if _, had := base.Families[n]; had {
			continue
		}
		names := append([]string{n}, head.Families[n]...)
		hit := false
		for _, nm := range names {
			if nm != "" && strings.Contains(strings.ToLower(sec), strings.ToLower(nm)) {
				hit = true
				break
			}
		}
		if !hit {
			add("family", n, "a new model family")
		}
	}
	// the aikit pin
	if head.Pin != "" && head.Pin != base.Pin && !strings.Contains(sec, head.Pin) && !strings.Contains(sec, strings.TrimPrefix(head.Pin, "v")) {
		add("pin", "aikit "+head.Pin, "the aikit pin moved from "+base.Pin)
	}
	return owed
}

func sortedKeysBool(m map[string]bool) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// changelogFragmentDir holds the unreleased entries, one file per change; CHANGELOG.md's [Unreleased] section is a pointer to
// it, and scripts/changelog.py moves them into a release's section.
const changelogFragmentDir = "changelog.d"

// unreleasedPointerPrefix begins the one line CHANGELOG.md's [Unreleased] section holds, the pointer to changelogFragmentDir.
const unreleasedPointerPrefix = "Each unreleased change is a file in"

// unreleasedStray returns the [Unreleased] body's lines other than the pointer: an entry written there instead of in
// changelogFragmentDir, which a release would leave behind.
func unreleasedStray(body string) []string {
	var out []string
	for _, ln := range strings.Split(body, "\n") {
		if strings.TrimSpace(ln) != "" && !strings.HasPrefix(ln, unreleasedPointerPrefix) {
			out = append(out, ln)
		}
	}
	return out
}

// changelogFragments returns every unreleased entry in changelogFragmentDir, newest first as the release renders them.
// A tree without the directory has none.
func changelogFragments(src clSource) (string, error) {
	names, err := src.list(changelogFragmentDir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	var b strings.Builder
	for _, n := range names {
		if n == "README.md" || !strings.HasSuffix(n, ".md") {
			continue
		}
		t, err := src.read(changelogFragmentDir + "/" + n)
		if err != nil {
			return "", err
		}
		b.Write(t)
	}
	return b.String(), nil
}

// loadChangelogWaivers reads docs/changelog-waivers.txt (absent is none): `<kind> <name> <reason>`. The name may contain spaces
// only for a route (`route POST /v1/x reason...` is read as kind route, name "POST /v1/x"), so a route's name is its method and path.
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

// newestVersionTag is the newest v* tag reachable from HEAD that is not skip (a submodule's tag, `demo/agent/v0.22.0`, is not one).
func newestVersionTag(root, skip string) (string, error) {
	out, err := gitOut(root, "tag", "--merged", "HEAD", "--list", "v[0-9]*", "--sort=-v:refname")
	if err != nil {
		return "", err
	}
	for _, t := range strings.Fields(out) {
		if t != skip && !strings.Contains(t, "/") {
			return t, nil
		}
	}
	return "", fmt.Errorf("no v* tag reachable from HEAD")
}

func runChangelog(args []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate changelog", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	base := fs.String("base", "", "the revision to compare with (default: the newest v* tag reachable from HEAD that is not -section)")
	section := fs.String("section", "Unreleased", "the CHANGELOG section that must name the changes: Unreleased, or a version such as v0.23.0")
	headRev := fs.String("head", "", "compare this revision instead of the working tree (a backtest of a past release: -base v0.21.0 -head v0.22.0 -section v0.22.0)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(w, err)
		return 2
	}
	if *base == "" {
		if *base, err = newestVersionTag(root, *section); err != nil {
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
	sec, ok := clSection(string(doc), *section)
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
	// A scan that finds nothing reads as "nothing changed": the route pass must find routes in both trees.
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
	owed := changelogOwed(bs, hs, sec, waived)
	if len(stray) > 0 {
		owed = append(owed, owedItem{Kind: "entry", Name: strings.TrimSpace(stray[0]),
			Why: fmt.Sprintf("%d line(s) under CHANGELOG.md's [Unreleased], which holds only the pointer; move the entry into %s/", len(stray), changelogFragmentDir)})
	}
	fmt.Fprintf(w, "CHANGELOG [%s] against %s: %d routes, %d flags, %d families, aikit %s (was %s)\n", *section, *base, len(hs.Routes), len(hs.Flags), len(hs.Families), hs.Pin, bs.Pin)
	for _, k := range sortedKeysBool(used) {
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
