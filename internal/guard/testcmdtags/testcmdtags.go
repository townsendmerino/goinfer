// Package testcmdtags checks a `go test ... -run <regex> <packages>` command against the tests it names: a test whose file's
// build constraint the command's tags cannot satisfy never compiles, and `go test` prints ok with no SKIP line for it, so
// the command reads as a pass of something it did not run.
package testcmdtags

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// Finding is one problem with a command: a named test its tags leave uncompiled, or a -run alternative that names no test.
type Finding struct {
	Test, Package, Why string
}

func (f Finding) String() string { return f.Package + ": " + f.Test + ": " + f.Why }

// platform reports whether a build tag is a target's OS, architecture or toolchain term, which a command line does not
// choose (its GOOS and GOARCH come from the box that runs it), as opposed to a tag -tags sets.
func platform(tag string) bool {
	switch tag {
	case "unix", "cgo", "gc", "gccgo", "purego",
		"linux", "darwin", "windows", "freebsd", "openbsd", "netbsd", "android", "ios", "js", "wasip1", "plan9", "solaris", "illumos", "aix", "dragonfly",
		"amd64", "arm64", "386", "arm", "wasm", "riscv64", "ppc64", "ppc64le", "s390x", "mips", "mipsle", "mips64", "mips64le", "loong64":
		return true
	}
	return strings.HasPrefix(tag, "go1.")
}

// satisfiable reports whether some target satisfies expr with exactly tags set: the platform terms are free, every other
// tag is set when the command names it.
func satisfiable(expr constraint.Expr, tags map[string]bool) bool {
	var free []string
	seen := map[string]bool{}
	expr.Eval(func(tag string) bool {
		if platform(tag) && !seen[tag] {
			seen[tag] = true
			free = append(free, tag)
		}
		return false
	})
	for mask := 0; mask < 1<<len(free); mask++ {
		on := map[string]bool{}
		for i, t := range free {
			on[t] = mask&(1<<i) != 0
		}
		if expr.Eval(func(tag string) bool {
			if platform(tag) {
				return on[tag]
			}
			return tags[tag]
		}) {
			return true
		}
	}
	return false
}

// testFunc is a top-level test, benchmark, example or fuzz target and the constraint of its file (nil: none).
type testFunc struct {
	name string
	expr constraint.Expr
}

// parsed caches testsIn by directory: a doc scan asks for the same packages many times.
var parsed sync.Map // string -> []testFunc

// testsIn lists the test functions declared in dir's _test.go files.
func testsIn(dir string) ([]testFunc, error) {
	if v, ok := parsed.Load(dir); ok {
		return v.([]testFunc), nil
	}
	out, err := parseTests(dir)
	if err == nil {
		parsed.Store(dir, out)
	}
	return out, err
}

func parseTests(dir string) ([]testFunc, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*_test.go"))
	if err != nil {
		return nil, err
	}
	var out []testFunc
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		var expr constraint.Expr
		for _, line := range strings.Split(string(src), "\n") {
			if constraint.IsGoBuild(line) {
				if expr, err = constraint.Parse(line); err != nil {
					return nil, fmt.Errorf("%s: %w", f, err)
				}
				break
			}
			if strings.HasPrefix(line, "package ") {
				break
			}
		}
		file, err := parser.ParseFile(token.NewFileSet(), f, src, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		for _, d := range file.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv != nil {
				continue
			}
			for _, p := range []string{"Test", "Benchmark", "Example", "Fuzz"} {
				if strings.HasPrefix(fd.Name.Name, p) {
					out = append(out, testFunc{fd.Name.Name, expr})
					break
				}
			}
		}
	}
	return out, nil
}

// Command is a parsed `go test` invocation: the directory it runs in, its tags, its -run expression and its packages.
type Command struct {
	Dir      string
	Tags     map[string]bool
	Run      string
	Packages []string
}

// tokens splits a shell command line into words, honouring single and double quotes.
func tokens(s string) []string {
	var out []string
	var cur strings.Builder
	var quote rune
	have := false
	for _, r := range s {
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '\'' || r == '"':
			quote, have = r, true
		case r == ' ' || r == '\t':
			if have {
				out = append(out, cur.String())
				cur.Reset()
				have = false
			}
		default:
			cur.WriteRune(r)
			have = true
		}
	}
	if have {
		out = append(out, cur.String())
	}
	return out
}

func addTags(tags map[string]bool, v string) {
	for _, t := range strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' }) {
		tags[t] = true
	}
}

// Parse reads a command line: an optional `cd <dir> &&` prefix, GOFLAGS=-tags=... in the environment, and `go test` with
// -tags, -run and package arguments. ok is false when the line holds no `go test`.
func Parse(line string) (c Command, ok bool) {
	c = Command{Dir: ".", Tags: map[string]bool{}}
	for _, part := range strings.Split(line, "&&") {
		w := tokens(strings.TrimSpace(part))
		if len(w) >= 2 && w[0] == "cd" {
			c.Dir = filepath.Join(c.Dir, w[1])
			continue
		}
		i := 0
		for i < len(w) && strings.Contains(w[i], "=") && !strings.HasPrefix(w[i], "-") {
			if v, found := strings.CutPrefix(w[i], "GOFLAGS="); found {
				for _, f := range strings.Fields(v) {
					if tv, ok := strings.CutPrefix(f, "-tags="); ok {
						addTags(c.Tags, tv)
					}
				}
			}
			i++
		}
		if i+1 >= len(w) || w[i] != "go" || w[i+1] != "test" {
			continue
		}
		ok = true
		for j := i + 2; j < len(w); j++ {
			a := w[j]
			switch {
			case a == "-tags" && j+1 < len(w):
				j++
				addTags(c.Tags, w[j])
			case strings.HasPrefix(a, "-tags="):
				addTags(c.Tags, strings.TrimPrefix(a, "-tags="))
			case a == "-run" && j+1 < len(w):
				j++
				c.Run = w[j]
			case strings.HasPrefix(a, "-run="):
				c.Run = strings.TrimPrefix(a, "-run=")
			case a == "-args":
				j = len(w)
			case strings.HasPrefix(a, "-"):
				if j+1 < len(w) && !strings.HasPrefix(w[j+1], "-") && !strings.HasPrefix(w[j+1], ".") && !strings.Contains(a, "=") &&
					(a == "-timeout" || a == "-count" || a == "-bench" || a == "-benchtime" || a == "-p" || a == "-parallel" || a == "-cpu") {
					j++
				}
			case a == "." || strings.HasPrefix(a, "./") || strings.HasPrefix(a, "../"):
				c.Packages = append(c.Packages, a)
			}
		}
		return c, ok
	}
	return c, ok
}

// alternatives splits the top level of a -run expression's first segment (before any subtest `/`) on `|`. A whole expression
// anchored as ^(...)$ keeps the anchors on each alternative.
func alternatives(run string) []string {
	seg := splitTop(run, '/')[0]
	inner, anchored := strings.CutPrefix(seg, "^(")
	if anchored {
		inner, anchored = strings.CutSuffix(inner, ")$")
	}
	if !anchored {
		return splitTop(seg, '|')
	}
	var out []string
	for _, a := range splitTop(inner, '|') {
		out = append(out, "^(?:"+a+")$")
	}
	return out
}

// splitTop splits s on sep outside parentheses and brackets.
func splitTop(s string, sep rune) []string {
	var out []string
	depth, start := 0, 0
	for i, r := range s {
		switch r {
		case '(', '[':
			depth++
		case ')', ']':
			depth--
		case sep:
			if depth == 0 {
				out = append(out, s[start:i])
				start = i + 1
			}
		}
	}
	return append(out, s[start:])
}

// Check resolves c against root: the test functions its -run names in its packages, and, for each, whether the command's
// tags compile a file that declares it. A command with no -run, -run '^$' or no package is not checked.
func Check(root string, c Command) ([]Finding, error) {
	if c.Run == "" || c.Run == "^$" || len(c.Packages) == 0 {
		return nil, nil // no -run, or the idiom that runs none (before -bench)
	}
	var dirs []string
	for _, p := range c.Packages {
		base := filepath.Join(root, c.Dir, strings.TrimSuffix(p, "/..."))
		if st, err := os.Stat(base); err != nil || !st.IsDir() {
			rel, _ := filepath.Rel(root, base)
			return []Finding{{p, filepath.ToSlash(rel), "no such package directory: the path is stale or the command runs from another directory"}}, nil
		}
		if !strings.HasSuffix(p, "/...") {
			dirs = append(dirs, base)
			continue
		}
		err := filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() && (d.Name() == "testdata" || strings.HasPrefix(d.Name(), ".")) && path != base {
				return filepath.SkipDir
			}
			if d.IsDir() {
				dirs = append(dirs, path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	var out []Finding
	for _, alt := range alternatives(c.Run) {
		re, err := regexp.Compile(alt)
		if err != nil {
			return nil, fmt.Errorf("-run %q: %w", alt, err)
		}
		matched := false
		for _, d := range dirs {
			fns, err := testsIn(d)
			if err != nil {
				return nil, err
			}
			rel, _ := filepath.Rel(root, d)
			byName := map[string][]constraint.Expr{}
			for _, f := range fns {
				if re.MatchString(f.name) {
					byName[f.name] = append(byName[f.name], f.expr)
				}
			}
			names := make([]string, 0, len(byName))
			for n := range byName {
				names = append(names, n)
			}
			sort.Strings(names)
			for _, n := range names {
				matched = true
				compiles := false
				var need []string
				for _, e := range byName[n] {
					if e == nil || satisfiable(e, c.Tags) {
						compiles = true
						break
					}
					need = append(need, e.String())
				}
				if !compiles {
					out = append(out, Finding{n, filepath.ToSlash(rel), "its file needs //go:build " + strings.Join(need, " or ") +
						"; the command's tags do not compile it, so it never runs and the run still prints ok"})
				}
			}
		}
		if !matched {
			out = append(out, Finding{alt, filepath.ToSlash(c.Dir), "-run alternative names no test in the packages given"})
		}
	}
	return out, nil
}
