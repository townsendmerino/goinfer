// Package hwcensus holds the hardware census test (docs/tasks/task-hardware-coverage-2026-10.md, H0). It has no
// non-test code: the census is docs/hardware-coverage.json, and this package checks it against the source.
package hwcensus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// H0, the hardware census: a census, deliberately not a verdict, in the shape of decoder/dispatch_census_test.go.
//
// The class: a code path selected by hardware our machines lack (an AVX-512 VNNI kernel in aikit was off the AVX2 path and surfaced only
// because GitHub's runner pool mixes CPU models; docs/code-notes/internal-hwcensus.md#hardwareCensus.header). Nothing here can tell whether
// such a path is correct. What it can do is make sure no hardware-gated branch lands unseen: every predicate that selects code by hardware
// must have an entry in docs/hardware-coverage.json, which says where that path last executed (or that it never has).
//
// So this detects CHANGE. A green result means "every hardware predicate in the source is in the census, and every
// census predicate is still in the source". It does NOT mean the paths are tested; the entries' last_executed records
// say that, and the never-executed list this logs is the honest summary.
//
// When this goes red: a new predicate needs an entry (what it selects, and where it has run, or [] for never); or an
// entry names a predicate the source no longer has, and the entry is stale. Adding an entry with a made-up record
// defeats the census: a record names a machine, a date, how it ran, both commits, and the gate that ran.
//
// WHAT IS SCANNED. goinfer's whole tree (all five modules) and, at the versions goinfer's go.mod files pin, aikit's root
// module and aikit/gpu: the CPU feature predicates live in aikit's linalg, and the CUDA and Metal device queries partly
// in aikit/gpu. A pinned module that cannot be found FAILS the test rather than skipping it: a census that silently left
// aikit out would cover nothing, the citation lint's rule for the same situation. go/scanner tokenizes each file, so a
// comment never counts and a string literal (an objc selector, a sysctl name, a /proc path) does.
//
// THE PREDICATES ARE DERIVED, NOT LISTED, so a new one enters the census without anyone remembering to add it here:
//   - cpu:<name>   a package-level `var has<X> = ...` (aikit's dispatch flags: hasAVX2, hasDotProd, hasF16C, ...)
//   - cuda:<name>  any DeviceAttribute<X> identifier (a CUDA device-attribute query)
//   - metal:<sel>  an objc selector registered with RegisterName whose name reads as a device capability or limit
//     (max*, supports*, recommended*, has*, threadExecutionWidth, currentAllocatedSize, registryID)
//   - mem:<probe>@<file>  a memory probe: a sysctl or /proc or cgroup name, or a driver call (MemGetInfo, MemInfo, ...),
//     keyed with its file, because two files reading the same name are two different gates; and every backend's
//     registered fit-guard probe, mem:probe:<backend>@<file> (decoder.RegisterMemoryProbe)
//
// Keys are by name, never by line, for the dispatch census's reason: a census that cries wolf on every reformat is a
// census someone disables.

// censusPath is the census, relative to this package.
const censusPath = "../../docs/hardware-coverage.json"

var (
	cpuFlag      = regexp.MustCompile(`^has[A-Z][A-Za-z0-9]*$`)
	cudaAttr     = regexp.MustCompile(`^DeviceAttribute[A-Z][A-Za-z0-9]*$`)
	metalDevSel  = regexp.MustCompile(`^(max[A-Z][A-Za-z0-9]*|supports[A-Z][A-Za-z0-9]*:?|recommended[A-Z][A-Za-z0-9]*|has[A-Z][A-Za-z0-9]*|threadExecutionWidth|currentAllocatedSize|registryID)$`)
	memProbeStr  = regexp.MustCompile(`^(hw\.memsize|hw\.physmem|hw\.usermem|vm\.swapusage|/proc/meminfo|/sys/fs/cgroup/.*memory.*)$`)
	memProbeCall = map[string]bool{"MemGetInfo": true, "MemInfo": true, "GlobalMemoryStatusEx": true, "Sysinfo": true}
)

// scanSource returns every predicate key in one Go file, mapped to the file.
func scanSource(src []byte, rel string) []string {
	fset := token.NewFileSet()
	f := fset.AddFile(rel, -1, len(src))
	var s scanner.Scanner
	s.Init(f, src, nil, 0) // mode 0: comments are skipped
	type tk struct {
		tok token.Token
		lit string
	}
	var toks []tk
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		toks = append(toks, tk{tok, lit})
	}
	var keys []string
	add := func(k string) { keys = append(keys, k) }
	unq := func(lit string) string {
		if v, err := strconv.Unquote(lit); err == nil {
			return v
		}
		return ""
	}
	inVarBlock, depth := false, 0
	for i, t := range toks {
		next := func(n int) tk {
			if i+n < len(toks) {
				return toks[i+n]
			}
			return tk{}
		}
		// Package-level dispatch flags: `var hasX = ...`, or `hasX = ...` inside a `var ( ... )` block.
		switch t.tok {
		case token.VAR:
			if next(1).tok == token.LPAREN {
				inVarBlock, depth = true, 0
			} else if next(1).tok == token.IDENT && cpuFlag.MatchString(next(1).lit) && next(2).tok == token.ASSIGN {
				add("cpu:" + next(1).lit)
			}
		case token.LPAREN:
			if inVarBlock {
				depth++
			}
		case token.RPAREN:
			if inVarBlock {
				depth--
				if depth == 0 {
					inVarBlock = false
				}
			}
		case token.IDENT:
			if inVarBlock && depth == 1 && cpuFlag.MatchString(t.lit) && next(1).tok == token.ASSIGN {
				add("cpu:" + t.lit)
			}
			if cudaAttr.MatchString(t.lit) {
				add("cuda:" + t.lit)
			}
			if memProbeCall[t.lit] && next(1).tok == token.LPAREN {
				add("mem:" + t.lit + "@" + rel)
			}
			// A backend's free-memory probe for the fit guard: decoder.RegisterMemoryProbe("<backend>", ...).
			if t.lit == "RegisterMemoryProbe" && next(1).tok == token.LPAREN && next(2).tok == token.STRING {
				add("mem:probe:" + unq(next(2).lit) + "@" + rel)
			}
		case token.STRING:
			v := unq(t.lit)
			if i >= 2 && toks[i-1].tok == token.LPAREN && toks[i-2].tok == token.IDENT && toks[i-2].lit == "RegisterName" && metalDevSel.MatchString(v) {
				add("metal:" + strings.TrimSuffix(v, ":"))
			}
			if memProbeStr.MatchString(v) {
				add("mem:" + v + "@" + rel)
			}
		}
	}
	return keys
}

// scanTree scans every non-test .go file under dir, keying files as prefix/<path>. testdata, _to_delete and dot
// directories are skipped.
func scanTree(t *testing.T, dir, prefix string, into map[string][]string) {
	t.Helper()
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if p != dir && (name == "testdata" || name == "_to_delete" || name == "node_modules" || strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		rel = prefix + "/" + filepath.ToSlash(rel)
		for _, k := range scanSource(src, rel) {
			into[k] = appendUnique(into[k], rel)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scanning %s: %v", dir, err)
	}
}

func appendUnique(s []string, v string) []string {
	if slices.Contains(s, v) {
		return s
	}
	return append(s, v)
}

// pinned returns the version of module mod that the go.mod files under root require (every distinct one).
func pinned(t *testing.T, root, mod string, gomods ...string) []string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^\s*` + regexp.QuoteMeta(mod) + `\s+(v[^\s]+)`)
	var vs []string
	for _, g := range gomods {
		b, err := os.ReadFile(filepath.Join(root, g))
		if err != nil {
			t.Fatalf("reading %s: %v", g, err)
		}
		for _, m := range re.FindAllStringSubmatch(string(b), -1) {
			vs = appendUnique(vs, m[1])
		}
	}
	if len(vs) == 0 {
		t.Fatalf("no go.mod under %s requires %s (looked in %v)", root, mod, gomods)
	}
	return vs
}

// moduleDir finds mod@ver's source in the module cache, downloading it if needed. It FAILS rather than skipping:
// a census that could not read aikit would report every aikit predicate missing from the source, or worse, pass.
func moduleDir(t *testing.T, root, mod, ver string) string {
	t.Helper()
	cmd := exec.Command("go", "mod", "download", "-json", mod+"@"+ver)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod")
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		t.Fatalf("CANNOT SEARCH %s@%s (go mod download: %v %s): the census needs the pinned source; fix with `go mod download`, never by skipping", mod, ver, err, errb.String())
	}
	var j struct{ Dir string }
	if err := json.Unmarshal(out.Bytes(), &j); err != nil || j.Dir == "" {
		t.Fatalf("CANNOT SEARCH %s@%s: no module directory in %q", mod, ver, out.String())
	}
	return j.Dir
}

// discover scans goinfer and the pinned aikit modules.
func discover(t *testing.T) map[string][]string {
	t.Helper()
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	found := map[string][]string{}
	scanTree(t, root, "goinfer", found)
	for _, v := range pinned(t, root, "github.com/townsendmerino/aikit", "go.mod") {
		scanTree(t, moduleDir(t, root, "github.com/townsendmerino/aikit", v), "aikit", found)
	}
	for _, v := range pinned(t, root, "github.com/townsendmerino/aikit/gpu", "metal/go.mod", "cuda/go.mod") {
		scanTree(t, moduleDir(t, root, "github.com/townsendmerino/aikit/gpu", v), "aikit/gpu", found)
	}
	return found
}

// census is docs/hardware-coverage.json.
type census struct {
	About       string  `json:"_about"`
	InventoryAt string  `json:"inventory_at"`
	Entries     []entry `json:"entries"`
}

type entry struct {
	ID           string   `json:"id"`
	Gate         string   `json:"gate"`
	Kind         string   `json:"kind"`
	Predicates   []string `json:"predicates"`
	SelectedWhen string   `json:"selected_when"`
	Files        []string `json:"files"`
	OurMachines  string   `json:"our_machines"`
	LastExecuted []record `json:"last_executed"`
	Notes        string   `json:"notes"`
}

type record struct {
	Machine string `json:"machine"`
	Date    string `json:"date"`
	How     string `json:"how"`
	Goinfer string `json:"goinfer"`
	Aikit   string `json:"aikit"`
	Gate    string `json:"gate"`
}

var (
	validHow  = map[string]bool{"native": true, "emulated": true, "ci-pool": true, "forced": true}
	validKind = map[string]bool{"cpu-feature": true, "cuda-device": true, "metal-device": true, "memory-probe": true, "size": true, "platform": true, "driver-os": true}
	isoDate   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

func loadCensus(t *testing.T) census {
	t.Helper()
	b, err := os.ReadFile(censusPath)
	if err != nil {
		t.Fatalf("reading the census: %v", err)
	}
	var c census
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		t.Fatalf("parsing %s: %v", censusPath, err)
	}
	return c
}

// TestHardwareCensus: every hardware predicate in the source has a census entry, every census predicate is still in the
// source, and every last_executed record is a whole record. It logs the never-executed entries (H6's list).
func TestHardwareCensus(t *testing.T) {
	found := discover(t)
	c := loadCensus(t)
	if os.Getenv("HWCENSUS_DUMP") != "" {
		var ks []string
		for k := range found {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		for _, k := range ks {
			t.Logf("%s\t%s", k, strings.Join(found[k], ", "))
		}
	}
	listed := map[string]string{}
	ids := map[string]bool{}
	for _, e := range c.Entries {
		if e.ID == "" || ids[e.ID] {
			t.Errorf("entry %q: missing or duplicate id", e.ID)
		}
		ids[e.ID] = true
		if !validKind[e.Kind] {
			t.Errorf("entry %s: kind %q is not one of %v", e.ID, e.Kind, keys(validKind))
		}
		if e.Gate == "" || e.SelectedWhen == "" || e.OurMachines == "" {
			t.Errorf("entry %s: gate, selected_when and our_machines are required", e.ID)
		}
		for _, p := range e.Predicates {
			if prev, dup := listed[p]; dup {
				t.Errorf("predicate %s is in two entries (%s and %s)", p, prev, e.ID)
			}
			listed[p] = e.ID
		}
		for i, r := range e.LastExecuted {
			if r.Machine == "" || r.Goinfer == "" || r.Aikit == "" || r.Gate == "" || !isoDate.MatchString(r.Date) || !validHow[r.How] {
				t.Errorf("entry %s, last_executed[%d] %+v: a record names a machine, an ISO date, how (one of %v), both commits and the gate that ran", e.ID, i, r, keys(validHow))
			}
		}
	}
	var missing, stale []string
	for k, files := range found {
		if _, ok := listed[k]; !ok {
			missing = append(missing, k+"  ("+strings.Join(files, ", ")+")")
		}
	}
	for k, id := range listed {
		if _, ok := found[k]; !ok {
			stale = append(stale, k+"  (entry "+id+")")
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	for _, m := range missing {
		t.Errorf("hardware predicate with no census entry: %s — add it to docs/hardware-coverage.json with what it selects and where it last ran ([] for never)", m)
	}
	for _, s := range stale {
		t.Errorf("census predicate no longer in the source: %s — the entry is stale", s)
	}
	var never []string
	for _, e := range c.Entries {
		if len(e.LastExecuted) == 0 {
			never = append(never, e.ID+": "+e.Gate)
		}
	}
	t.Logf("%d predicates found, %d census entries; NEVER EXECUTED (%d):\n  %s", len(found), len(c.Entries), len(never), strings.Join(never, "\n  "))
}

func keys(m map[string]bool) []string {
	var s []string
	for k := range m {
		s = append(s, k)
	}
	sort.Strings(s)
	return s
}

// TestScanSource pins the scanner on inputs whose answer is known, so a scanner change that stops seeing a predicate
// fails here, not by the census quietly shrinking: a comment never counts, a string literal does, a local `hasX := `
// is not a dispatch flag, and a var block is read.
func TestScanSource(t *testing.T) {
	src := []byte(`package x
// var hasInComment = detect()   (a comment: not a predicate)
var hasAVX9 = detectAVX9()
var (
	hasBlockFlag = detect()
	other        = 1
)
func f() {
	hasLocal := true // a local, not a dispatch flag
	_ = hasLocal
	_ = d.Attribute(gc.DeviceAttributeWarpSize)
	selA = objc.RegisterName("maxTotalThreadsPerThreadgroup")
	selB = objc.RegisterName("newBufferWithLength:options:") // not a capability query
	_, _ = unix.Sysctl("hw.memsize")
	decoder.RegisterMemoryProbe("tpu", func() (int64, bool) { free, _, _ := dev.Context().MemInfo(); return int64(free), true })
	_ = os.ReadFile("/proc/meminfo")
	_ = "maxTokens" // a string that is not a selector registration
}
`)
	got := scanSource(src, "goinfer/x.go")
	sort.Strings(got)
	want := []string{
		"cpu:hasAVX9", "cpu:hasBlockFlag", "cuda:DeviceAttributeWarpSize",
		"mem:/proc/meminfo@goinfer/x.go", "mem:MemInfo@goinfer/x.go", "mem:hw.memsize@goinfer/x.go", "mem:probe:tpu@goinfer/x.go",
		"metal:maxTotalThreadsPerThreadgroup",
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("scanSource:\n got %v\nwant %v", got, want)
	}
}
