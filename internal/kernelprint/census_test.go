package kernelprint

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

const (
	aikitModule = "github.com/townsendmerino/aikit"
	censusPath  = "../../docs/hardware-coverage.json"
)

// censusPackages are the aikit packages whose kernels the suite runs.
var censusPackages = []string{"linalg", "embed"}

// uncalled lists aikit functions holding census sites that goinfer's production code never reaches, keyed
// package.Function, with the exported entry points they are reached through. The test checks that goinfer's non-test
// sources name none of those entry points, so the claim cannot go stale silently.
var uncalled = map[string]struct {
	via    []string
	reason string
}{
	"linalg.dotPartials16":                 {[]string{"MatmulBTGGUFQ4K", "MatmulBTGGUFQ6K"}, "the k-quant integer dot behind MatmulBTGGUFQ4K/Q6K; goinfer's Q4_K weights run WeightMat(q4k).MatmulBTInto"},
	"linalg.WrapInt4Row4F16":               {[]string{"WrapInt4Row4F16"}, "goinfer wraps row4 weights with WrapInt4Row4OnlyF16 or repacks with RepackInt4Row4"},
	"linalg.WrapInt4Row4Only":              {[]string{"WrapInt4Row4Only"}, "the f32-scale form; goinfer's int4 scales are binary16 (WrapInt4Row4OnlyF16)"},
	"linalg.RepackInt4Row4InPlace":         {[]string{"RepackInt4Row4InPlace"}, "the f32-scale form; goinfer calls RepackInt4Row4InPlaceF16"},
	"linalg.dotNEON8x4":                    {[]string{"Dot8x4", "MatmulBTQ8Fused", "MatmulBTQ8FusedInto"}, "backs Dot8x4, which off arm64 only the fused Q8 matmul calls (the f32 packed path is arm64-gated); goinfer calls neither"},
	"linalg.dotI8x8":                       {[]string{"DotI8x8"}, "goinfer's int8 dots go through DotI8 and the W8A8 matmuls"},
	"linalg.expContractImpl":               {[]string{"ExpContractInto"}, "goinfer's only contract-exp call is SoftmaxRowContractInto"},
	"linalg.measureFMAPeak":                {[]string{"MeasuredFMAPeakGFLOPS"}, "a hardware probe, not a kernel goinfer runs"},
	"linalg.hammingRows":                   {[]string{"HammingRows"}, "binary-embedding search; goinfer runs no Hamming kernel"},
	"linalg.RepackInt4SplitHalfInPlaceF16": {[]string{"RepackInt4SplitHalfInPlace", "RepackInt4SplitHalfInPlaceF16"}, "goinfer builds split-half only through (*WeightMat).RepackInt4SplitHalf"},
	"linalg.RepackInt4SplitHalfInPlace":    {[]string{"RepackInt4SplitHalfInPlace", "RepackInt4SplitHalfInPlaceF16"}, "as RepackInt4SplitHalfInPlaceF16"},
	"linalg.WrapInt4SplitHalfOnlyF16":      {[]string{"WrapInt4SplitHalfOnly", "WrapInt4SplitHalfOnlyF16"}, "as RepackInt4SplitHalfInPlaceF16"},
	"linalg.WrapInt4SplitHalfOnly":         {[]string{"WrapInt4SplitHalfOnly", "WrapInt4SplitHalfOnlyF16"}, "as RepackInt4SplitHalfInPlaceF16"},
}

const (
	repair        = "SelfCheck's repair: it runs only when a kernel disagrees with its reference at goinfer's startup self-test"
	noSplitHalf   = "reached only through a split-half layout, which RepackInt4SplitHalf builds only where splitHalfUsable (AVX2 without VNNI+VL); with AVX2 off nothing has one"
	canonicalOnly = "goinfer repacks only canonical int4 weights (decoder's repackW4A8IfEligible), so q4 is never nil there; with the predicate decided the early return is required"
)

// unreachableSide lists census sites whose required side goinfer cannot reach in the configurations where the site
// reads as the verdict named, keyed file|function|condition|verdict. Each reason says why.
var unreachableSide = map[string]string{
	"linalg/kernels_arm64.go|disableTopTier|hasDotProd|true here":                                                                    repair,
	"linalg/kernels_arm64.go|disableTopTier|hasDotProd|false here":                                                                   repair,
	"linalg/kernels_amd64.go|disableTopTier|hasAVX2|true here":                                                                       repair,
	"linalg/kernels_amd64.go|disableTopTier|hasAVX2|false here":                                                                      repair,
	"linalg/kernels_amd64.go|disableTopTier|hasAVX512VNNI || hasAVX512VNNIVL|true here":                                              repair,
	"linalg/kernels_amd64.go|disableTopTier|hasAVX512VNNI || hasAVX512VNNIVL|false here":                                             repair,
	"linalg/weightmat_row4_arm64.go|(*WeightMat).RepackInt4Row4|w.q4 == nil || !hasDotProd|undecided here":                           canonicalOnly,
	"linalg/weightmat_splithalf_amd64.go|(*WeightMat).RepackInt4SplitHalf|w.q4 == nil || !hasAVX2 || hasAVX512VNNIVL|undecided here": canonicalOnly,
	"linalg/attn_acc64_amd64.go|avAcc64Blocks|!hasAVX2 || nKeys == 0|undecided here":                                                 "an attention row always has at least one key (a query attends to itself), so nKeys is never 0",
	"linalg/exp_contract_amd64.go|softmaxContractWithMaxImpl|!hasAVX2 || len(src) == 0|undecided here":                               "internal/whisper's softmax rows are attention rows over at least one key, never empty",
	"linalg/q4k_dot_amd64.go|dotQ4K|hasQ4KAVX2 && K >= qkK|undecided here":                                                           "a Q4_K row is whole 256-element super-blocks (WrapQ4K refuses other cols), so K >= qkK always",
	"linalg/dot_i8_scaled_amd64.go|w8a8GroupedSpan|hasAVX2 && nG > 0|undecided here":                                                 "its callers (matmulW8A8Grouped and the grouped batch) take it only for K a positive multiple of 32, so nG >= 1",
	"linalg/quant.go|w4a8BatchOp|op.SplitHalf != nil && splitHalfUsable() && M == 1 && group == 32 && K%group == 0|undecided here":   "goinfer's batch ops (decoder's wmW4A8Op) never set SplitHalf",
	"linalg/weightmat_splithalf_amd64.go|w4a8GroupedFastRowsQ|l.sh == nil || !splitHalfUsable()|true here":                           noSplitHalf,
	"linalg/matmul_w4a8_splithalf_tile_amd64.go|w4a8SplitHalfTileRows|!hasAVX2 || hasAVX512VNNIVL || M < 4 || K < 32|true here":      noSplitHalf,
	"linalg/weightmat_splithalf_amd64.go|w4a8GroupedFastRowsQ|l.sh == nil || !splitHalfUsable()|undecided here":                      "its one caller (the grouped W4A8 path) calls it only with l.sh or l.r4 set, and amd64 builds no row4 layout, so l.sh is never nil there",
}

// predicateValue maps each census predicate to its value in package pkg under this configuration. A census predicate
// missing here fails the test: a new dispatch flag needs a line saying which tier it follows. embed keeps its own
// hasAVX2 (x/sys/cpu), which no aikit_* tag forces, so it follows the detected CPU, not the active tiers.
func predicateValue(pkg string, k linalg.Kernels) map[string]bool {
	on := func(tier string) bool { return slices.Contains(k.Active, tier) }
	v := map[string]bool{
		"hasDotProd":      on("dotprod"),
		"hasAVX2":         on("avx2"),
		"hasF16C":         on("f16c"),
		"hasAVX512VNNI":   on("avx512vnni"),
		"hasAVX512VNNIVL": on("avx512vnni+vl"),
		"hasPOPCNT":       on("popcnt"),
		"hasQ4KAVX2":      on("avx2") && on("f16c"),
	}
	if pkg == "embed" {
		v["hasAVX2"] = slices.Contains(k.Detected, "avx2")
	}
	return v
}

// validated are the configurations K3's exclusions were written against (docs/measurements/vk1-2026-10.md). Elsewhere
// (an AVX-512 VNNI runner, a CPU without AVX2) the test says so and stops: an exclusion list read under a configuration
// nobody has run would fail or pass by accident.
var validated = []string{
	"arm64[dotprod]",
	"arm64[] forced=nodotprod",
	"amd64[avx2,f16c,popcnt]",
	"amd64[f16c,popcnt] forced=noavx2",
	"amd64[avx2,f16c] forced=nopopcnt",
}

// forcedTier is the tier each forcing tag turns off; a tag whose tier the CPU lacks forced nothing.
var forcedTier = map[string]string{"noavx2": "avx2", "noavx512": "avx512vnni", "nopopcnt": "popcnt", "nodotprod": "dotprod"}

// TestCompute_reachesCensus is K3: under this configuration, every `if` in aikit's linalg and embed whose condition names
// a census predicate has its active side executed by Compute, read from a coverage profile of a child run.
func TestCompute_reachesCensus(t *testing.T) {
	if testing.Short() {
		t.Skip("builds aikit with coverage; not in -short")
	}
	cfg, k := CurrentConfig(), linalg.ActiveKernels()
	for _, f := range cfg.Forced {
		if !slices.Contains(k.Detected, forcedTier[f]) {
			t.Skipf("VOID: tag aikit_%s forced nothing on this CPU (detected %v); run it where the CPU has %s", f, k.Detected, forcedTier[f])
		}
	}
	if !slices.Contains(validated, cfg.String()) {
		t.Skipf("configuration %s is not one K3 was validated under (%v); run it there and extend the exclusions", cfg, validated)
	}
	for _, p := range censusPredicates(t) {
		if _, ok := predicateValue("linalg", k)[p]; !ok {
			t.Fatalf("census predicate %s has no value mapping in predicateValue", p)
		}
	}
	tags := forcedTags(cfg)
	dir := aikitDir(t)
	profile := coverChild(t, tags)
	t.Logf("configuration %s, tags %q, aikit at %s", cfg, tags, dir)

	var sites, reached, excluded int
	var unreached []string
	seen := map[string]bool{}
	builtFiles := map[string]bool{}
	for _, pkg := range censusPackages {
		vals := predicateValue(pkg, k)
		files := parsePackage(t, filepath.Join(dir, pkg), tags)
		derived := derivedPredicates(files, vals)
		for _, f := range files {
			rel := pkg + "/" + filepath.Base(f.name)
			builtFiles[rel] = true
			for _, c := range allConds(f) {
				seen[rel+"|"+c] = true
			}
			for _, s := range censusSites(f, vals, derived) {
				sites++
				key := rel + "|" + s.fn + "|" + s.cond
				missing := []string{}
				for _, need := range s.need {
					if !profile.covered(aikitModule+"/"+rel, f.fset.Position(need.pos)) {
						missing = append(missing, need.side)
					}
				}
				switch {
				case len(missing) == 0:
					reached++
				case uncalled[pkg+"."+s.fn].reason != "",
					unreachableSide[key+"|"+s.verdict] != "":
					excluded++
				default:
					unreached = append(unreached, fmt.Sprintf("%s (%s, line %d): %s side not executed", key, s.verdict, f.fset.Position(s.pos).Line, strings.Join(missing, " and ")))
				}
			}
		}
	}
	for key := range unreachableSide {
		site := key[:strings.LastIndex(key, "|")]
		if file := strings.SplitN(key, "|", 2)[0]; builtFiles[file] && !seen[site] {
			t.Errorf("stale exclusion: %s is built here but has no such site", site)
		}
	}
	checkUncalled(t)
	sort.Strings(unreached)
	for _, u := range unreached {
		t.Errorf("unreached: %s", u)
	}
	t.Logf("%d census sites: %d reached, %d excluded as unreached by goinfer, %d unreached", sites, reached, excluded, len(unreached))
	if sites == 0 {
		t.Fatal("no census sites found: the scan covered nothing")
	}
}

// checkUncalled fails when goinfer's production code names an entry point uncalled says it does not.
func checkUncalled(t *testing.T) {
	ref := regexp.MustCompile(`\b(linalg|embed)\.([A-Z][A-Za-z0-9_]*)\b`)
	named := map[string]bool{}
	err := filepath.WalkDir("../..", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case "docs", "testdata", ".git", ".claude", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range ref.FindAllSubmatch(b, -1) {
			named[string(m[1])+"."+string(m[2])] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !named["linalg.MatmulBT"] {
		t.Fatal("the source scan found no linalg.MatmulBT: it read nothing")
	}
	for fn, u := range uncalled {
		pkg := strings.SplitN(fn, ".", 2)[0]
		for _, v := range u.via {
			if named[pkg+"."+v] {
				t.Errorf("uncalled says goinfer never reaches %s, but its production code names %s.%s", fn, pkg, v)
			}
		}
	}
}

// allConds returns function|condition for every `if` in f, so a stale exclusion can be told from one whose site is not a
// census site in this configuration.
func allConds(f parsedFile) []string {
	var out []string
	for _, d := range f.file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if is, ok := n.(*ast.IfStmt); ok {
				out = append(out, funcName(fd)+"|"+exprString(f.fset, is.Cond))
			}
			return true
		})
	}
	return out
}

func censusPredicates(t *testing.T) []string {
	b, err := os.ReadFile(censusPath)
	if err != nil {
		t.Fatal(err)
	}
	var c struct {
		Entries []struct {
			Predicates []string `json:"predicates"`
		} `json:"entries"`
	}
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range c.Entries {
		for _, p := range e.Predicates {
			if name, ok := strings.CutPrefix(p, "cpu:"); ok && !slices.Contains(out, name) {
				out = append(out, name)
			}
		}
	}
	if len(out) == 0 {
		t.Fatal("the census names no cpu predicates")
	}
	return out
}

func forcedTags(cfg Config) []string {
	var tags []string
	for _, f := range cfg.Forced {
		tags = append(tags, "aikit_"+f)
	}
	return tags
}

// aikitDir is the directory of the aikit module this test builds against: the module cache at go.mod's pin, or a
// go.work checkout.
func aikitDir(t *testing.T) string {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", aikitModule).Output()
	if err != nil {
		t.Fatalf("go list -m %s: %v", aikitModule, err)
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" {
		t.Fatalf("go list -m %s: no directory (module not downloaded?)", aikitModule)
	}
	return dir
}

// coverChild runs TestCompute_child with coverage over aikit's census packages and returns the profile.
func coverChild(t *testing.T, tags []string) coverProfile {
	tmp := t.TempDir()
	prof := filepath.Join(tmp, "cover.out")
	var pkgs []string
	for _, p := range censusPackages {
		pkgs = append(pkgs, aikitModule+"/"+p)
	}
	args := []string{"test", "-count=1", "-run", "^TestCompute_child$", "-covermode=set", "-coverpkg=" + strings.Join(pkgs, ","), "-coverprofile=" + prof}
	if len(tags) > 0 {
		args = append(args, "-tags", strings.Join(tags, ","))
	}
	cmd := exec.Command("go", append(args, ".")...)
	cmd.Env = append(os.Environ(), "GOINFER_KERNELPRINT_OUT="+filepath.Join(tmp, "report.json"))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("coverage child: %v\n%s", err, out)
	}
	return readProfile(t, prof)
}

type coverBlock struct {
	l0, c0, l1, c1 int
	count          int
}

type coverProfile map[string][]coverBlock

func readProfile(t *testing.T, path string) coverProfile {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p := coverProfile{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "mode:") {
			continue
		}
		// file:l0.c0,l1.c1 nstmt count
		colon := strings.LastIndex(line, ":")
		fields := strings.Fields(line[colon+1:])
		if colon < 0 || len(fields) != 3 {
			t.Fatalf("cover profile line %q", line)
		}
		var b coverBlock
		if _, err := fmt.Sscanf(fields[0], "%d.%d,%d.%d", &b.l0, &b.c0, &b.l1, &b.c1); err != nil {
			t.Fatalf("cover profile line %q: %v", line, err)
		}
		b.count, _ = strconv.Atoi(fields[2])
		p[line[:colon]] = append(p[line[:colon]], b)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return p
}

// covered reports whether a block holding pos ran.
func (p coverProfile) covered(file string, pos token.Position) bool {
	for _, b := range p[file] {
		after := pos.Line > b.l0 || pos.Line == b.l0 && pos.Column >= b.c0
		before := pos.Line < b.l1 || pos.Line == b.l1 && pos.Column < b.c1
		if after && before && b.count > 0 {
			return true
		}
	}
	return false
}

type parsedFile struct {
	name string
	fset *token.FileSet
	file *ast.File
}

// parsePackage parses the non-test files of dir that this GOOS, GOARCH and tags build.
func parsePackage(t *testing.T, dir string, tags []string) []parsedFile {
	ctx := build.Default
	ctx.GOOS, ctx.GOARCH = runtime.GOOS, runtime.GOARCH
	ctx.BuildTags = tags
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out []parsedFile
	for _, e := range ents {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if ok, err := ctx.MatchFile(dir, name); err != nil || !ok {
			continue
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, parsedFile{filepath.Join(dir, name), fset, f})
	}
	return out
}

// site is one census-predicate `if`, with the positions that must have run.
type site struct {
	fn, cond, verdict string
	pos               token.Pos
	need              []struct {
		side string
		pos  token.Pos
	}
}

// tri is a three-valued truth: the predicates are known, every other operand is not.
type tri int

const (
	triFalse tri = iota
	triTrue
	triUnknown
)

// derivedPredicates finds the package's functions whose body is `return <expr>` over a predicate or another such
// function (row4Usable, splitHalfUsable): a call to one is a predicate site too.
func derivedPredicates(files []parsedFile, vals map[string]bool) map[string]ast.Expr {
	derived := map[string]ast.Expr{}
	for changed := true; changed; {
		changed = false
		for _, f := range files {
			for _, d := range f.file.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || fd.Recv != nil || fd.Body == nil || len(fd.Body.List) != 1 || derived[fd.Name.Name] != nil {
					continue
				}
				ret, ok := fd.Body.List[0].(*ast.ReturnStmt)
				if ok && len(ret.Results) == 1 && mentions(ret.Results[0], vals, derived) {
					derived[fd.Name.Name] = ret.Results[0]
					changed = true
				}
			}
		}
	}
	return derived
}

func censusSites(f parsedFile, vals map[string]bool, derived map[string]ast.Expr) []site {
	var out []site
	for _, d := range f.file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		if _, isDerived := derived[fd.Name.Name]; isDerived {
			continue
		}
		next := successors(fd.Body)
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			is, ok := n.(*ast.IfStmt)
			if !ok || !mentions(is.Cond, vals, derived) {
				return true
			}
			s := site{fn: funcName(fd), cond: exprString(f.fset, is.Cond), pos: is.Pos()}
			add := func(side string, p token.Pos) {
				s.need = append(s.need, struct {
					side string
					pos  token.Pos
				}{side, p})
			}
			thenPos := is.Pos()
			if len(is.Body.List) > 0 {
				thenPos = is.Body.List[0].Pos()
			}
			elsePos := is.Pos()
			switch e := is.Else.(type) {
			case *ast.BlockStmt:
				if len(e.List) > 0 {
					elsePos = e.List[0].Pos()
				}
			case *ast.IfStmt:
				elsePos = e.Pos()
			case nil:
				if nx := next[is]; nx != nil {
					elsePos = nx.Pos()
				}
			}
			switch eval(is.Cond, vals, derived) {
			case triTrue:
				s.verdict = "true here"
				add("then", thenPos)
			case triFalse:
				s.verdict = "false here"
				add("else", elsePos)
			default:
				s.verdict = "undecided here"
				add("then", thenPos)
				add("else", elsePos)
			}
			out = append(out, s)
			return true
		})
	}
	return out
}

// successors maps each `if` in a statement list to the statement after it.
func successors(body *ast.BlockStmt) map[*ast.IfStmt]ast.Stmt {
	m := map[*ast.IfStmt]ast.Stmt{}
	walk := func(list []ast.Stmt) {
		for i, st := range list {
			if is, ok := st.(*ast.IfStmt); ok && i+1 < len(list) {
				m[is] = list[i+1]
			}
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch b := n.(type) {
		case *ast.BlockStmt:
			walk(b.List)
		case *ast.CaseClause:
			walk(b.Body)
		case *ast.CommClause:
			walk(b.Body)
		}
		return true
	})
	return m
}

func mentions(e ast.Expr, vals map[string]bool, derived map[string]ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Ident:
			if _, ok := vals[x.Name]; ok {
				found = true
			}
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok && derived[id.Name] != nil {
				found = true
			}
		}
		return !found
	})
	return found
}

func eval(e ast.Expr, vals map[string]bool, derived map[string]ast.Expr) tri {
	switch x := e.(type) {
	case *ast.ParenExpr:
		return eval(x.X, vals, derived)
	case *ast.Ident:
		if v, ok := vals[x.Name]; ok {
			if v {
				return triTrue
			}
			return triFalse
		}
	case *ast.UnaryExpr:
		if x.Op == token.NOT {
			switch eval(x.X, vals, derived) {
			case triTrue:
				return triFalse
			case triFalse:
				return triTrue
			}
		}
	case *ast.CallExpr:
		if id, ok := x.Fun.(*ast.Ident); ok && derived[id.Name] != nil {
			return eval(derived[id.Name], vals, derived)
		}
	case *ast.BinaryExpr:
		l, r := eval(x.X, vals, derived), eval(x.Y, vals, derived)
		switch x.Op {
		case token.LAND:
			if l == triFalse || r == triFalse {
				return triFalse
			}
			if l == triTrue && r == triTrue {
				return triTrue
			}
		case token.LOR:
			if l == triTrue || r == triTrue {
				return triTrue
			}
			if l == triFalse && r == triFalse {
				return triFalse
			}
		}
	}
	return triUnknown
}

func funcName(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return fd.Name.Name
	}
	var b bytes.Buffer
	printer.Fprint(&b, token.NewFileSet(), fd.Recv.List[0].Type)
	return "(" + b.String() + ")." + fd.Name.Name
}

func exprString(fset *token.FileSet, e ast.Expr) string {
	var b bytes.Buffer
	printer.Fprint(&b, fset, e)
	return b.String()
}
