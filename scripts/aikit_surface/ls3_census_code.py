#!/usr/bin/env python3
"""LS3's follow-up, the code half: the edits to goinfer's two census tests that go with ls3_census_fix.py's re-keying.

Usage: python3 scripts/aikit_surface/ls3_census_code.py [--check]

Each edit replaces one exact passage of internal/diag/kernelprint/census_test.go (K3) or
internal/diag/hwcensus/census_test.go, and fails by name if the passage is neither there nor already replaced, so a
change to the test on main is seen rather than skipped. A second run changes nothing.

  K3: scan linalg, linalg/attn, linalg/elem, linalg/bitvec, internal/cpufeat and embed; map cpufeat's Has* flags (and
      embed's own hasAVX2); evaluate a cpufeat.HasX selector; and recognise the moved packages' qualifiers in the
      uncalled check.
  hwcensus: scan the go.work aikit when one is in effect (else go.mod's pin), and match Has<X> as well as has<X>.
  prezerolint (LS4 and LS2): read the output contracts of linalg/attn, linalg/elem and linalg/bitvec too, and name
      exported writers as its anti-vacuity sentinels (SoftmaxRowInto, ExpF32Into and QuantizeActivationsInto moved
      to elem or were unexported).
"""

import subprocess
import sys

K3 = "internal/diag/kernelprint/census_test.go"
HW = "internal/diag/hwcensus/census_test.go"
PZ = "internal/guard/prezerolint/prezero_test.go"

EDITS = [
    (K3, '''// censusPackages are the aikit packages whose kernels the suite runs.
var censusPackages = []string{"linalg", "embed"}''', '''// censusPackages are the aikit packages whose kernels the suite runs, and internal/cpufeat, whose tier switches
// (DisableTopTier) name the predicates too.
var censusPackages = []string{"linalg", "linalg/attn", "linalg/elem", "linalg/bitvec", "internal/cpufeat", "embed"}'''),
    (K3, '''// predicateValue maps each census predicate to its value in package pkg under this configuration. A census predicate
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
}''', '''// predicateValue maps each census predicate to its value in package pkg under this configuration. A census predicate
// missing here fails the test: a new dispatch flag needs a line saying which tier it follows. The kernel packages read
// internal/cpufeat's Has* flags (a condition names them as cpufeat.HasX; the site scan matches the identifier). embed
// keeps its own hasAVX2 (x/sys/cpu), which no aikit_* tag forces, so it follows the detected CPU, not the active tiers.
func predicateValue(pkg string, k linalg.Kernels) map[string]bool {
	on := func(tier string) bool { return slices.Contains(k.Active, tier) }
	return map[string]bool{
		"HasDotProd":      on("dotprod"),
		"HasAVX2":         on("avx2"),
		"HasF16C":         on("f16c"),
		"HasAVX512VNNI":   on("avx512vnni"),
		"HasAVX512VNNIVL": on("avx512vnni+vl"),
		"HasPOPCNT":       on("popcnt"),
		"HasQ4KAVX2":      on("avx2") && on("f16c"),
		"hasAVX2":         pkg == "embed" && slices.Contains(k.Detected, "avx2"),
	}
}'''),
    (K3, '''	ref := regexp.MustCompile(`\\b(linalg|embed)\\.([A-Z][A-Za-z0-9_]*)\\b`)''', '''	// A moved package may be imported under aikit<name> (the surface rewrite's qualifier where goinfer already has
	// an identifier of that name).
	ref := regexp.MustCompile(`\\b(?:aikit)?(linalg|attn|elem|bitvec|embed)\\.([A-Z][A-Za-z0-9_]*)\\b`)'''),
    (K3, '''	for fn, u := range uncalled {
		pkg, _, _ := strings.Cut(fn, ".")
''', '''	for fn, u := range uncalled {
		pkg, _, _ := strings.Cut(fn, ".")
		pkg = path.Base(pkg)
'''),
    (K3, '''	"os/exec"
	"path/filepath"''', '''	"os/exec"
	"path"
	"path/filepath"'''),
    (K3, '''	case *ast.UnaryExpr:
		if x.Op == token.NOT {
			switch eval(x.X, vals, derived) {''', '''	case *ast.SelectorExpr:
		// A kernel package names a flag as cpufeat.HasX.
		if pkg, ok := x.X.(*ast.Ident); ok && pkg.Name == "cpufeat" {
			return eval(x.Sel, vals, derived)
		}
	case *ast.UnaryExpr:
		if x.Op == token.NOT {
			switch eval(x.X, vals, derived) {'''),
    # LS6 step 1: the f32-scale and attach-row4 constructors are gone and the split-half ones are unexported helpers
    # behind WrapInt4 and RepackInt4InPlace; goinfer passes those LayoutCanonical and LayoutRow4, never LayoutSplitHalf.
    (K3, '''	"linalg.WrapInt4Row4F16":               {[]string{"WrapInt4Row4F16"}, "goinfer wraps row4 weights with WrapInt4Row4OnlyF16 or repacks with RepackInt4Row4"},
	"linalg.WrapInt4Row4Only":              {[]string{"WrapInt4Row4Only"}, "the f32-scale form; goinfer's int4 scales are binary16 (WrapInt4Row4OnlyF16)"},
	"linalg.RepackInt4Row4InPlace":         {[]string{"RepackInt4Row4InPlace"}, "the f32-scale form; goinfer calls RepackInt4Row4InPlaceF16"},
''', ""),
    (K3, '''	"linalg.RepackInt4SplitHalfInPlaceF16": {[]string{"RepackInt4SplitHalfInPlace", "RepackInt4SplitHalfInPlaceF16"}, "goinfer builds split-half only through (*WeightMat).RepackInt4SplitHalf"},
	"linalg.RepackInt4SplitHalfInPlace":    {[]string{"RepackInt4SplitHalfInPlace", "RepackInt4SplitHalfInPlaceF16"}, "as RepackInt4SplitHalfInPlaceF16"},
	"linalg.WrapInt4SplitHalfOnlyF16":      {[]string{"WrapInt4SplitHalfOnly", "WrapInt4SplitHalfOnlyF16"}, "as RepackInt4SplitHalfInPlaceF16"},
	"linalg.WrapInt4SplitHalfOnly":         {[]string{"WrapInt4SplitHalfOnly", "WrapInt4SplitHalfOnlyF16"}, "as RepackInt4SplitHalfInPlaceF16"},
''', '''	"linalg.repackInt4SplitHalfInPlace": {[]string{"LayoutSplitHalf"}, "RepackInt4InPlace's split-half form: goinfer asks WrapInt4 and RepackInt4InPlace for the canonical and row4 layouts only, and builds split-half through (*WeightMat).RepackInt4SplitHalf"},
	"linalg.wrapInt4SplitHalfOnly":      {[]string{"LayoutSplitHalf"}, "WrapInt4's split-half form, as repackInt4SplitHalfInPlace"},
'''),
    (PZ, '''	for _, sub := range []string{"linalg", "embed", "encoder"} {''',
     '''	for _, sub := range []string{"linalg", "linalg/attn", "linalg/elem", "linalg/bitvec", "embed", "encoder"} {'''),
    (PZ, '''	for _, must := range []string{"MatmulBT", "MatmulBTW4A8Into", "SoftmaxRowInto", "ExpF32Into", "QuantizeActivationsInto", "DequantizeRowInt4"} {''',
     '''	for _, must := range []string{"MatmulBT", "MatmulBTW4A8F16Into", "SoftmaxRowContractInto", "ExpContractInto", "QuantizeActivationsGroupedInto", "DequantizeRowInt4"} {'''),
    (PZ, '''	if len(writers) < 60 {
		t.Fatalf("only %d writers with an overwrite/zeroes contract found in the pinned aikit; expected at least 60", len(writers))''',
     '''	// A floor against a scan that reads nothing; the sentinels above already prove the contracts are there. The
	// linalg surface rework (aikit LS2, LS6) unexported several writers, so the exported count fell from 60-odd to 56.
	if len(writers) < 50 {
		t.Fatalf("only %d writers with an overwrite/zeroes contract found in the pinned aikit; expected at least 50", len(writers))'''),
    (HW, '''// metal/internal/gpudev included) and aikit's root module at the version go.mod pins, where the CPU feature predicates
// live in linalg. A pinned module that cannot be found FAILS the test rather than skipping it: a census that silently left
// aikit out would cover nothing, the citation lint's rule for the same situation.''', '''// metal/internal/gpudev included) and aikit's root module: the checkout a go.work in effect uses, or else the version
// go.mod pins (CI's case). aikit's CPU feature predicates live in internal/cpufeat, and embed keeps its own hasAVX2. A
// pinned module that cannot be found FAILS the test rather than skipping it: a census that silently left aikit out would
// cover nothing, the citation lint's rule for the same situation.'''),
    (HW, '''//   - cpu:<name>   a package-level `var has<X> = ...` (aikit's dispatch flags: hasAVX2, hasDotProd, hasF16C, ...)''',
     '''//   - cpu:<name>   a package-level `var has<X> = ...` or `var Has<X> = ...` (aikit's dispatch flags: cpufeat's HasAVX2,
//     HasDotProd, HasF16C, ..., and embed's own hasAVX2)'''),
    (HW, 'cpuFlag      = regexp.MustCompile(`^has[A-Z][A-Za-z0-9]*$`)', 'cpuFlag      = regexp.MustCompile(`^[Hh]as[A-Z][A-Za-z0-9]*$`)'),
    (HW, '''// discover scans goinfer and the pinned aikit modules.
func discover(t *testing.T) map[string][]string {
	t.Helper()
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	found := map[string][]string{}
	scanTree(t, root, "goinfer", found)
	for _, v := range pinned(t, root, "github.com/townsendmerino/aikit", "go.mod") {''', '''// discover scans goinfer and aikit: the go.work checkout when one is in effect, else every version go.mod pins.
func discover(t *testing.T) map[string][]string {
	t.Helper()
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	found := map[string][]string{}
	scanTree(t, root, "goinfer", found)
	if dir := workspaceModuleDir(t, root, "github.com/townsendmerino/aikit"); dir != "" {
		scanTree(t, dir, "aikit", found)
		return found
	}
	for _, v := range pinned(t, root, "github.com/townsendmerino/aikit", "go.mod") {'''),
    (HW, '''// census is docs/hardware-coverage.json.''', '''// workspaceModuleDir is the directory of mod when a go.work in effect at root uses a checkout of it (the build then
// compiles that tree, not go.mod's pin), or "" when none does.
func workspaceModuleDir(t *testing.T, root, mod string) string {
	t.Helper()
	cmd := exec.Command("go", "env", "GOWORK")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go env GOWORK: %v", err)
	}
	if w := strings.TrimSpace(string(out)); w == "" || w == "off" {
		return ""
	}
	cmd = exec.Command("go", "list", "-m", "-f", "{{if .Main}}{{.Dir}}{{end}}", mod)
	cmd.Dir = root
	out, err = cmd.Output()
	if err != nil {
		t.Fatalf("go list -m %s: %v", mod, err)
	}
	return strings.TrimSpace(string(out))
}

// census is docs/hardware-coverage.json.'''),
    (HW, '''var hasAVX9 = detectAVX9()
var (''', '''var hasAVX9 = detectAVX9()
var HasAVX10 = detectAVX10() // an exported flag, as internal/cpufeat declares them
var ('''),
    (HW, '''		"cpu:hasAVX9", "cpu:hasBlockFlag", "cuda:DeviceAttributeWarpSize",''',
     '''		"cpu:HasAVX10", "cpu:hasAVX9", "cpu:hasBlockFlag", "cuda:DeviceAttributeWarpSize",'''),
]


def main():
    check = "--check" in sys.argv[1:]
    srcs = {p: open(p).read() for p in {e[0] for e in EDITS}}
    out = dict(srcs)
    for path, old, new in EDITS:
        if (new and new in out[path]) or (not new and old not in out[path]):
            continue
        if old not in out[path]:
            sys.exit(f"{path}: the passage to replace is not there (main changed it?):\n{old[:200]}")
        out[path] = out[path].replace(old, new, 1)
    changed = [p for p in out if out[p] != srcs[p]]
    for p in changed:
        print(("would change " if check else "changed ") + p)
        if not check:
            open(p, "w").write(out[p])
            subprocess.run(["gofmt", "-w", p], check=True)  # the edited passages' alignment
    if check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
