package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// B6, the DISPATCH half of sibling drift — a census, deliberately not a verdict.
//
// The class: a dispatch that names one member of a set instead of dispatching on the property it
// means. P7 was the instance — `matmulInto` special-cased `isW8A8(w)` and sent everything else to a
// path that allocates per call, so W4A8 never reached the per-stream Workspace its six call sites
// already passed in. Nothing failed; one quantization silently got a different implementation.
//
// WHY THIS IS NOT A VERDICT, measured rather than assumed. `if isW8A8(w)` is syntactically
// indistinguishable from a legitimate special case, and on this tree 4 of the 5 non-definition sites
// ARE legitimate — W8A8 genuinely has its own fused batched kernel. Only the 5th was the defect.
// Separating them requires knowing the intended set, which no lint has.
//
// So this detects CHANGE, not correctness. A green result means "the dispatch surface is what it was
// when a person last reviewed it" — it does NOT mean "no dispatch drift exists". Anyone reading a
// pass as the second thing has made the class's own mistake one level up.
//
// When this goes red: look at the new site and decide whether it dispatches on the MEMBER or on the
// PROPERTY. If the property, rewrite it. If it is a genuine special case, add it below with a
// one-line reason. Adding it without a reason defeats the census.

// declaredIdentityDispatch is every use of an identity predicate in a dispatch position, with why it
// is legitimate. Definition sites and comments are excluded by the scanner.
//
// KEYED ON CONTENT, NOT POSITION. The first version keyed on file:line and tripped the moment a
// mechanical edit (G2's minmax rewrite) removed three lines above a declared site: mlp.go:356 became
// :353, the site itself unchanged. A census that cries wolf on every reformat is a census someone
// disables, and the thing it is supposed to notice — a member being named in a dispatch — is a
// property of the CODE, not of where the code sits.
var declaredIdentityDispatch = map[string]string{
	"decoder/attention.go|if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {": "fused QKV batched W8A8 kernel — a real fused kernel exists only for W8A8, so the guard selects a capability, not a member",
	"decoder/forwardn.go|if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {":  "same fused QKV path on the batched-prefill forward",
	"decoder/forwardn.go|if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {":                   "fused gate+up batched W8A8 kernel — same capability guard",
	"decoder/mlp.go|if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {":                        "fused gate+up on the decode path — same capability guard",
	"decoder/weightmat.go|if isW8A8(w) {": "matmulInto's W8A8 branch. Legitimate SINCE P7 (91f359f): it now selects the " +
		"W8A8-specific QuantBackend kernel, and the int4 branch beside it reaches the same Workspace. " +
		"Before P7 this was the defect — the else-path silently excluded every other quantization",
}

// declaredTypeSwitches are type switches over a value whose dynamic type carries an identity. All
// three are GGUF metadata decoding, where the switch IS the parse and there is no property to
// dispatch on instead.
var declaredTypeSwitches = map[string]string{
	"decoder/gguf.go|switch n := e.(type) {":        "GGUF tensor-shape element decode — switching on the wire type is the parse",
	"decoder/gguf.go|switch v := kvArr[i].(type) {": "GGUF KV array element decode — same",
	"decoder/gguf.go|switch v := arr[i].(type) {":   "GGUF KV array element decode (nested) — same",
}

// The predicate set is DERIVED, not listed: a definition `func isX(... *linalg.WeightMat ...) bool`
// is a quantization-identity predicate; `func isT3Method(string) bool` is not. Scanning for every
// `is*(` call instead swept in a dozen architecture predicates from arch.go that have nothing to do
// with this class — a scanner whose noise would have got it disabled.
//
// A NEW identity predicate therefore enters the census automatically, which is the property that
// matters: the census must notice a member being named, including by a predicate that did not exist
// when it was written.
var (
	identityDef = regexp.MustCompile(`(?m)^func (is[A-Z][A-Za-z0-9]*)\([^)]*\*linalg\.WeightMat[^)]*\) bool`)
	typeSwitch  = regexp.MustCompile(`\.\(type\)`) // any type switch; the subject can be an index expr, a call, anything
)

func scanDispatchSites(t *testing.T) (ident, tsw []string) {
	t.Helper()
	var nScanned, nSkipped int
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	sort.Strings(files)

	// Pass 1: which predicates are quantization-identity predicates at all.
	names := map[string]bool{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			nSkipped++
			continue
		}
		nScanned++
		raw, _ := os.ReadFile(f)
		for _, m := range identityDef.FindAllStringSubmatch(string(raw), -1) {
			names[m[1]] = true
		}
	}
	if len(names) == 0 {
		t.Fatal("no quantization-identity predicate found — the definition pattern no longer matches, " +
			"so the census would report an empty surface rather than a changed one")
	}
	var alts []string
	for n := range names {
		alts = append(alts, n)
	}
	sort.Strings(alts)
	identityCall := regexp.MustCompile(`\b(` + strings.Join(alts, "|") + `)\(`)

	// Pass 2: their call sites.
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for line := range strings.SplitSeq(string(raw), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "func is") {
				continue
			}
			loc := fmt.Sprintf("decoder/%s|%s", f, trimmed)
			if identityCall.MatchString(line) {
				ident = append(ident, loc)
			}
			if typeSwitch.MatchString(line) {
				tsw = append(tsw, loc)
			}
		}
	}
	// DENOMINATOR. A census that reports only its numerator hides its own shrinkage: if the glob
	// stops matching, or the package is split, the count drops to a smaller green number and reads
	// exactly like a clean tree. So state the universe examined, every run.
	t.Logf("EXAMINED: %d non-test .go file(s) in package decoder/ (glob \"*.go\"; %d skipped as "+
		"_test.go). That is the whole denominator — dispatch outside decoder/ is NOT scanned here.",
		nScanned, nSkipped)
	return
}

func TestDispatchCensus(t *testing.T) {
	ident, tsw := scanDispatchSites(t)

	check := func(kind string, found []string, declared map[string]string) {
		fs := map[string]bool{}
		for _, f := range found {
			fs[f] = true
		}
		for _, f := range found {
			if _, ok := declared[f]; !ok {
				t.Errorf("%s: NEW site %s is not declared.\n"+
					"  Does it dispatch on the MEMBER or on the PROPERTY?\n"+
					"  If the property: rewrite it (see matmulInto, P7/91f359f).\n"+
					"  If a genuine special case: add it to the declared map WITH A REASON.\n"+
					"  This census detects change, not correctness — a person has to look.", kind, f)
			}
		}
		for d := range declared {
			if !fs[d] {
				t.Errorf("%s: declared site %s no longer exists — the declaration is stale.\n"+
					"  Remove it, or find where the dispatch moved to.", kind, d)
			}
		}
	}
	check("identity-predicate dispatch", ident, declaredIdentityDispatch)
	check("type switch", tsw, declaredTypeSwitches)

	// A scan that finds nothing is indistinguishable from a broken scanner.
	if len(ident) == 0 && len(tsw) == 0 {
		t.Fatal("the scanner found ZERO dispatch sites — it is broken, not the tree clean")
	}
	t.Logf("dispatch census: %d identity-predicate site(s), %d type switch(es), all declared "+
		"(keyed on content, so line moves do not trip it). "+
		"GREEN MEANS UNCHANGED, NOT CORRECT.", len(ident), len(tsw))
}
