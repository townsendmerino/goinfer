package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/parser"
	"go/token"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// EQUIVALENCE for `gate identity` (TE6(b), docs/tasks/task-test-efficiency-2026-09.md). Each claim
// is pinned by a test here, and each was shown to go RED with the property deliberately broken
// (recorded in the TE6(b) report; reproduce with `go run ./cmd/gate mutation …`):
//
//  1. a comment-only change between two revisions comes back IDENTICAL for every family;
//  2. a 1-ulp numeric change in one path is flagged DIFFERENT for exactly the families that execute
//     it (the NormParallel residual: cohere and cohere2), and every other family stays IDENTICAL;
//  3. the determinism check catches a dumper that is not reproducible run to run.
//
// (1)-(3) build the dumper at real revisions: a `git clone --shared` scratch copy of this repo gets a
// synthetic commit on top of HEAD (a temporary index + commit-tree — the real tree and the real
// repository are never written), and identity runs against it with the real checkout's fixtures.
// They compile decoder twice, so they are opt-in like the package's other real-tree tests.
// The rest are pure and always run.

// ---- pure ----

func TestIdentityAssets_coverTheManifest(t *testing.T) {
	root, _ := filepath.Abs("../..")
	fams, err := readManifestFamilies(root)
	if err != nil {
		t.Skipf("no parity manifest: %v", err)
	}
	listed := map[string]bool{}
	for _, a := range identityTiny {
		listed[a.Family] = true
		if _, ok := fams[a.Family]; !ok {
			t.Errorf("identityTiny lists %q, which is not a parity-manifest family", a.Family)
		}
		if a.Path == "" || a.Golden == "" {
			t.Errorf("identityTiny %s/%s: a tiny asset needs both a checkpoint and the golden holding its parity prompt", a.Family, a.Name)
		}
	}
	for f := range identityNoTiny {
		if listed[f] {
			t.Errorf("%q is in both identityTiny and identityNoTiny", f)
		}
		listed[f] = true
	}
	for f := range fams {
		if !listed[f] {
			t.Errorf("manifest family %q has no identity asset and no recorded reason: add it to identityTiny or identityNoTiny", f)
		}
	}
	for _, a := range identityReal {
		if _, ok := fams[a.Family]; !ok {
			t.Errorf("identityReal lists %q, which is not a parity-manifest family", a.Family)
		}
	}
}

func TestIdentityDumperTemplate_isGoAndHasItsMarkers(t *testing.T) {
	for _, m := range []string{"//REGISTER", "//HOOK"} {
		if n := strings.Count(identityDumperTmpl, m); n != 1 {
			t.Fatalf("template has %d %s markers, want 1", n, m)
		}
	}
	for _, reg := range []string{"", `_ "github.com/townsendmerino/goinfer/metal"`} {
		src := strings.Replace(identityDumperTmpl, "//REGISTER", reg, 1)
		if _, err := parser.ParseFile(token.NewFileSet(), "main.go", src, parser.AllErrors); err != nil {
			t.Fatalf("template (register %q) does not parse: %v", reg, err)
		}
	}
	// Public API only: the dumper must build at an older revision.
	for _, bad := range []string{"GOINFER_", "decoder.New", "ForTest"} {
		if strings.Contains(identityDumperTmpl, bad) {
			t.Errorf("template mentions %q — the dumper uses only the public decoder API and reads no env", bad)
		}
	}
}

func TestIdentityPathAllowed_refusesTheArchive(t *testing.T) {
	for _, p := range []string{"/Volumes/models/x.gguf", "/srv/models/qwen", "/srv/models"} {
		if identityPathAllowed(p) == nil {
			t.Errorf("%s was allowed; the archive is never a read path", p)
		}
	}
	if err := identityPathAllowed(filepath.Join(os.TempDir(), "x")); err != nil {
		t.Errorf("a local path was refused: %v", err)
	}
}

// synthDump builds a dump of `prompts` prompts × `steps` steps over vocab v, logit i of step s of
// prompt p = f(p,s,i).
func synthDump(prompts, steps, v int, f func(p, s, i int) float32) *dump {
	m := &dumpMeta{OK: true, Vocab: v, DecodePath: "cpu (f32)"}
	var b bytes.Buffer
	for p := 0; p < prompts; p++ {
		m.Prompts = append(m.Prompts, struct {
			Len    int   `json:"len"`
			Steps  int   `json:"steps"`
			Tokens []int `json:"tokens"`
		}{Len: 8, Steps: steps, Tokens: make([]int, steps)})
		for s := 0; s < steps; s++ {
			for i := 0; i < v; i++ {
				_ = binary.Write(&b, binary.LittleEndian, math.Float32bits(f(p, s, i)))
			}
		}
	}
	return &dump{Meta: m, Bin: b.Bytes()}
}

func base(p, s, i int) float32 { return float32(i%7) - float32(s)*0.25 + float32(p) }

func TestCompareDumps_locatesTheFirstDifference(t *testing.T) {
	a := synthDump(2, 4, 16, base)
	if d := compareDumps(a, synthDump(2, 4, 16, base)); !d.Identical {
		t.Fatalf("identical dumps compared different: %+v", d)
	}
	// One ulp, prompt 1, step 2, logit 5 — the smallest change there is.
	ulp := func(p, s, i int) float32 {
		x := base(p, s, i)
		if p == 1 && s == 2 && i == 5 {
			return math.Nextafter32(x, float32(math.Inf(1)))
		}
		return x
	}
	d := compareDumps(a, synthDump(2, 4, 16, ulp))
	if d.Identical || !d.Located || d.FirstPrompt != 1 || d.FirstStep != 2 || d.FirstLogit != 5 {
		t.Fatalf("1-ulp change not located at prompt 1 step 2 logit 5: %+v", d)
	}
	if d.MaxAbs <= 0 || d.MaxAbs > 1e-6 {
		t.Fatalf("max |diff| %g, want one ulp of ~6", d.MaxAbs)
	}
	if d.ArgmaxAgree != d.ArgmaxTotal || d.ArgmaxTotal != 8 {
		t.Fatalf("argmax %d/%d, want 8/8", d.ArgmaxAgree, d.ArgmaxTotal)
	}
	// An argmax flip, and a NaN.
	flip := func(p, s, i int) float32 {
		if p == 0 && s == 3 && i == 0 {
			return 100
		}
		if p == 1 && s == 0 && i == 1 {
			return float32(math.NaN())
		}
		return base(p, s, i)
	}
	d = compareDumps(a, synthDump(2, 4, 16, flip))
	if d.FirstArgmaxPrompt != 0 || d.FirstArgmaxStep != 3 || !math.IsInf(d.MaxAbs, 1) {
		t.Fatalf("argmax flip at prompt 0 step 3 and a NaN not reported: %+v", d)
	}
	// Fewer steps captured (an EOS came earlier) is structural, not silently truncated.
	d = compareDumps(a, synthDump(2, 3, 16, base))
	if d.Identical || !strings.Contains(d.Why, "captured steps differ") {
		t.Fatalf("a step-count difference was not reported: %+v", d)
	}
}

func TestJudgeCell_verdicts(t *testing.T) {
	c := &identityCell{ID: "x", Family: "f", Asset: "a", Quant: "f32"}
	good := synthDump(1, 2, 8, base)
	noisy := func(ulps int) *dump {
		return synthDump(1, 2, 8, func(p, s, i int) float32 {
			x := base(p, s, i)
			for k := 0; k < ulps; k++ {
				x = math.Nextafter32(x, float32(math.Inf(1)))
			}
			return x
		})
	}
	failed := &dump{Meta: &dumpMeta{Err: "load: boom"}}
	cases := []struct {
		name          string
		backend       string
		n1, n2, old   *dump
		want          string
		deterministic bool
	}{
		{"identical", "cpu", good, good, good, vIdentical, true},
		{"different", "cpu", good, good, noisy(1), vDifferent, true},
		{"new fails, old runs", "cpu", failed, failed, good, vDifferent, false},
		{"both fail", "cpu", failed, failed, failed, vNotRun, false},
		// Nondeterministic: new runs differ by 2 ulp, so the tolerance is ~4 ulp.
		{"tolerance: inside", "metal", withPath(good, "metal-resident (int4)"), withPath(noisy(2), "metal-resident (int4)"), withPath(noisy(1), "metal-resident (int4)"), vTolerant, false},
		{"tolerance: outside", "metal", withPath(good, "metal-resident (int4)"), withPath(noisy(1), "metal-resident (int4)"), withPath(noisy(40), "metal-resident (int4)"), vDifferent, false},
		{"gpu fell back to cpu", "metal", good, good, good, vNotRun, false},
	}
	for _, tc := range cases {
		out := judgeCell(c, tc.backend, tc.n1, tc.n2, tc.old)
		if out.Verdict != tc.want {
			t.Errorf("%s: verdict %s (%s), want %s", tc.name, out.Verdict, out.Detail, tc.want)
		}
		if tc.want != vNotRun && out.Deterministic != tc.deterministic && out.DetChecked {
			t.Errorf("%s: deterministic=%v, want %v", tc.name, out.Deterministic, tc.deterministic)
		}
	}
}

func withPath(d *dump, p string) *dump {
	m := *d.Meta
	m.DecodePath = p
	return &dump{Meta: &m, Bin: d.Bin}
}

// ---- equivalence, at real revisions (opt-in) ----

type identityFixture struct {
	root  string // the real checkout (fixtures)
	clone string // scratch clone the revs live in
	head  string
}

func newIdentityFixture(t *testing.T) *identityFixture {
	t.Helper()
	needTools(t)
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("identity equivalence: set GOINFER_HEAVY_TESTS=1 (builds decoder at two revisions, ~20-60 s each)")
	}
	root, _ := filepath.Abs("../..")
	head, err := gitOut(root, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	clone := filepath.Join(t.TempDir(), "repo")
	if _, err := gitOut(root, "clone", "--quiet", "--shared", "--no-checkout", root, clone); err != nil {
		t.Fatal(err)
	}
	return &identityFixture{root: root, clone: clone, head: head}
}

// commit writes a commit on top of parent in the scratch clone whose only change is one replacement
// in rel, without a checkout: a temporary index and commit-tree. It fails if the replacement is a
// no-op (a vacuous mutation proves nothing).
func (fx *identityFixture) commit(t *testing.T, parent, rel, from, to, msg string) string {
	t.Helper()
	body, err := gitOut(fx.clone, "show", parent+":"+rel)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, from) {
		t.Fatalf("mutation target %q not in %s — the mutation would be vacuous", from, rel)
	}
	mutated := strings.Replace(body, from, to, 1) + "\n"
	run := func(stdin string, env []string, args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = fx.clone
		cmd.Env = append(os.Environ(), env...)
		cmd.Stdin = strings.NewReader(stdin)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	blob := run(mutated, nil, "hash-object", "-w", "--stdin")
	idx := []string{"GIT_INDEX_FILE=" + filepath.Join(t.TempDir(), "index"),
		"GIT_AUTHOR_NAME=gate", "GIT_AUTHOR_EMAIL=gate@example.invalid", "GIT_COMMITTER_NAME=gate", "GIT_COMMITTER_EMAIL=gate@example.invalid"}
	run("", idx, "read-tree", parent)
	run("", idx, "update-index", "--cacheinfo", "100644,"+blob+","+rel)
	tree := run("", idx, "write-tree")
	return run("", idx, "commit-tree", tree, "-p", parent, "-m", msg)
}

func (fx *identityFixture) run(t *testing.T, oldRev, newRev string, fams []string, dumper string) (int, string) {
	t.Helper()
	var out bytes.Buffer
	t0 := time.Now()
	rc := identityMain(identityOpts{
		Repo: fx.clone, FixtureRoot: fx.root, OldRev: oldRev, NewRev: newRev, Backend: "cpu",
		Families: fams, Assets: "tiny", Quants: []string{"f32", "int4"}, Steps: 4,
		LogDir: t.TempDir(), Progress: os.Stderr, DumperSource: dumper,
	}, &out)
	s := regexp.MustCompile(`\x1b\[[0-9;]*m`).ReplaceAllString(out.String(), "")
	fmt.Fprintf(os.Stderr, "identity %s..%s: rc=%d in %s\n", shortSHA(oldRev), shortSHA(newRev), rc, time.Since(t0).Round(time.Second))
	return rc, s
}

var identityEquivFamilies = []string{"cohere", "cohere2", "llama", "lfm2", "qwen3_moe"}

// familyVerdicts reads the per-family lines of the report.
func familyVerdicts(t *testing.T, out string) map[string]string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^  (IDENTICAL|WITHIN TOLERANCE|DIFFERENT|NOT RUN)\s+(\S+)\s`)
	got := map[string]string{}
	for _, m := range re.FindAllStringSubmatch(out, -1) {
		got[m[2]] = m[1]
	}
	return got
}

// (1) A comment-only change is identical for every family.
func TestIdentity_commentOnlyChangeIsIdentical(t *testing.T) {
	fx := newIdentityFixture(t)
	rev := fx.commit(t, fx.head, "decoder/model.go", "\t\tparallel := placement == NormParallel\n",
		"\t\t// gate identity equivalence (1): a comment, and nothing else.\n\t\tparallel := placement == NormParallel\n", "comment only")
	rc, out := fx.run(t, fx.head, rev, identityEquivFamilies, "")
	got := familyVerdicts(t, out)
	for _, f := range identityEquivFamilies {
		if got[f] != vIdentical {
			t.Errorf("%s: %q, want IDENTICAL", f, got[f])
		}
	}
	if rc != 0 {
		t.Errorf("rc=%d, want 0", rc)
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("report:\n%s", out)
	}
}

// (2) A 1-ulp nudge to the residual in the NormParallel block — which cohere and cohere2, and no
// other family, execute (registry.go) — is DIFFERENT for exactly those two.
func TestIdentity_oneULPChangeFlagsExactlyTheFamiliesThatExecuteIt(t *testing.T) {
	fx := newIdentityFixture(t)
	rev := fx.commit(t, fx.head, "decoder/model.go", "\t\t\taddResidual2(h, scr.sub, scr.sub2)\n",
		"\t\t\taddResidual2(h, scr.sub, scr.sub2)\n\t\t\th[0] = math.Nextafter32(h[0], float32(math.Inf(1))) // equivalence (2): one ulp\n", "one ulp in NormParallel")
	rc, out := fx.run(t, fx.head, rev, identityEquivFamilies, "")
	got := familyVerdicts(t, out)
	want := map[string]string{"cohere": vDifferent, "cohere2": vDifferent, "llama": vIdentical, "lfm2": vIdentical, "qwen3_moe": vIdentical}
	for f, w := range want {
		if got[f] != w {
			t.Errorf("%s: %q, want %s", f, got[f], w)
		}
	}
	if !strings.Contains(out, "goes to its reference gate") || !strings.Contains(out, "first differing logit") {
		t.Errorf("a DIFFERENT family must be sent to its reference gate with the first differing logit located")
	}
	if rc != 1 {
		t.Errorf("rc=%d, want 1", rc)
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("report:\n%s", out)
	}
}

// (3) A dumper that is not reproducible run to run is caught by the determinism check before any
// old-vs-new comparison is trusted: on the CPU that is TE6's kill criterion, and red.
func TestIdentity_determinismCatchesANondeterministicDumper(t *testing.T) {
	fx := newIdentityFixture(t)
	// The injected noise: XOR the low mantissa bits of one cohere logit with the clock, so no two
	// processes write the same bytes. Every other family's dumps are untouched.
	noisy := strings.Replace(identityDumperTmpl, "//HOOK",
		`if strings.HasPrefix(c.ID, "cohere__") {
				bits := binary.LittleEndian.Uint32(buf)
				binary.LittleEndian.PutUint32(buf, bits^(uint32(time.Now().UnixNano())&0xffff|1))
			}`, 1)
	rc, out := fx.run(t, fx.head, fx.head, identityEquivFamilies, noisy)
	if !regexp.MustCompile(`NONDETERMINISTIC\s+cohere__tiny__f32`).MatchString(out) {
		t.Errorf("the nondeterministic cohere cells were not reported NONDETERMINISTIC")
	}
	if regexp.MustCompile(`NONDETERMINISTIC\s+(cohere2|llama|lfm2|qwen3_moe)__`).MatchString(out) {
		t.Errorf("a deterministic family was reported NONDETERMINISTIC")
	}
	if !strings.Contains(out, "KILL CRITERION") {
		t.Errorf("CPU nondeterminism must be reported as TE6's kill criterion")
	}
	if got := familyVerdicts(t, out); got["cohere"] == vIdentical {
		t.Errorf("cohere reported IDENTICAL although its own runs differ")
	}
	if rc != 1 {
		t.Errorf("rc=%d, want 1", rc)
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("report:\n%s", out)
	}
}
