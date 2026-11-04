package gpudev

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Kernel .cu sources live in this package dir and in anncuda/ — the same two dirs
// build_ptx.sh walks. Discovery is by glob, never by a hand-maintained list; see
// TestKernelFMALint's doc for why.
var kernelDirs = []string{".", "anncuda"}

// The two declarations a kernel source may carry. A file opts ITSELF into the lint;
// nothing outside the file decides. Exactly one is required of any .cu whose PTX ships
// (TestKernelFMALint_coversEmbeddedPTX).
const (
	markContract = "BIT-IDENTITY-CONTRACT"
	markExempt   = "BIT-IDENTITY-EXEMPT"
)

// A DECLARATION is a comment line that BEGINS with the marker and a colon. A mention of
// the token anywhere else — in prose, in a cross-reference, in a "flip this to X if …"
// note — is not a declaration.
//
// This distinction is not pedantic: a plain substring test classified vit.cu as
// CONTRACTED on the strength of the sentence "flip this to BIT-IDENTITY-CONTRACT and
// expect real work" inside its own EXEMPT block, and promptly reported 13 violations in
// a kernel that is deliberately uncontracted. Anchoring the match is what lets the
// declarations be explained in the file they govern rather than merely asserted.
var (
	reContract = regexp.MustCompile(`(?m)^\s*//\s*` + markContract + `:`)
	reExempt   = regexp.MustCompile(`(?m)^\s*//\s*` + markExempt + `:`)
)

// discoverKernels returns every .cu under kernelDirs, sorted, as repo-relative paths.
func discoverKernels(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, d := range kernelDirs {
		m, err := filepath.Glob(filepath.Join(d, "*.cu"))
		if err != nil {
			t.Fatalf("glob %s: %v", d, err)
		}
		out = append(out, m...)
	}
	sort.Strings(out)
	if len(out) == 0 {
		t.Fatal("no .cu sources found — this lint is no longer measuring anything")
	}
	return out
}

// TestKernelFMALint enforces the bit-identity rule at BUILD TIME: no bare float
// multiply-accumulate in any kernel that declares a bit-identity contract.
//
// WHY. A bare MAC (`x += a*b`, `= a*b + c`) lets the compiler CHOOSE fma vs mul+add.
// Separately-compiled kernels sharing a contract then land ~1 ULP apart,
// DATA-DEPENDENTLY. That is not hypothetical here: gemv_w4a8_fwd/gemv_w8a8_fwd
// compiled to mul+add (11 fma + 3 mul) while goinfer's batched gemv_w4a8_rn compiled
// to fma (192 fma + 0 mul), and the result was an 84% token-stream divergence on real
// weights — invisible on uniform random fixtures, which is why aikit's own
// tolerance-based gemv parity test did not catch it. Fixed in be049df by making every
// MAC an explicit intrinsic; this lint is what keeps it so.
//
// This is the PRIMARY check because it removes the compiler's discretion at the source,
// needs no NVRTC, no GPU and no rebuild, and therefore runs in every job — unlike the
// PTX histogram, which detects only what one toolchain version happened to choose.
//
// OPT-IN IS BY DECLARATION, NOT BY LIST. goinfer's equivalent keeps an explicit
// `lintedKernels` slice, and that is exactly how router_f32.cu — production PTX on the
// decode path — shipped unguarded from the day it was added (their audit C-16): a kernel
// was not covered by being contracted, only by someone remembering to list it. Here a
// file opts itself in with a BIT-IDENTITY-CONTRACT line in its own header, so the
// contract travels with the code it constrains and cannot be forgotten in a second
// place. TestKernelFMALint_coversEmbeddedPTX closes the remaining gap — forgetting to
// declare anything at all.
//
// The regexes are ported from goinfer's lint, including two scars worth keeping:
// stripping only the `for (…)` HEADER rather than the whole line (a single-line
// `for (…) acc += a[k]*b[k];` carries a real MAC — their audit R-04), and treating any
// (long)/(int)/(unsigned) cast as integer arithmetic.
func TestKernelFMALint(t *testing.T) {
	// Float MAC forms, with the multiply as ` * ` (spaces both sides, so a pointer
	// declaration `float* p` is not matched):
	//   accumulate:  ident += <expr> * <expr>
	//   expression:  … <expr> * <expr> [+-] <expr>
	macAccum := regexp.MustCompile(`\b\w[\w.]*\s*\+=\s*[^;/]* \* `)
	macExpr := regexp.MustCompile(` \* [^;/]* [+\-] `)
	compliant := regexp.MustCompile(`__fmaf_rn|__fmul_rn|__fadd_rn|__fma_rn|__dp4a|__shfl`)
	arrayIndex := regexp.MustCompile(`\[[^\]]*\]`)
	isDeclOrIndex := regexp.MustCompile(`^\s*(const\s+)?(unsigned\s+)?(int|long|short|char|size_t|void|float\*|double\*|__half\*|const int\*|const unsigned)\b|` +
		`^\s*(const\s+)?[\w]+\s*\*+\s*\w+\s*=|` +
		`\(long\)|\(int\)|\(unsigned`)
	forHeader := regexp.MustCompile(`\bfor\s*\([^)]*\)`)

	var contracted, exempt []string
	violations := 0
	for _, fn := range discoverKernels(t) {
		b, err := os.ReadFile(fn)
		if err != nil {
			t.Fatalf("read %s: %v", fn, err)
		}
		src := string(b)
		switch {
		case reContract.MatchString(src):
			contracted = append(contracted, fn)
		case reExempt.MatchString(src):
			exempt = append(exempt, fn)
			continue
		default:
			// Undeclared files are not linted here; whether that is allowed is
			// TestKernelFMALint_coversEmbeddedPTX's call, since it depends on
			// whether the kernel's PTX actually ships.
			continue
		}

		for i, line := range strings.Split(src, "\n") {
			code := line
			if k := strings.Index(code, "//"); k >= 0 {
				code = code[:k]
			}
			code = forHeader.ReplaceAllString(code, "")
			if strings.TrimSpace(code) == "" || compliant.MatchString(code) || isDeclOrIndex.MatchString(code) {
				continue
			}
			code = arrayIndex.ReplaceAllString(code, "[]")
			if !macAccum.MatchString(code) && !macExpr.MatchString(code) {
				continue
			}
			t.Errorf("%s:%d bare float MAC under a bit-identity contract — use __fmaf_rn / __fmul_rn / __fadd_rn: %s",
				fn, i+1, strings.TrimSpace(line))
			violations++
		}
	}
	if len(contracted) == 0 {
		t.Fatal("no .cu declares " + markContract + " — either the marker was renamed or the lint is vacuous")
	}
	if violations == 0 {
		t.Logf("every float MAC is an explicit intrinsic in %d contracted kernel(s): %v", len(contracted), contracted)
	}
	t.Logf("exempt (declared, not linted): %v", exempt)
}

// TestKernelFMALint_coversEmbeddedPTX is the gate on the gate.
//
// The lint above only sees files that declare something. A kernel author who declares
// NOTHING gets silence — which is the failure class goinfer hit with router_f32.cu, just
// reached by a different route. So: every .cu whose PTX is embedded into PRODUCTION code
// must carry a BIT-IDENTITY-CONTRACT or a BIT-IDENTITY-EXEMPT line. Exemption is fine;
// exemption WITHOUT SAYING SO is not, because it is indistinguishable from an oversight.
//
// The required set is derived from what the Go sources actually //go:embed, not from a
// list, so a new shipped kernel fails here the moment it is embedded — while the author
// still holds the context needed to decide which declaration it deserves.
//
// PTX embedded from a _test.go file is test-only scaffolding (smoke.ptx) and is reported
// but not required to declare: it ships to nobody.
func TestKernelFMALint_coversEmbeddedPTX(t *testing.T) {
	embedRe := regexp.MustCompile(`//go:embed\s+testdata/([\w]+)\.ptx`)
	type site struct{ goFile, cu string }
	var prod, testOnly []site

	for _, d := range kernelDirs {
		goFiles, err := filepath.Glob(filepath.Join(d, "*.go"))
		if err != nil {
			t.Fatalf("glob %s: %v", d, err)
		}
		for _, gf := range goFiles {
			b, err := os.ReadFile(gf)
			if err != nil {
				t.Fatalf("read %s: %v", gf, err)
			}
			for _, m := range embedRe.FindAllStringSubmatch(string(b), -1) {
				s := site{goFile: gf, cu: filepath.Join(d, m[1]+".cu")}
				if strings.HasSuffix(gf, "_test.go") {
					testOnly = append(testOnly, s)
				} else {
					prod = append(prod, s)
				}
			}
		}
	}
	if len(prod) == 0 {
		t.Fatal("no production //go:embed testdata/*.ptx found — this test is no longer measuring anything")
	}

	for _, s := range prod {
		b, err := os.ReadFile(s.cu)
		if err != nil {
			t.Errorf("%s embeds %s but its source is unreadable: %v", s.goFile, filepath.Base(s.cu), err)
			continue
		}
		src := string(b)
		switch {
		case reContract.MatchString(src):
			t.Logf("%-28s %s (linted)", s.cu, markContract)
		case reExempt.MatchString(src):
			t.Logf("%-28s %s", s.cu, markExempt)
		default:
			t.Errorf("%s ships as production PTX (embedded by %s) but declares neither %s nor %s — "+
				"a contracted kernel shipping without the FMA lint is exactly how goinfer's router_f32.cu "+
				"went unguarded; an exempt kernel with no stated reason is indistinguishable from an oversight",
				s.cu, s.goFile, markContract, markExempt)
		}
	}
	for _, s := range testOnly {
		t.Logf("%-28s test-only (embedded by %s) — declaration not required", s.cu, s.goFile)
	}
}
