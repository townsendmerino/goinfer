package cuda

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// M-35: the shipped moe.ptx stopped being the audited 12.6.85 artifact — it was regenerated at
// 12.9.86 (610ce7f) — and SIX places went on asserting the old state: REGEN.md, kernels.go,
// backend.go, moe.cu, router_f32.cu and this package's FMA-lint exemption ("frozen audited MoE
// PTX"). Nothing compared the claim with the file, so the label outlived the artifact and every
// comment steering the next editor around the frozen file was steering around something that no
// longer existed.
//
// This reads the version out of the PTX itself and holds the DOCUMENTATION to it. It does not
// take a side on which version should ship — that decision needs the pinned toolchain and the
// Linux box. It only makes the two agree, in whichever direction they are next changed.
func TestMoEPTX_versionMatchesItsDocumentation(t *testing.T) {
	ptx, err := os.ReadFile("testdata/moe.ptx")
	if err != nil {
		t.Skipf("no moe.ptx: %v", err)
	}
	m := regexp.MustCompile(`(?m)^// Cuda compilation tools, release [\d.]+, V([\d.]+)`).FindSubmatch(ptx)
	if m == nil {
		t.Fatal("moe.ptx has no NVRTC version banner — the header format changed; update this pin")
	}
	shipped := string(m[1])

	// REGEN.md must NOT read as though the audited version is what ships, unless it is.
	regen, err := os.ReadFile("testdata/REGEN.md")
	if err != nil {
		t.Fatalf("read REGEN.md: %v", err)
	}
	const audited = "12.6.85"
	if shipped != audited && !strings.Contains(string(regen), shipped) {
		t.Errorf("moe.ptx ships NVRTC %s but REGEN.md never mentions that version — it still "+
			"documents %s as the shipped artifact. The label and the file have drifted (M-35)",
			shipped, audited)
	}

	// And the FMA-lint exemption's REASON must not call it frozen-and-audited while it is not.
	lint, err := os.ReadFile("kernel_fma_lint_test.go")
	if err != nil {
		t.Fatalf("read kernel_fma_lint_test.go: %v", err)
	}
	reason := fmaLintExempt["moe.cu"]
	if reason == "" {
		t.Fatal("moe.cu is no longer exempt — if the MACs became intrinsics and it joined " +
			"lintedKernels, M-35 is closed: delete this test and the note in REGEN.md")
	}
	if shipped != audited && strings.Contains(reason, "frozen audited") {
		t.Errorf("the FMA-lint exemption reason still says %q while moe.ptx ships NVRTC %s: the "+
			"exemption protects a file the audited toolchain did not build (M-35)", reason, shipped)
	}
	if !strings.Contains(string(lint), "M-35") {
		t.Error("kernel_fma_lint_test.go does not record why moe.cu is still exempt; the next " +
			"reader sees an exemption whose stated reason has expired")
	}
}
