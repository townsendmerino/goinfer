package decoder

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestResidentMoECap_pinsKernelConstant gates M-17: the feature-matrix MoE caps in
// residentBackendMoECap MUST equal the router kernel's actual fixed-array bounds, or the
// hardware-matrix generator declines archs BuildResident admits (or vice versa) — the
// one-source-of-truth invariant the map's own comment promises. This reads the kernel sources
// and compares. The cuda groups cap was 32 while cuda/moe.cu's MOE_MAX_G is 64, so an arch with
// n_group in 33..64 was published not-CUDA-resident while the runtime admitted it. The test
// extracts each constant from its source and pins the cap to it; a future kernel bump that
// forgets the map (or vice versa) fails here.
func TestResidentMoECap_pinsKernelConstant(t *testing.T) {
	grep := func(path, pat string) int {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Skipf("kernel source %s unavailable: %v", path, err)
		}
		m := regexp.MustCompile(pat).FindSubmatch(b)
		if m == nil {
			t.Fatalf("%s: pattern %q not found (kernel refactored? update this pin)", path, pat)
		}
		n, _ := strconv.Atoi(string(m[1]))
		return n
	}
	// cuda: #define MOE_MAX_E 256 / MOE_MAX_G 64
	cudaE := grep("../cuda/moe.cu", `#define MOE_MAX_E (\d+)`)
	cudaG := grep("../cuda/moe.cu", `#define MOE_MAX_G (\d+)`)
	if got := residentBackendMoECap["cuda"]; got.experts != cudaE || got.groups != cudaG {
		t.Errorf("cuda cap %+v != kernel (experts %d, groups %d) — one-source-of-truth broken (M-17)", got, cudaE, cudaG)
	}
	// webgpu: array<f32, 256> score / array<f32, 32> gscore
	wE := grep("../gpu/moe.go", `var score: array<f32, (\d+)>`)
	wG := grep("../gpu/moe.go", `var gscore: array<f32, (\d+)>`)
	if got := residentBackendMoECap["webgpu"]; got.experts != wE || got.groups != wG {
		t.Errorf("webgpu cap %+v != kernel (experts %d, groups %d) (M-17)", got, wE, wG)
	}
}

// M-31: the pin above reads the KERNEL sources and features_test.go asserts ResidentEligible.
// Both were green while gpu/residency.go carried its own hardcoded 256/32 — so a 384-expert
// Kimi-K2 was published "✅ resident" in both generated matrices, admitted by ResidentEligible,
// and then declined to CPU by a line naming a number nothing else agreed with. Neither existing
// test reads the file that actually makes the decision.
//
// So this one does. It asserts the RUNTIME decline site does not restate a literal cap — the
// numbers must come from ResidentBackendMoECap — because a literal there is precisely the drift
// the map exists to prevent, and it is invisible to a test that only greps the kernel.
func TestResidentMoECap_runtimeDeclineReadsTheDeclaration(t *testing.T) {
	b, err := os.ReadFile("../gpu/residency.go")
	if err != nil {
		t.Skipf("gpu/residency.go unavailable: %v", err)
	}
	src := string(b)
	if !regexp.MustCompile(`ResidentBackendMoECap\(`).MatchString(src) {
		t.Error("gpu/residency.go does not call decoder.ResidentBackendMoECap: its MoE decline " +
			"restates the cap instead of reading it, which is how residency.go and the feature " +
			"map disagreed about 256 vs 512 (M-31)")
	}
	// And no literal cap comparison survives. Comment lines are skipped — the explanation of
	// the rule is not an instance of it, a trap two earlier guards in this audit fell into.
	lit := regexp.MustCompile(`nE\s*>\s*\d+|nGroup\s*>\s*\d+`)
	for i, ln := range strings.Split(src, "\n") {
		if strings.HasPrefix(strings.TrimSpace(ln), "//") {
			continue
		}
		if lit.MatchString(ln) {
			t.Errorf("gpu/residency.go:%d compares the expert/group count against a LITERAL: %s\n"+
				"the cap belongs to residentBackendMoECap, which the kernel pin above keeps "+
				"honest; a literal here is invisible to both (M-31)", i+1, strings.TrimSpace(ln))
		}
	}
}

// The declared webgpu cap must actually admit the models the published matrices call resident.
// Without this, raising the kernel and the map together while some third place still says 256
// would go unnoticed — which is the shape of M-31 itself.
func TestResidentMoECap_admitsThePublishedModels(t *testing.T) {
	e, g, ok := ResidentBackendMoECap("webgpu")
	if !ok {
		t.Fatal("webgpu has no declared MoE cap")
	}
	// Kimi-K2 and DeepSeek-V4-Pro: 384 experts, both listed "✅ resident" for webgpu.
	if e < 384 {
		t.Errorf("webgpu expert cap %d < 384: docs/hardware-matrix.md and "+
			"docs/capability-matrix.md publish Kimi-K2 / DeepSeek-V4-Pro as resident", e)
	}
	if g < 32 {
		t.Errorf("webgpu group cap %d < 32", g)
	}
}
