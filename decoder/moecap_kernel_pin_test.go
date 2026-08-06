package decoder

import (
	"os"
	"regexp"
	"strconv"
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
