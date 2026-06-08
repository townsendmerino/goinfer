//go:build gpu

package gpu

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// naiveMatmulBT is the ground-truth reference: dst[m,n] = Σ_k a[m,k]·b[n,k].
func naiveMatmulBT(a, b []float32, M, K, N int) []float32 {
	dst := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var s float32
			for k := range K {
				s += a[m*K+k] * b[n*K+k]
			}
			dst[m*N+n] = s
		}
	}
	return dst
}

func randSlice(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64() * 0.1)
	}
	return out
}

// newOrSkip builds a Context or skips the test (headless box / no GPU).
func newOrSkip(t *testing.T) *Context {
	t.Helper()
	c, err := New()
	if err != nil {
		t.Skipf("no GPU available: %v", err)
	}
	return c
}

// isSoftwareAdapter reports whether the context resolved to a CPU/software
// renderer (Mesa lavapipe/llvmpipe, SwiftShader, …). CI's headless runner has
// no real GPU and falls back to one of these; its limits are lower (can't bind
// a real model's 233 MB LM head) and its math is imprecise/slow, so
// hardware-sensitive tests must skip on it. AdapterType==CPU catches the Vulkan
// software path (lavapipe → what CI uses); the name match catches the OpenGL
// software path (llvmpipe reports AdapterTypeUnknown) and other backends.
func isSoftwareAdapter(c *Context) bool { return softwareAdapterInfo(c.adapter.GetInfo()) }

// softwareAdapterInfo is the pure detection (unit-tested without a GPU).
func softwareAdapterInfo(info wgpu.AdapterInfo) bool {
	if info.AdapterType == wgpu.AdapterTypeCPU {
		return true
	}
	name := strings.ToLower(info.Name + " " + info.DriverDescription)
	for _, s := range []string{"llvmpipe", "lavapipe", "softpipe", "swiftshader", "software"} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// TestSoftwareAdapterDetection pins the software-adapter heuristic (runs with no
// GPU). CI's lavapipe reports AdapterType==CPU; the OpenGL software path
// (llvmpipe) reports Unknown but a tell-tale name; a real GPU (this box's NVIDIA,
// via the GL backend, also reports Unknown) must NOT match.
func TestSoftwareAdapterDetection(t *testing.T) {
	cases := []struct {
		info wgpu.AdapterInfo
		want bool
	}{
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeCPU, Name: "llvmpipe (LLVM 17.0.6, 256 bits)"}, true}, // CI Vulkan/lavapipe
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeUnknown, Name: "llvmpipe (LLVM 17.0.6, 256 bits)"}, true},
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeUnknown, Name: "SwiftShader Device (LLVM)"}, true},
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeUnknown, Name: "NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2"}, false}, // this box, GL backend
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeDiscreteGPU, Name: "AMD Radeon RX 7900"}, false},
		{wgpu.AdapterInfo{AdapterType: wgpu.AdapterTypeIntegratedGPU, Name: "Intel(R) Arc(tm) Graphics"}, false},
	}
	for _, tc := range cases {
		if got := softwareAdapterInfo(tc.info); got != tc.want {
			t.Errorf("softwareAdapterInfo(type=%v %q) = %v, want %v", tc.info.AdapterType, tc.info.Name, got, tc.want)
		}
	}
}

// newOrSkipHW builds a Context or skips on no-adapter OR a software adapter —
// for hardware-sensitive tests: bit-exact parity gates (imprecise on software),
// perf/microbenches (meaningless on software), and full-model tests (their large
// buffers exceed software-adapter binding limits). Keeps the bit-exact gates
// running on real hardware while making CI robust to environment drift.
func newOrSkipHW(t *testing.T) *Context {
	t.Helper()
	c := newOrSkip(t)
	if isSoftwareAdapter(c) {
		info := c.adapter.GetInfo()
		c.Close()
		t.Skipf("software adapter (%s %q); hardware-dependent test", info.AdapterType, info.Name)
	}
	return c
}

// TestMatmulBT_matchesNaive: the GPU kernel must match the scalar
// reference within f32 reduction-order tolerance. Shapes deliberately
// include dims that are NOT multiples of the 16×16 workgroup, to prove
// the in-kernel bounds check (row/col ≥ M/N → early return) is correct.
func TestMatmulBT_matchesNaive(t *testing.T) {
	c := newOrSkip(t)
	defer c.Close()
	t.Logf("GPU backend: %s", c.Backend())

	cases := []struct {
		name    string
		M, K, N int
	}{
		{"tiny", 3, 5, 2},             // smaller than one workgroup
		{"one_wg", 16, 64, 16},        // exactly one workgroup
		{"ragged", 17, 65, 33},        // +1 past workgroup on every axis
		{"L80_outproj", 80, 768, 768}, // forward shape
		{"L80_fc11", 80, 768, 3072},   // forward shape
		{"L80_fc2", 80, 3072, 768},    // forward shape (wide K)
		{"L256_fc11", 256, 768, 3072}, // batch-ish shape
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := randSlice(rng, tc.M*tc.K)
			b := randSlice(rng, tc.N*tc.K)
			ref := naiveMatmulBT(a, b, tc.M, tc.K, tc.N)

			got, err := c.MatmulBT(a, b, tc.M, tc.K, tc.N)
			if err != nil {
				t.Fatalf("MatmulBT: %v", err)
			}
			if len(got) != len(ref) {
				t.Fatalf("len(got)=%d want %d", len(got), len(ref))
			}
			// f32 accumulation in a different order than the CPU loop;
			// tolerance scales with K (reduction length).
			tol := float32(1e-4)*float32(tc.K)/768.0 + 1e-5
			for i := range ref {
				if d := float32(math.Abs(float64(got[i] - ref[i]))); d > tol {
					t.Fatalf("idx %d (m=%d,n=%d): got %v want %v (diff %v, tol %v)",
						i, i/tc.N, i%tc.N, got[i], ref[i], d, tol)
				}
			}
		})
	}
}

// TestMatmulBT_inputValidation: bad dims/short inputs return errors, not
// panics or GPU faults.
func TestMatmulBT_inputValidation(t *testing.T) {
	c := newOrSkip(t)
	defer c.Close()

	if _, err := c.MatmulBT([]float32{1, 2}, []float32{1, 2}, 0, 2, 1); err == nil {
		t.Error("expected error for M=0")
	}
	if _, err := c.MatmulBT(make([]float32, 3), make([]float32, 8), 4, 4, 2); err == nil {
		t.Error("expected error for short a (len 3, need 16)")
	}
}

// BenchmarkMatmulBT_GPU measures the full offload (upload + dispatch +
// readback) at forward-pass shapes. Compare the reported GFLOP/s
// (MB/s column ÷ 1000, since SetBytes = 2*M*K*N FLOPs) against the
// encoder package's BenchmarkMatmul{Serial,Parallel}_* CPU numbers.
// EXPECT the GPU to trail the CPU here at these shapes — per-call
// transfer dominates; this benchmark exists to quantify that gap and
// track it as the resident-buffer follow-up lands.
func benchGPU(b *testing.B, M, K, N int) {
	c, err := New()
	if err != nil {
		b.Skipf("no GPU available: %v", err)
	}
	defer c.Close()
	rng := rand.New(rand.NewPCG(1, 2))
	a := randSlice(rng, M*K)
	w := randSlice(rng, N*K)
	// Warm up (first dispatch compiles/caches the Metal pipeline state).
	if _, err := c.MatmulBT(a, w, M, K, N); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.SetBytes(int64(2 * M * K * N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.MatmulBT(a, w, M, K, N); err != nil {
			b.Fatalf("MatmulBT: %v", err)
		}
	}
}

func BenchmarkMatmulBT_GPU_L80_fc11(b *testing.B)  { benchGPU(b, 80, 768, 3072) }
func BenchmarkMatmulBT_GPU_L256_fc11(b *testing.B) { benchGPU(b, 256, 768, 3072) }
func BenchmarkMatmulBT_GPU_L512_fc11(b *testing.B) { benchGPU(b, 512, 768, 3072) }
func BenchmarkMatmulBT_GPU_L512_fc2(b *testing.B)  { benchGPU(b, 512, 3072, 768) }
