//go:build gpu

package gpu

// Test helpers copied across the gpu / gpu/internal/kernels line: each is the same code as its original, kept
// on both sides because tests on both sides call it.

import (
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"os"
	"sync/atomic"
	"testing"
	"time"
)

func argmaxF(v []float32) int {
	bi, bv := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bv {
			bi, bv = i, x
		}
	}
	return bi
}

func cosSim(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
		if d := math.Abs(float64(a[i] - b[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 1, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}

func cosine(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12), maxAbs
}

var gpuWasAvailable atomic.Bool

// isSoftwareAdapter reports whether the context resolved to a CPU/software
// renderer (Mesa lavapipe/llvmpipe, SwiftShader, …). CI's headless runner has
// no real GPU and falls back to one of these; its limits are lower (can't bind
// a real model's 233 MB LM head) and its math is imprecise/slow, so
// hardware-sensitive tests must skip on it. AdapterType==CPU catches the Vulkan
// software path (lavapipe → what CI uses); the name match catches the OpenGL
// software path (llvmpipe reports AdapterTypeUnknown) and other backends.
func isSoftwareAdapter(c *Context) bool {
	return kernels.SoftwareAdapterInfo((*kernels.ContextInfo(c)))
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// newOrSkip builds a Context or skips the test (headless box / no GPU).
func newOrSkip(t *testing.T) *Context {
	t.Helper()
	c, err := New()
	requireGPU(t, err) // skips if this machine has no GPU; FAILS if the process exhausted one
	return c
}

// newOrSkipHW builds a Context or skips on no-adapter OR a software adapter —
// for hardware-sensitive tests: bit-exact parity gates (imprecise on software),
// perf/microbenches (meaningless on software), and full-model tests (their large
// buffers exceed software-adapter binding limits). Keeps the bit-exact gates
// running on real hardware while making CI robust to environment drift.
func newOrSkipHW(t *testing.T) *Context {
	t.Helper()
	reportLive(t)
	c := newOrSkip(t)
	if isSoftwareAdapter(c) {
		info := (*kernels.ContextInfo(c))
		c.Close()
		t.Skipf("software adapter (%s %q); hardware-dependent test", info.AdapterType, info.Device)
	}
	return c
}

func readWords(c *Context, b *wgpu.Buffer, n int) []uint32 {
	f, _ := c.Readback(kernels.DeviceBufferLit(b, n))
	out := make([]uint32, n)
	for i, v := range f {
		out[i] = math.Float32bits(v)
	}
	return out
}

// reportLive logs the number of Contexts currently open, so a full-suite run shows WHICH test
// leaked one. The driver allows 63 live at once; past that every later test loses its GPU and
// skips, which is how TestWebGPU_forwardParity and TestResidentForwardN_parity stopped being
// gates without anyone noticing.
func reportLive(t *testing.T) {
	if n := kernels.LiveContexts.Load(); n > 0 {
		t.Logf("[live-contexts] %d already open before this test", n)
	}
	if b := LiveBufferBytes(); b > 8<<20 {
		t.Logf("[live-vram] %.1f MiB of device buffers already held before this test", float64(b)/(1<<20))
	}
}

// requireGPU converts a lost device into a loud failure, and a genuinely absent one into a
// skip. Call it wherever a test would otherwise skip on a device error.
func requireGPU(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		gpuWasAvailable.Store(true)
		return
	}
	if gpuWasAvailable.Load() {
		t.Fatalf("GPU was available earlier in this process and is now gone — the test binary has "+
			"exhausted it (resident weights not released; see gpu/modelw_release.go). This is a real "+
			"defect, not a missing GPU, so it fails instead of skipping: %v", err)
	}
	t.Skipf("no GPU available: %v", err)
}

// requireHeavyModel gates a test that loads a multi-GB checkpoint from ~/models behind an
// explicit env opt-in: the package-local twin of decoder/heavytest_test.go's helper, same
// GOINFER_HEAVY_TESTS key, so `GOINFER_HEAVY_TESTS=1 go test ./...` runs every backend's heavy
// tests uniformly. A model happening to be on disk is not a request to run a multi-GB test
// (deciding by path existence alone once fired the whole zoo opportunistically and thrashed the
// box); the per-test os.Stat skip stays as a second guard, so opting in on a bare box is still
// harmless.
//
// It takes testing.TB so the measurement harnesses can be Benchmarks: they report numbers and
// assert nothing, and as Test* a green suite would include a PASS that proves nothing about the
// win they map.
func requireHeavyModel(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}

func ubuf(c *Context, data []uint32) *wgpu.Buffer {
	b, _ := (*kernels.ContextDevice(c)).TryCreateBufferInit(&wgpu.BufferInitDescriptor{Label: "i8t-u", Contents: wgpu.ToBytes(data), Usage: wgpu.BufferUsageUniform})
	return b
}

func zbuf(c *Context, n int) *wgpu.Buffer { // zeroed storage, readable
	b, _ := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Label: "i8t-z", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	return b
}
