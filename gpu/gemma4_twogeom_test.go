//go:build gpu && goinfer_testhooks

package gpu

import (
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

var twoGeomPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88}

const twoGeomDir = "../testdata/gemma4-dense-twogeom-tiny"

func cosMaxAbs(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		if i >= len(b) {
			break
		}
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), maxAbs
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func TestGemma4DenseTwoGeom_residentParity(t *testing.T) {
	if _, err := os.Stat(twoGeomDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", twoGeomDir)
	}
	mg, err := decoder.Load(twoGeomDir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (webgpu): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("webgpu resident DECLINED dense Gemma 4 — admission regressed")
	}

	mcpu, err := decoder.Load(twoGeomDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	cache := mcpu.NewCache(len(twoGeomPrompt))
	minCos := 1.0
	var maxMaxAbs float64
	exact := 0
	for i, tok := range twoGeomPrompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("webgpu pos %d: %v", i, err)
		}
		c, m := cosMaxAbs(cpuL, gpuL)
		if c < minCos {
			minCos = c
		}
		if m > maxMaxAbs {
			maxMaxAbs = m
		}
		if argmax(cpuL) == argmax(gpuL) {
			exact++
		}
		t.Logf("  pos %2d cosine %.6f maxAbs %.4e argmax cpu=%d webgpu=%d", i, c, m, argmax(cpuL), argmax(gpuL))
	}
	t.Logf("webgpu two-geometry K=V resident parity: minCosine=%.6f maxAbs=%.4e exact-argmax %d/%d", minCos, maxMaxAbs, exact, len(twoGeomPrompt))
	// S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. minCosine 0.977930 -> 1.000000.
	if minCos < 0.995 {
		t.Errorf("minCosine %.6f < 0.995 — the resident two-geometry/K=V forward diverges from CPU", minCos)
	}
}

func TestGemma4DenseScaled_webgpuParity(t *testing.T) {
	const dir = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (webgpu int4): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("webgpu resident DECLINED scaled dense Gemma 4 — admission regressed")
	}

	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()

	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200}
	cache := mc4.NewCache(len(prompt))
	minCos := 1.0
	c0 := 0.0
	exact := 0
	for i, tok := range prompt {
		cpuL, err := mc4.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("webgpu pos %d: %v", i, err)
		}
		c, m := cosMaxAbs(cpuL, gpuL)
		if i == 0 {
			c0 = c
		}
		if c < minCos {
			minCos = c
		}
		if argmax(cpuL) == argmax(gpuL) {
			exact++
		}
		t.Logf("  pos %2d  cosine %.6f maxAbs %.4e argmax cpu=%d webgpu=%d", i, c, m, argmax(cpuL), argmax(gpuL))
	}
	t.Logf("scaled dense (256-local / 512-global, 12 layers): minCosine=%.6f exact-argmax %d/%d pos0=%.6f", minCos, exact, len(prompt), c0)
	// S1.0 amendment 2026-10-07 (docs/tasks/task-multimodal-support-2026-10.md): the bar sits between the before-v_norm-fix and after readings; the fix is the mechanism. Never loosened. pos0 0.996182 -> 1.000000, minCosine 0.800099 -> 0.975300 (exact-argmax 15/16 -> 16/16 on nobara's RTX 2070 SUPER only; the Apple GPU reads 15/16 either way, so the argmax bar below stays at 15/16).
	if c0 < 0.999 {
		t.Errorf("pos-0 cosine %.6f < 0.999 — the 256-local/512-global resident geometry diverges at first token", c0)
	}
	if minCos < 0.9 {
		t.Errorf("minCosine %.6f < 0.9 — the scaled-dense resident forward diverges from CPU (v_norm on the sliding layers? S1.0)", minCos)
	}
	if exact < 15 {
		// 15 of 16 is the pre-S1.0 bar, restored 2026-10-07: the 16/16 this line asked for was one device's reading (RTX 2070 SUPER, Vulkan). The
		// Apple GPU's WebGPU reads 15/16 with the fix, and 15/16 without it, so an argmax count is a near-tie flip and not the discriminating metric.
		// pos0 and minCosine above carry the fix's signature: with v_norm re-dropped they fail on both devices.
		t.Errorf("exact-argmax %d/%d < 15/16", exact, len(prompt))
	}
}
