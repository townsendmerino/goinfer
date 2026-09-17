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
	if minCos < 0.97 {
		t.Errorf("minCosine %.6f < 0.97 — the resident two-geometry/K=V forward diverges from CPU", minCos)
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
	if c0 < 0.97 {
		t.Errorf("pos-0 cosine %.6f < 0.97 — the 256-local/512-global resident geometry diverges at first token", c0)
	}
	if exact < 15 {
		t.Errorf("exact-argmax %d/%d < 15/16", exact, len(prompt))
	}
}
