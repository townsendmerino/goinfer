//go:build linux

package qwencuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

const ckpt = "../../../internal/vision/testdata/qwen25vl-vision-tiny"

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
}

func maxAbs(a, b []float32) float64 {
	w := 0.0
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > w {
			w = d
		}
	}
	return w
}

// load returns a tower. quant=false is the Qwen parity configuration (the CPU gate
// runs fp32), so the resident path is exercised on f32 weights here.
func load(t *testing.T, quant bool) *vision.QwenVisionEncoder {
	t.Helper()
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no qwen25vl-vision-tiny checkpoint (%v); run scripts/oracle/pin_qwen25vl_vision.py", err)
	}
	e, err := vision.LoadQwenVisionEncoder(ckpt, quant)
	if err != nil {
		t.Skipf("LoadQwenVisionEncoder: %v", err)
	}
	return e
}

// synthPixels builds a deterministic pixel tensor for a grid.
func synthPixels(n, patchDim int) []float32 {
	px := make([]float32, n*patchDim)
	var s uint32 = 12345
	for i := range px {
		s = s*1664525 + 1013904223
		px[i] = float32(int32(s>>8)%2000-1000) / 1000.0
	}
	return px
}

// grid is the fixture's shape: h and w must be divisible by spatial_merge_size.
var grid = [][3]int{{1, 4, 4}}

func nPatches(g [][3]int) int {
	n := 0
	for _, x := range g {
		n += x[0] * x[1] * x[2]
	}
	return n
}

// TestQwenCUDA_parityWithCPU is the Phase-3 Qwen gate: the whole ViT tower on the GPU
// must reproduce the pure-Go tower's pre-merge hidden state, AND the merged image
// features that the CPU merger derives from it.
//
// Both stages are checked because they fail differently: the ViT is where the windowed
// attention, 2D rotary and window permutation live, while the merger amplifies any
// upstream error (the CPU fixture audit showed a 1e-6 ViT drift becoming 8.6e-03 after
// merging). A gate on the ViT alone would under-report.
func TestQwenCUDA_parityWithCPU(t *testing.T) {
	cpu := load(t, false)
	n := nPatches(grid)
	w := cpu.GPUWeights()
	px := synthPixels(n, w.PatchDim)

	wantViT, err := cpu.ForwardViT(px, grid)
	if err != nil {
		t.Fatalf("CPU ForwardViT: %v", err)
	}
	wantMerged, err := cpu.Forward(px, grid)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	g := load(t, false)
	if err := g.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v (no CUDA device?)", err)
	}
	defer g.Close()
	if !g.ResidentEnabled() {
		t.Fatal("ResidentEnabled false after EnableResident")
	}

	gotViT, err := g.ForwardViT(px, grid)
	if err != nil {
		t.Fatalf("GPU ForwardViT: %v", err)
	}
	if len(gotViT) != len(wantViT) {
		t.Fatalf("ViT: %d values, want %d", len(gotViT), len(wantViT))
	}
	cv := cosine(gotViT, wantViT)
	t.Logf("Qwen ViT GPU≡CPU: %d values, cosine %.9f, worst abs Δ %.3g", len(wantViT), cv, maxAbs(gotViT, wantViT))
	if cv < 1-1e-6 {
		t.Errorf("ViT cosine %.9f below 1-1e-6", cv)
	}

	gotMerged, err := g.Forward(px, grid)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	cm := cosine(gotMerged, wantMerged)
	t.Logf("Qwen merged GPU≡CPU: %d values, cosine %.9f, worst abs Δ %.3g", len(wantMerged), cm, maxAbs(gotMerged, wantMerged))
	if cm < 1-1e-6 {
		t.Errorf("merged cosine %.9f below 1-1e-6", cm)
	}

	// Break-it-first: a different input must not clear the bar, or "cosine ≈ 1" is
	// measuring the tower's output distribution rather than this forward.
	other := make([]float32, len(px))
	for i := range px {
		other[i] = -px[i]
	}
	alt, err := cpu.ForwardViT(other, grid)
	if err != nil {
		t.Fatalf("CPU ForwardViT (alt): %v", err)
	}
	if c := cosine(gotViT, alt); c >= 1-1e-6 {
		t.Errorf("break-it-first vacuous: a negated input still scored %.9f", c)
	} else {
		t.Logf("break-it-first: negated input scores cosine %.6f", c)
	}
}

// TestQwenCUDA_windowedVsFullBlocks guards the detail most likely to be silently
// wrong: the tower alternates windowed and full attention by block index, and the
// resident path uploads different segment bounds per block. If it used one kind
// throughout, this fixture's cosine would drop — so a passing parity test above
// already implies the switch works, and this asserts the config actually HAS both
// kinds, which is what makes that implication meaningful.
func TestQwenCUDA_fixtureExercisesBothAttentionKinds(t *testing.T) {
	e := load(t, false)
	w := e.GPUWeights()
	if len(w.FullattBlockIndexes) == 0 {
		t.Skip("fixture has no fullatt blocks — the windowed/full switch is untested by parity")
	}
	full, windowed := 0, 0
	for i := range w.NumLayers {
		if e.IsFullAtt(i) {
			full++
		} else {
			windowed++
		}
	}
	t.Logf("fixture blocks: %d full-attention, %d windowed", full, windowed)
	if full == 0 || windowed == 0 {
		t.Errorf("fixture exercises only one attention kind (full=%d windowed=%d) — parity cannot catch a switch bug", full, windowed)
	}
}

// TestQwenCUDA_repeatable checks scratch reuse across calls.
func TestQwenCUDA_repeatable(t *testing.T) {
	g := load(t, false)
	if err := g.EnableResident(); err != nil {
		t.Skipf("EnableResident: %v", err)
	}
	defer g.Close()
	w := g.GPUWeights()
	px := synthPixels(nPatches(grid), w.PatchDim)
	first, err := g.ForwardViT(px, grid)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}
	for i := range 3 {
		again, err := g.ForwardViT(px, grid)
		if err != nil {
			t.Fatalf("Forward %d: %v", i+2, err)
		}
		for j := range first {
			if again[j] != first[j] {
				t.Fatalf("call %d diverged at %d (scratch not reset?)", i+2, j)
			}
		}
	}
	t.Log("4 forwards on reused scratch: bit-identical")
}

// TestQwenCUDA_allocationFailureIsAnErrorAndTheEncoderRecovers forces a real device allocation failure in the scratch growth (a patch count whose buffers are terabytes,
// which cuMemAlloc refuses at once, so nothing is actually consumed) and checks the three things that were wrong: it comes back as an error and not a panic (gpu's
// NewBufferLenOf panics by contract, and an unrecovered panic on the request goroutine killed a serving process), no buffer made on the way is leaked, and the encoder still
// produces the bit-identical result afterwards (the old non-atomic growth released the working buffers first, so a failure left the encoder pointing at freed memory).
func TestQwenCUDA_allocationFailureIsAnErrorAndTheEncoderRecovers(t *testing.T) {
	src := load(t, false)
	e, err := newEncoder(src)
	if err != nil {
		t.Skipf("no CUDA device for the Qwen tower: %v", err)
	}
	defer e.Close()
	w := src.GPUWeights()
	px := synthPixels(nPatches(grid), w.PatchDim)
	first, err := e.ForwardViT(px, grid)
	if err != nil {
		t.Fatalf("forward before the failure: %v", err)
	}
	allocsBefore, _ := e.dev.LedgerLen()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("ensure panicked instead of returning the allocation failure: %v", r)
			}
		}()
		if err := e.ensure(1 << 31); err == nil {
			t.Fatal("ensure(2^31 patches) succeeded; the failure this test needs did not happen")
		} else {
			t.Logf("allocation failure came back as an error: %.140s", err)
		}
	}()

	again, err := e.ForwardViT(px, grid)
	if err != nil {
		t.Fatalf("forward after the failure: %v", err)
	}
	for i := range first {
		if again[i] != first[i] {
			t.Fatalf("the encoder disagrees with itself after a failed growth at %d: %v vs %v", i, again[i], first[i])
		}
	}
	if allocsAfter, _ := e.dev.LedgerLen(); allocsAfter != allocsBefore {
		t.Errorf("a failed growth leaked device allocations: %d before, %d after", allocsBefore, allocsAfter)
	}
	// And a growth that works still works after the failure.
	bigger := [][3]int{{1, 8, 8}}
	if _, err := e.ForwardViT(synthPixels(nPatches(bigger), w.PatchDim), bigger); err != nil {
		t.Fatalf("a larger image after the failure: %v", err)
	}
}
