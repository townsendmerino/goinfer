//go:build darwin

package metal

import (
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_depthSweep turns "int4 conditioning at depth" from an argument into a MEASUREMENT
// (standing caution: this repo attributed a quality deficit to int4 twice — 625303e, bcadd44 — and
// overturned it both times). Same scaled-dense geometry (hd 256/512, K=V globals) at 12/24/48/64
// layers, only depth varies. Part A: the floor-vs-depth curve (CPUint4-vs-f32 mean) + the Metal
// envelope (Metal-vs-CPUint4 mean). Part B: the FULL 64-layer per-layer trace, to see whether the
// smooth-then-steep collapse the 26B showed (0.97→0.73 over 11 layers, then 0.47 over 3) is the
// fixture's own near-floor behaviour (explained) or a discontinuity the fixture doesn't reproduce.
// Env-gated (loads GB-scale f32 checkpoints). GOINFER_DEPTH_SWEEP=1.
func TestGemma4_depthSweep(t *testing.T) {
	if os.Getenv("GOINFER_DEPTH_SWEEP") == "" {
		t.Skip("set GOINFER_DEPTH_SWEEP=1 (loads 12/24/48/64-layer f32 checkpoints)")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200}
	cos := func(a, b []float32) float64 { c, _ := cosMaxAbs(a, b); return c }

	dirFor := func(d int) string {
		if d == 12 {
			return "../testdata/gemma4-dense-scaled"
		}
		return fmt.Sprintf("../testdata/gemma4-dense-scaled-%d", d)
	}

	// --- Part A: floor-vs-depth + Metal envelope (final logits) ---
	t.Logf("=== Part A: floor-vs-depth (16-token prompt, mean over positions) ===")
	t.Logf("depth  CPUint4-vs-f32(floor)  Metal-vs-CPUint4(env)  envelope-holds")
	for _, D := range []int{12, 24, 48, 64} {
		dir := dirFor(D)
		if _, err := os.Stat(dir); err != nil {
			t.Logf("  d=%-3d SKIP (no fixture %s)", D, dir)
			continue
		}
		floor, env := meanFloorEnv(t, dir, prompt, cos)
		hold := "YES"
		if env < floor {
			hold = "NO <<<"
		}
		t.Logf("  d=%-3d %.4f                 %.4f                 %s", D, floor, env, hold)
	}

	// --- Part B: full 64-layer per-layer trace (Metal-int4 vs CPU-int4, pos 3) ---
	dir := dirFor(64)
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no 64-layer fixture")
	}
	t.Logf("=== Part B: 64-layer fixture per-layer Metal-vs-CPUint4 at pos 3 (vs the 26B's shape) ===")
	perLayer64(t, dir)
}

// meanFloorEnv loads a fixture int4 (Metal + CPU) and f32 (CPU), runs the prompt, returns
// mean(CPUint4-vs-f32) [the int4 floor] and mean(Metal-vs-CPUint4) [the envelope].
func meanFloorEnv(t *testing.T, dir string, prompt []int, cos func(a, b []float32) float64) (floor, env float64) {
	t.Helper()
	mg, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4 %s: %v", dir, err)
	}
	defer mg.Close()
	r, err := BuildResident(mg)
	if err != nil {
		t.Fatalf("BuildResident %s: %v", dir, err)
	}
	defer r.Close()
	mc4, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu int4: %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(dir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load cpu f32: %v", err)
	}
	defer mcF.Close()
	c4c, cFc := mc4.NewCache(len(prompt)), mcF.NewCache(len(prompt))
	var sf, se float64
	for i, tok := range prompt {
		l4, _ := mc4.ForwardForTest(tok, c4c)
		lF, _ := mcF.ForwardForTest(tok, cFc)
		lM := r.ForwardEmb(mg.EmbedResidentForTest(tok), i)
		sf += cos(lF, l4)
		se += cos(l4, lM)
	}
	return sf / float64(len(prompt)), se / float64(len(prompt))
}

// perLayer64 captures the 64-layer fixture's per-layer hidden (Metal-int4 vs CPU-int4) at pos 3 and
// logs the trace, so its shape can be compared to the 26B's (0.97→0.73→0.26).
func perLayer64(t *testing.T, dir string) {
	t.Helper()
	mg, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mg.Close()
	r, err := BuildResident(mg)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	prompt := []int{1, 7, 42, 100}
	nL1 := r.nL + 1
	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	cc := mg.NewCache(len(prompt))
	for _, tk := range prompt {
		if _, err := mg.ForwardForTest(tk, cc); err != nil {
			t.Fatalf("cpu int4: %v", err)
		}
	}
	cpuAll := decoder.Gemma4HiddenCaptureForTest()
	cpu := cpuAll[(len(prompt)-1)*nL1:]

	metal := func() [][]float32 {
		var capLast [][]float32
		for pos, tk := range prompt {
			copy(r.x.Floats(), mg.EmbedResidentForTest(tk))
			c := r.forwardPagedCaptureForTest(pos)
			if pos == len(prompt)-1 {
				capLast = c
			}
		}
		return capLast
	}()
	prev, firstBigStep := 1.0, -1
	for l := 0; l < r.nL; l++ {
		c, _ := cosMaxAbs(cpu[l+1], metal[l])
		step := ""
		if c < prev-0.10 && firstBigStep < 0 {
			firstBigStep = l
			step = "  <<< first >0.10 step"
		}
		if l%4 == 0 || c < 0.5 || step != "" {
			t.Logf("  L%02d Metal-vs-CPUint4 %.4f%s", l, c, step)
		}
		prev = c
	}
	t.Logf("64-layer fixture (random weights, WORSE-conditioned than trained 26B): first >0.10 step at L%d "+
		"(26B trace: 0.97@L01 → 0.73@L11 → 0.26@L14, worst 0.244)", firstBigStep)
}
