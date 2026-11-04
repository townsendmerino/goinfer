//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// Hardware-coverage H1.5 (docs/tasks/task-hardware-coverage-2026-10.md): the device-shape and memory-budget decisions are made from numbers our one card
// (RTX 2070 SUPER: 40 SMs, 8 GB) never takes other values of. These tests tell a real resident build that it has another card's SM count, or another amount of
// free VRAM, and run the existing correctness expectation (a greedy decode must not change) under each. Nothing new is asserted about the answer: a different SM
// count only changes launch grids (the kernels are bit-identical for any rows-per-warp), and a smaller budget only changes the context and slots a load picks.
//
// Each test also asserts it is not vacuous: that the forced configurations really reach DIFFERENT decisions, because a decode that passes under five forced
// shapes which all pick the same grid has exercised one path five times.

const forcedModelDir = "models/qwen2.5-0.5b-instruct"

// forcedDecode loads the 0.5B resident on CUDA at int4, decodes n greedy tokens from a fixed prompt, and returns the tokens, the resident (nil when the build
// declined to the CPU) and a close func. The prompt is arbitrary valid ids: only determinism matters.
func forcedDecode(t *testing.T, n int) (toks []int, r *cudaResident, decline string, closeFn func()) {
	t.Helper()
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, forcedModelDir)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no %s: this is a real-checkpoint gate", forcedModelDir)
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if rr, ok := m.ResidentForwardForTest().(*cudaResident); ok && rr != nil {
		r = rr
	} else {
		decline = m.ResidentDecline()
	}
	ch, gen := m.Generate(context.Background(), []int{9707, 11, 847, 829, 374, 1207}, n, decoder.SamplingParams{})
	for id := range ch {
		toks = append(toks, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return toks, r, decline, func() { m.Close() }
}

func sameTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestForcedSMShapes_decodeIsIdentical: the 0.5B decodes the same 32 greedy tokens whatever SM count, threads per SM and shared memory per SM the wave sizing is
// told, including shapes that make it pick different rows-per-warp for the fused gate/up projection (asserted, so the test cannot pass by running one grid five times).
func TestForcedSMShapes_decodeIsIdentical(t *testing.T) {
	base, r0, decl, closeBase := forcedDecode(t, 32)
	defer closeBase()
	if r0 == nil {
		t.Skipf("not CUDA-resident here: %s", decl)
	}
	t.Logf("real device shape: %d SMs, %d threads/SM, %d B shared/SM", r0.smCount, r0.smThreads, r0.smSmem)
	smemBlock := (r0.hidden + 256 + r0.hidden/4) * 4 // the fused gate/up block's dynamic shared memory (resident.go)
	rpw := map[int]string{}
	for _, sh := range []struct {
		name          string
		sms, thr, mem int
	}{
		{"20 SMs (a small card)", 20, 1024, 65536},
		{"46 SMs, Ampere-sized smem (3060)", 46, 1536, 102400},
		{"82 SMs (3080)", 82, 1536, 102400},
		{"128 SMs (4090-class)", 128, 1536, 102400},
		{"170 SMs, 228 KiB shared (Blackwell-class)", 170, 1536, 233472},
	} {
		restore := SetSMShapeForTest(sh.sms, sh.thr, sh.mem)
		toks, r, decl, closeFn := forcedDecode(t, 32)
		restore()
		if r == nil {
			closeFn()
			t.Fatalf("%s: the build declined to the CPU (%s), so the forced shape reached nothing", sh.name, decl)
		}
		if r.smCount != sh.sms || r.smThreads != sh.thr || r.smSmem != sh.mem {
			t.Errorf("%s: the resident saw %d/%d/%d, not the forced shape: the hook did not reach BuildResident", sh.name, r.smCount, r.smThreads, r.smSmem)
		}
		got := r.waveRowsPerWarp(2*r.inter, smemBlock)
		rpw[got] = sh.name
		closeFn()
		if !sameTokens(toks, base) {
			t.Errorf("%s (gate/up rows-per-warp %d): decoded %v, the real shape decoded %v", sh.name, got, toks, base)
		}
	}
	realRPW := r0.waveRowsPerWarp(2*r0.inter, smemBlock)
	rpw[realRPW] = "the real device"
	if len(rpw) < 2 {
		t.Fatalf("every shape, the real one included, picks rows-per-warp %d for the gate/up projection: this run exercised one grid, not the launch math across shapes", realRPW)
	}
	t.Logf("gate/up rows-per-warp picked across the shapes: %v", rpw)
}

// TestForcedFreeVRAM_residentDecisions: a load told it has less free VRAM than the card does picks a smaller context or declines to the CPU, and the model still
// answers. Budgets above the real card are not forced (a resident that then asked the card for memory it lacks would fail, which proves nothing about the
// decision; Plan's own tests cover those figures). The resident decisions (context, placement) must differ across the budgets, or the test says so.
func TestForcedFreeVRAM_residentDecisions(t *testing.T) {
	base, r0, decl, closeBase := forcedDecode(t, 16)
	defer closeBase()
	if r0 == nil {
		t.Skipf("not CUDA-resident here: %s", decl)
	}
	outcomes := map[string]int64{}
	outcomes[fmt.Sprintf("resident ctx %d", r0.ctxCap)]++
	for _, mib := range []int64{600, 900, 1200, 1800, 3000} {
		restore := decoder.SetMemoryProbeForTest("cuda", mib<<20, true)
		toks, r, decl, closeFn := forcedDecode(t, 16)
		restore()
		var outcome string
		if r != nil {
			outcome = fmt.Sprintf("resident ctx %d", r.ctxCap)
			if !sameTokens(toks, base) {
				t.Errorf("%d MiB free: resident at ctx %d decoded %v, the unforced run decoded %v", mib, r.ctxCap, toks, base)
			}
		} else {
			outcome = "declined to the CPU"
			if len(toks) == 0 {
				t.Errorf("%d MiB free: declined to the CPU (%s) and then produced no tokens", mib, decl)
			}
		}
		outcomes[outcome]++
		t.Logf("%5d MiB free -> %s", mib, outcome)
		closeFn()
	}
	if len(outcomes) < 2 {
		t.Fatalf("every budget, the real one included, reached the same decision %v: this run exercised one path", outcomes)
	}
}
