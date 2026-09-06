package decoder

import (
	"strings"
	"testing"
)

// R3 gate (docs/measurements/cold-user-2026-09-06.md, scenario D). On v0.16.0 a 21 GB model on a
// 16 GB machine loaded without a word and drove the box +7,819 MB into swap in five seconds. The
// bar for this test is therefore not "an error is returned" — it is that NOTHING WAS ALLOCATED
// when the error was returned, and that the message names the flag that fixes it.
//
// The RAM figure is injected, so the test exercises the 16 GB machine's arithmetic on any box.
func TestFitGuard_refusesBeforeAllocating(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"

	// A machine far too small for even the tiny fixture, which prices at ~104 KB at int4: 128 KiB
	// of RAM is a 91 KiB budget. The 16 GB / 21 GB arithmetic from the actual failure is driven
	// directly in TestFitCheck_arithmeticMatchesTheMeasuredFailure, which needs no fixture at all.
	restore := injectHostRAM(t, 128<<10)
	defer restore()

	before := weightAllocs.Load()
	m, err := Load(gguf, Options{Quant: "int4"})
	if err == nil {
		if m != nil {
			m.Close()
		}
		t.Fatal("Load succeeded on a machine too small to hold the model — the guard did not fire")
	}
	if m != nil {
		t.Error("Load returned a non-nil model alongside the refusal")
	}
	if got := weightAllocs.Load() - before; got != 0 {
		t.Errorf("loadWeights was entered %d time(s) despite the refusal — the guard fires AFTER "+
			"the allocation, which is the swap storm it exists to prevent", got)
	}

	// The message has to carry the remedy and the arithmetic, because the user who reads it is
	// the one who has no idea -stream-weights exists — that was the whole finding.
	msg := err.Error()
	for _, want := range []string{"-stream-weights", "GB RAM", "budget", "GOINFER_NO_FIT_GUARD"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal does not mention %q:\n%s", want, msg)
		}
	}
	// And it must say what the flag DOES, or "-stream-weights" is just another name to guess at.
	if !strings.Contains(msg, ".giw") || !strings.Contains(msg, "pages weights") {
		t.Errorf("refusal names the flag without saying what it does:\n%s", msg)
	}
}

// The escape hatch has to work, or a machine the threshold is wrong about has no way forward.
// This also proves the refusal above came from the GUARD and not from something else about the
// fixture: same file, same injected RAM, only the variable differs.
func TestFitGuard_envOverrideLoads(t *testing.T) {
	restore := injectHostRAM(t, 128<<10)
	defer restore()
	t.Setenv("GOINFER_NO_FIT_GUARD", "1")

	m, err := Load("testdata/gptoss_tiny.gguf", Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("GOINFER_NO_FIT_GUARD=1 did not bypass the guard: %v", err)
	}
	m.Close()
}

// A machine with room must not be told no, and must not be warned either. The guard's failure
// mode is allowed to be letting a doomed load through; it is never allowed to be refusing one
// that would have run.
func TestFitGuard_amplyProvisionedMachineIsSilent(t *testing.T) {
	restore := injectHostRAM(t, 64<<30)
	defer restore()

	m, err := Load("testdata/gptoss_tiny.gguf", Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("guard refused a model on a 64 GB machine: %v", err)
	}
	m.Close()
}

// Unknown RAM must proceed. This is the branch that keeps Windows and the BSDs behaving exactly
// as they did before the guard existed (HostRAMBytes returns 0 there).
func TestFitGuard_unknownRAMProceeds(t *testing.T) {
	restore := injectHostRAM(t, 0)
	defer restore()

	f := fitCheckFor("testdata/gptoss_tiny.gguf", "int4", quantInt4, Options{})
	if f.known() {
		t.Fatal("a zero RAM figure reported itself as known")
	}
	if !f.fits() {
		t.Error("an unknown RAM figure produced a refusal — unknown must always proceed")
	}
	if err := guardFit(f); err != nil {
		t.Errorf("guardFit refused on unknown RAM: %v", err)
	}
}

// The arithmetic itself, driven with the numbers from the measurement rather than a 21 GB
// checkpoint: 16 GB machine, 21 GB of weights (cold run scenario D, attempt 1).
func TestFitCheck_arithmeticMatchesTheMeasuredFailure(t *testing.T) {
	f := fitCheck{
		name: "Qwen3.5-35B-A3B-Q4_K_M.gguf", quant: "int4",
		weightBytes: 21 << 30, ramBytes: 16 << 30,
	}
	if f.fits() {
		t.Fatal("21 GB of weights on a 16 GB machine reported as fitting — this is the case the guard exists for")
	}
	ram := int64(16) << 30
	if got, want := f.budget(), int64(float64(ram)*0.70); got != want {
		t.Errorf("budget = %d, want %d (70%% of physical)", got, want)
	}
	msg := f.declineErr().Error()
	for _, want := range []string{"21.0 GB", "16.0 GB", "11.2 GB"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal omits %q — a refusal without the numbers is the message the cold run already had:\n%s", want, msg)
		}
	}

	// And the warning band: a load at 80% of budget fits, but says so.
	tight := fitCheck{name: "x.gguf", quant: "int4", weightBytes: int64(0.80 * 0.70 * float64(ram)), ramBytes: ram}
	if !tight.fits() {
		t.Fatal("a load at 80% of budget was refused")
	}
	if tight.ratio() < fitWarnRatio {
		t.Fatalf("ratio %.2f is below the warn band %.2f — the banner would stay silent one run before the cliff", tight.ratio(), fitWarnRatio)
	}
	if !strings.Contains(tight.warning(), "-stream-weights") {
		t.Errorf("the tight-fit warning does not name the remedy:\n%s", tight.warning())
	}
}

// The estimator must not be free to drift from the accountant M-01 completed. It prices the model
// from GGUF metadata BEFORE the load; ResidentWeightBytes sums the matrices AFTER it. They answer
// the same question from opposite sides, so a large disagreement means one of them is wrong.
func TestFitEstimate_agreesWithResidentWeightBytes(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	for _, q := range []struct {
		name string
		mode quantMode
	}{{"int4", quantInt4}, {"int8int8", quantInt8I8}} {
		t.Run(q.name, func(t *testing.T) {
			est := estimateGGUFWeightBytes(gguf, q.mode)
			if est <= 0 {
				t.Fatal("estimator returned 0 for a real GGUF")
			}
			m, err := Load(gguf, Options{Quant: q.name})
			if err != nil {
				t.Skipf("cannot load fixture at %s: %v", q.name, err)
			}
			defer m.Close()
			actual := m.ResidentWeightBytes()
			if actual <= 0 {
				t.Skip("accountant reported 0 for this fixture")
			}
			ratio := float64(est) / float64(actual)
			// TIGHT on purpose, and it was not always. The band started at 0.6-1.6 and passed on
			// linux/amd64 at 0.96 while darwin/arm64 sat at 0.53 — the arm64 W4A8 row4 repack
			// keeps a second buffer, so int4 costs about twice its encoding there and a
			// hand-derived constant was ~1.8x low on the one platform the guard exists for.
			// quantBytesPerElem now MEASURES through quantizeWM, so a wide band would only hide
			// the next such divergence. Some slack remains because the estimator prices every
			// tensor uniformly while the accountant reads the real backing slices.
			if ratio < 0.85 || ratio > 1.25 {
				t.Errorf("estimate %d vs accounted %d (ratio %.2f) — the pre-load estimate has "+
					"drifted from ResidentWeightBytes", est, actual, ratio)
			}
			t.Logf("%s: estimate %d, accounted %d, ratio %.2f", q.name, est, actual, ratio)
		})
	}
}

// injectHostRAM replaces the machine's RAM figure for one test. Returned as a restore func rather
// than only t.Cleanup so the intent reads at the call site.
func injectHostRAM(t *testing.T, bytes int64) func() {
	t.Helper()
	prev := hostRAM
	hostRAM = func() int64 { return bytes }
	restore := func() { hostRAM = prev }
	t.Cleanup(restore)
	return restore
}

// Every quant mode must produce a plausible measured cost. The trap this pins: quantInt4Mix is a
// load-time POLICY, not a resident precision, so quantizeWM does not handle it and hands back the
// untouched f32 — pricing an int4mix model at 4 bytes/element, six times its real cost, and
// refusing models that fit comfortably. That is the one direction the guard must never fail in,
// and it is invisible in the ratio test above, which never exercises int4mix.
func TestQuantBytesPerElem_everyModeIsPlausible(t *testing.T) {
	for _, tc := range []struct {
		name     string
		mode     quantMode
		lo, hi   float64
		alsoLess quantMode // must be strictly cheaper than this mode, or the ordering is broken
	}{
		{name: "f32", mode: quantNone, lo: 4, hi: 4},
		{name: "int8", mode: quantInt8, lo: 0.9, hi: 2.4, alsoLess: quantNone},
		{name: "int8int8", mode: quantInt8I8, lo: 0.9, hi: 2.4, alsoLess: quantNone},
		// The upper bounds admit the repacked hosts: arm64's row4 and AVX2-without-VNNI's
		// split-half each keep a SECOND buffer beside the canonical nibbles, so int4 legitimately
		// costs about twice its 0.625-byte encoding there.
		{name: "int4", mode: quantInt4, lo: 0.55, hi: 1.5, alsoLess: quantInt8},
		{name: "int4mix", mode: quantInt4Mix, lo: 0.55, hi: 1.5, alsoLess: quantInt8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := quantBytesPerElem(tc.mode)
			if got < tc.lo || got > tc.hi {
				t.Errorf("%s costs %.4f bytes/elem, want %.2f..%.2f", tc.name, got, tc.lo, tc.hi)
			}
			if tc.alsoLess != tc.mode && got >= quantBytesPerElem(tc.alsoLess) {
				t.Errorf("%s (%.4f) is not cheaper than the wider mode (%.4f) — the modes are "+
					"mis-measured or mis-ordered", tc.name, got, quantBytesPerElem(tc.alsoLess))
			}
		})
	}
}
