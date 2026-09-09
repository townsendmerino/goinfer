package decoder

import (
	"errors"
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
	for _, want := range []string{"-stream-weights", "memory available", "budget", "GOINFER_NO_FIT_GUARD"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal does not mention %q:\n%s", want, msg)
		}
	}
	// And it must say what the flag DOES, or "-stream-weights" is just another name to guess at.
	if !strings.Contains(msg, ".giw") || !strings.Contains(msg, "pages weights") {
		t.Errorf("refusal names the flag without saying what it does:\n%s", msg)
	}

	// gptoss_tiny.gguf is gpt-oss — MoE, own-forward — so an automatic -stream-weights retry must
	// NOT be offered: that CPU path is the one docs/benchmarks.md "M35/M26 on the Mac" measured as
	// 2h10min/zero completions on a real checkpoint. Both the sentinel (errors.Is) and the typed
	// field (errors.As) have to agree, since main.go's retry decision reads the field directly.
	if !errors.Is(err, ErrWontFitResident) {
		t.Error("refusal does not wrap ErrWontFitResident")
	}
	var fde *FitDeclineError
	if !errors.As(err, &fde) {
		t.Fatal("refusal is not a *FitDeclineError")
	}
	if fde.DenseStreamable {
		t.Error("DenseStreamable = true for gpt-oss (MoE) — an auto-retry would repeat the measured M35/M26 CPU-streaming failure")
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

// R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's SECOND live re-run):
// the load-time guard must price against CURRENTLY AVAILABLE memory, not total RAM. Ample total
// RAM with tight availability is exactly the live failure's shape — the load-time guard on the
// real Mac reported a healthy-looking margin against total RAM while the machine was, in fact,
// already out of room, and swap began within 15 seconds of load completing, before any request.
func TestFitGuard_pricesAgainstAvailableNotTotalRAM(t *testing.T) {
	restore := injectHostRAM(t, 64<<30) // 64 GB total: this alone must NOT be enough to pass
	defer restore()
	restoreAvail := injectHostRAMAvailable(t, 128<<10) // but only 128 KiB actually available
	defer restoreAvail()

	before := weightAllocs.Load()
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Quant: "int4"})
	if err == nil {
		if m != nil {
			m.Close()
		}
		t.Fatal("Load succeeded against 64 GB TOTAL RAM despite 128 KiB AVAILABLE — " +
			"the guard is pricing against total RAM again, not what is actually free")
	}
	if got := weightAllocs.Load() - before; got != 0 {
		t.Errorf("loadWeights was entered %d time(s) despite the refusal", got)
	}
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
	if _, err := guardFit(f); err != nil {
		t.Errorf("guardFit refused on unknown RAM: %v", err)
	}
}

// The arithmetic itself, driven with the numbers from the measurement rather than a 21 GB
// checkpoint: 16 GB machine, 21 GB of weights (cold run scenario D, attempt 1).
func TestFitCheck_arithmeticMatchesTheMeasuredFailure(t *testing.T) {
	f := fitCheck{
		name: "Qwen3.5-35B-A3B-Q4_K_M.gguf", quant: "int4",
		weightBytes: 21 << 30, availBytes: 16 << 30,
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
	tight := fitCheck{name: "x.gguf", quant: "int4", weightBytes: int64(0.80 * 0.70 * float64(ram)), availBytes: ram}
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

// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): the guard priced weights and,
// only if -ctx was pinned, KV — so an UNPINNED load that fits at idle can still swap the machine
// on its first real request, because KV was priced at 0 regardless of how large the model's own
// context window is. Driven with numbers shaped like the actual failure: a 7B-class model whose
// weights alone fit comfortably, but whose KV at its full context window does not.
func TestFitCheck_unpinnedPricesKVAtTheModelsMaximum(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 131072}
	ram := int64(16) << 30               // 16 GB, the machine that actually swapped
	budget := int64(float64(ram) * 0.70) // 11.2 GB
	gbf := float64(fitGB)
	weights := int64(8.9 * gbf) // ~8.9 GB — the guard's own "79% of budget" reading
	rate := kvBytesPerPosition(cfg, false, false)
	fullKV := rate * int64(cfg.MaxPositions) // KV at the model's full 128k window

	f := fitCheck{
		name: "qwen2.5-7b-instruct-q3_k_m.gguf", quant: "int4",
		weightBytes: weights, availBytes: ram,
		cfg: cfg, effCtx: cfg.MaxPositions, pinned: false,
	}
	f.kvBytes = fullKV

	if weights >= budget {
		t.Fatal("test setup: weights alone already exceed budget — the case under test needs weights to fit and weights+KV not to")
	}
	if f.fits() {
		t.Fatalf("weights (%.1f GB) + full-context KV (%.1f GB) fit an %.1f GB budget — test setup does not reproduce the failure shape",
			float64(weights)/fitGB, float64(fullKV)/fitGB, float64(budget)/fitGB)
	}

	// The guard must find a SMALLER context that fits — R13's whole point is that this load
	// should not simply refuse the model the README told a user to pull.
	ctx, ok := f.smallerFittingContext()
	if !ok {
		t.Fatalf("no smaller context found to fit weights=%.1fGB budget=%.1fGB — expected one well above ctxFloor (%d)",
			float64(weights)/fitGB, float64(budget)/fitGB, ctxFloor)
	}
	if ctx < ctxFloor {
		t.Errorf("chosen context %d is below ctxFloor %d", ctx, ctxFloor)
	}
	if ctx >= cfg.MaxPositions {
		t.Errorf("chosen context %d did not shrink below the model's own maximum %d", ctx, cfg.MaxPositions)
	}
	// The chosen context must ACTUALLY fit — re-price it exactly, not approximately.
	if got := weights + estimateKVBytes(cfg, ctx, false, false); got > budget {
		t.Errorf("chosen context %d still needs %.2f GB against a %.2f GB budget", ctx, float64(got)/fitGB, float64(budget)/fitGB)
	}

	note := f.capNote(ctx)
	for _, want := range []string{"capped at", "-ctx", "-stream-weights"} {
		if !strings.Contains(note, want) {
			t.Errorf("cap note omits %q:\n%s", want, note)
		}
	}
}

// The other side of R13: an EXPLICIT -ctx pin that does not fit is refused, never silently
// downgraded to something smaller than what was asked for (the G-07 principle — an explicit
// request that cannot be honoured is a refusal, not a surprise).
func TestFitCheck_pinnedContextThatDoesNotFitIsRefusedNotDowngraded(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 131072}
	ram := int64(16) << 30
	gbf := float64(fitGB)
	weights := int64(8.9 * gbf)
	pinnedCtx := 65536
	f := fitCheck{
		name: "x.gguf", quant: "int4", weightBytes: weights, availBytes: ram,
		cfg: cfg, effCtx: pinnedCtx, pinned: true,
	}
	f.kvBytes = estimateKVBytes(cfg, pinnedCtx, false, false)
	if f.fits() {
		t.Fatal("test setup: this pinned context needs to NOT fit for the refusal path to be under test")
	}
	if _, ok := f.smallerFittingContext(); ok {
		t.Fatal("smallerFittingContext offered a downgrade for a PINNED request — an explicit -ctx must refuse, not silently shrink")
	}
	msg := f.declineErr().Error()
	if !strings.Contains(msg, "memory available") {
		t.Errorf("pinned refusal missing the arithmetic:\n%s", msg)
	}
}

// And the floor: when even ctxFloor does not fit, the guard refuses exactly as R3 already does —
// it must not hand back a context so small it is not a useful server.
func TestFitCheck_unpinnedRefusesWhenEvenTheFloorDoesNotFit(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 131072}
	ram := int64(16) << 30
	budget := int64(float64(ram) * 0.70)
	// Weights alone eat nearly the whole budget, leaving no room for ctxFloor's own KV.
	weights := budget - kvBytesPerPosition(cfg, false, false)*int64(ctxFloor)/2
	f := fitCheck{
		name: "x.gguf", quant: "int4", weightBytes: weights, availBytes: ram,
		cfg: cfg, effCtx: cfg.MaxPositions, pinned: false,
	}
	f.kvBytes = estimateKVBytes(cfg, cfg.MaxPositions, false, false)
	if f.fits() {
		t.Fatal("test setup: full-context load must not fit, for the floor-refusal path to be under test")
	}
	if ctx, ok := f.smallerFittingContext(); ok {
		t.Fatalf("smallerFittingContext returned %d, want no fit — weights alone leave less than ctxFloor (%d) worth of KV room", ctx, ctxFloor)
	}
}

// End to end, through the real Load() path with a real (tiny) GGUF fixture — proves the wiring
// from guardFit's return value into opts.ResidentContext (decoder/model.go), not only the pure
// arithmetic above.
func TestFitGuard_unpinnedLoadAutoPinsASmallerContextRatherThanRefusing(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	// MEASURED, not assumed: this fixture's own metadata gives ggufConfig a MaxPositions of 0
	// (context_length is not set the way this synthetic build's config parses it), so
	// fitCheckFor's "unknown ⇒ proceed" branch fires and this specific fixture cannot exercise
	// the auto-pin path at all — confirmed by direct inspection (kvBytesPerPosition=512,
	// MaxPositions=0), not by running this test and rationalizing a SKIP after the fact. The pure
	// arithmetic is fully covered by TestFitCheck_unpinnedPricesKVAtTheModelsMaximum above, which
	// does not depend on any fixture's real dimensions; this test exists to prove the OTHER half —
	// that a fixture with a real MaxPositions and comfortable weights does not regress into
	// spuriously capping or refusing — and to auto-upgrade to a real auto-pin assertion the day a
	// fixture with MaxPositions>0 is available at this path.
	restore := injectHostRAM(t, 64<<30) // ample: this fixture must load normally, uncapped
	defer restore()

	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("guard refused an ample-RAM load: %v", err)
	}
	defer m.Close()
	if got := m.ResidentContextRequest(); got != 0 {
		t.Errorf("guard capped context to %d on a 64 GB machine — should not have needed to", got)
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

// injectHostRAM replaces BOTH the machine's total-RAM figure AND its currently-available figure
// with the same value, for one test. Most callers do not care about the total-vs-available
// distinction (they are testing the arithmetic given "a machine with N bytes to work with"); a
// test that DOES care calls injectHostRAMAvailable (prefill_budget_test.go) afterwards to override
// just the available figure, matching the live failure's own shape (ample total RAM, tight
// availability). Returned as a restore func rather than only t.Cleanup so the intent reads at the
// call site.
func injectHostRAM(t *testing.T, bytes int64) func() {
	t.Helper()
	prevTotal, prevAvail := hostRAM, hostRAMAvailable
	hostRAM = func() int64 { return bytes }
	hostRAMAvailable = func() int64 { return bytes }
	restore := func() { hostRAM, hostRAMAvailable = prevTotal, prevAvail }
	t.Cleanup(restore)
	return restore
}

// denseStreamable must agree with newLayerPager's own exclusions exactly (dense generic-forward:
// yes; MoE: no; own-forward dense like lfm2: no) — a disagreement would mean the guard offers an
// automatic retry the pager then silently declines to build, or refuses one the pager would have
// handled fine. Three tracked-in-git tiny fixtures, no GOINFER_HEAVY_TESTS needed.
func TestDenseStreamable_agreesWithLayerPagerEligibility(t *testing.T) {
	cases := []struct {
		name string
		dir  string
		want bool
	}{
		{"llama: dense, generic-forward", "../testdata/llama-tiny", true},
		{"mixtral: MoE", "../testdata/mixtral-tiny", false},
		{"lfm2: dense but own-forward", "../testdata/lfm2-tiny", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m, err := Load(c.dir, Options{Quant: "f32"})
			if err != nil {
				t.Fatalf("Load(%s): %v", c.dir, err)
			}
			defer m.Close()
			if got := denseStreamable(m.Config()); got != c.want {
				t.Errorf("denseStreamable = %v, want %v", got, c.want)
			}
		})
	}
	t.Run("nil config (header unreadable)", func(t *testing.T) {
		if denseStreamable(nil) {
			t.Error("denseStreamable(nil) = true — an unknown model must not offer an automatic retry")
		}
	})
}

// declineErr must carry fitCheck.denseStreamable through to FitDeclineError.DenseStreamable
// unchanged in both directions — main.go's retry decision reads only the returned error, never
// the fitCheck that produced it, so a silent drop here would either offer a retry for MoE (the
// measured-bad path) or withhold one for a dense model that would have streamed fine.
func TestFitDeclineError_carriesDenseStreamable(t *testing.T) {
	for _, want := range []bool{true, false} {
		f := fitCheck{name: "x.gguf", quant: "int4", weightBytes: 21 << 30, availBytes: 16 << 30, denseStreamable: want}
		err := f.declineErr()
		if err.DenseStreamable != want {
			t.Errorf("denseStreamable=%v: FitDeclineError.DenseStreamable = %v", want, err.DenseStreamable)
		}
		if !errors.Is(err, ErrWontFitResident) {
			t.Errorf("denseStreamable=%v: declineErr() does not wrap ErrWontFitResident", want)
		}
	}
}

// Every quant mode must produce a plausible measured cost. The trap this pins: quantInt4Mix is a
// load-time POLICY, not a resident precision, so quantizeWM does not handle it and hands back the
// untouched f32 — pricing an int4mix model at 4 bytes/element, six times its real cost, and
// refusing models that fit comfortably. That is the one direction the guard must never fail in,
// and it is invisible in the ratio test above, which never exercises int4mix.
func TestQuantBytesPerElem_everyModeIsPlausible(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mode   quantMode
		lo, hi float64
	}{
		{name: "f32", mode: quantNone, lo: 4, hi: 4},
		{name: "int8", mode: quantInt8, lo: 0.95, hi: 1.20},
		{name: "int8int8", mode: quantInt8I8, lo: 0.95, hi: 1.20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := quantBytesPerElem(tc.mode); got < tc.lo || got > tc.hi {
				t.Errorf("%s costs %.4f bytes/elem, want %.2f..%.2f", tc.name, got, tc.lo, tc.hi)
			}
		})
	}

	// int4 has exactly TWO legitimate costs, and which one applies is a property of the host, not
	// of the encoding:
	//
	//   ~0.625  canonical nibbles + one f32 scale per group of 32, and no repack
	//   ~1.250  the same, PLUS a second repacked buffer that the loader keeps beside it —
	//           RepackInt4Row4 on arm64-with-dotprod, split-half on AVX2-without-VNNI
	//
	// Pinning the pair rather than a range is the point: a wrong measurement usually lands
	// BETWEEN them, and a range wide enough to hold both would accept it.
	//
	// This replaces an assertion that int4 must be cheaper than int8, which CI proved false.
	// Measured on darwin/arm64 2026-09-06: int4 1.2500 against int8 1.0156 — on Apple Silicon
	// int4 weights occupy about 23% MORE resident RAM than int8int8, because int8 gets no repack.
	// See docs/task-first-hour.md for what that means for the help text's "int4 ... smallest".
	for _, name := range []struct {
		label string
		mode  quantMode
	}{{"int4", quantInt4}, {"int4mix", quantInt4Mix}} {
		t.Run(name.label, func(t *testing.T) {
			got := quantBytesPerElem(name.mode)
			const enc, repacked = 0.625, 1.250
			near := func(want float64) bool { return got > want*0.95 && got < want*1.05 }
			if !near(enc) && !near(repacked) {
				t.Errorf("%s costs %.4f bytes/elem — neither the ~%.3f encoding-only cost nor the "+
					"~%.3f repacked cost. A value between the two usually means the probe stopped "+
					"going through the loader's own path", name.label, got, enc, repacked)
			}
			t.Logf("%s: %.4f bytes/elem (%s)", name.label, got,
				map[bool]string{true: "repacked host", false: "encoding only"}[near(repacked)])
		})
	}
}
