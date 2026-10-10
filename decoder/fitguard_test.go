package decoder

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// Pins that the fit guard refuses BEFORE anything is allocated and that the message names the flag that fixes it:
// "an error is returned" is not the bar (docs/measurements/cold-user-2026-09-06.md, scenario D). The RAM figure is
// injected, so the 16 GB machine's arithmetic runs on any box.
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

	// The message must carry the remedy and the arithmetic: the user reading it does not know -stream-weights exists.
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

	// gptoss_tiny.gguf is gpt-oss (MoE, own-forward), so an automatic -stream-weights retry must NOT be offered: that CPU
	// path is the one docs/benchmarks.md "M35/M26 on the Mac" records as never completing on a real checkpoint. The sentinel
	// (errors.Is) and the typed field (errors.As) must agree, since main.go's retry decision reads the field directly.
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

// TestFitGuard_pricesTheOnDiskGGUFDuringLoadNotJustFinalWeights pins that the load-time estimate counts the on-disk
// GGUF (srcFileBytes) on top of weightBytes+kvBytes, because a CPU-resident .gguf load holds both
// (docs/measurements/cold-user-2026-09-18-nobara-pc.md, scenario D). It reads the REAL terms off fitCheckFor rather
// than re-deriving them by hand (a hand derivation under-counts KV and then passes against the pre-fix estimator),
// picks a RAM figure whose budget sits strictly between (weightBytes+kvBytes) and (weightBytes+kvBytes+srcFileBytes),
// and confirms the guard refuses on srcFileBytes alone.
func TestFitGuard_pricesTheOnDiskGGUFDuringLoadNotJustFinalWeights(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"

	// availBytes doesn't affect weightBytes/kvBytes/srcFileBytes, so this reads the real terms
	// before choosing what RAM to inject.
	probe := fitCheckFor(gguf, "int4", quantInt4, Options{})
	if probe.weightBytes <= 0 {
		t.Fatal("estimator returned 0 for a real GGUF")
	}
	if probe.srcFileBytes <= 0 {
		t.Fatal("srcFileBytes is 0 for a plain .gguf resident load — the fix did not populate it")
	}
	oldNeed := probe.weightBytes + probe.kvBytes // what the PRE-FIX need() computed
	newNeed := oldNeed + probe.srcFileBytes      // what need() computes now
	// Budget must land strictly between the two: old formula fits, new formula does not.
	ram := int64(float64(oldNeed)/fitMemFraction) + 4096
	budget := int64(float64(ram) * fitMemFraction)
	if oldNeed > budget {
		t.Fatalf("test setup: oldNeed (%d) already exceeds budget (%d) at ram=%d — margin too small", oldNeed, budget, ram)
	}
	if newNeed <= budget {
		t.Fatalf("test setup: newNeed (%d) still fits budget (%d) at ram=%d — fixture's file too small relative to its own weight+KV estimate to exercise this fix", newNeed, budget, ram)
	}

	restore := injectHostRAM(t, ram)
	defer restore()

	before := weightAllocs.Load()
	m, err := Load(gguf, Options{Quant: "int4"})
	if err == nil {
		if m != nil {
			m.Close()
		}
		t.Fatalf("Load succeeded with RAM sized for weights+KV alone (%d bytes, budget %d) — "+
			"the guard is not pricing the on-disk .gguf file (%d bytes) that stays mmap-resident "+
			"for the whole load, on top of the resident weights being built from it", ram, budget, probe.srcFileBytes)
	}
	if got := weightAllocs.Load() - before; got != 0 {
		t.Errorf("loadWeights was entered %d time(s) despite the refusal", got)
	}
	if !strings.Contains(err.Error(), "reading the checkpoint") {
		t.Errorf("refusal does not name the on-disk-file term:\n%s", err.Error())
	}
}

// TestFitGuard_pricesTheCUDAExpertCacheBuildPeak: --backend cuda --moe-cache-experts holds the canonical weights, a packed
// copy of them and a pinned copy of the experts on the host at once. In the regime that matters that exceeds the CPU path's
// weights+KV+file total, and it does not stack with the file term (the source is unmapped before the build), so need()
// takes the larger.
//
// Two halves, because the tiny fixture cannot be in that regime (its KV at the full window dwarfs its weights): the wiring
// is checked on the real GGUF, the arithmetic on realistic numbers.
func TestFitGuard_pricesTheCUDAExpertCacheBuildPeak(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	opts := Options{Quant: "int4", Backend: "cuda", MoECacheExperts: true}

	probe := fitCheckFor(gguf, "int4", quantInt4, opts)
	if probe.expertBytes <= 0 {
		t.Fatal("no routed-expert bytes found in a MoE GGUF — the '_exps' tensor match is broken")
	}
	if want := 2*probe.weightBytes + probe.expertBytes; probe.cudaBuildBytes != want {
		t.Fatalf("cudaBuildBytes = %d, want 2*weights+experts = %d", probe.cudaBuildBytes, want)
	}
	if plain := fitCheckFor(gguf, "int4", quantInt4, Options{Quant: "int4", Backend: "cuda"}); plain.cudaBuildBytes != 0 {
		t.Errorf("cudaBuildBytes = %d without --moe-cache-experts; the term is only for the expert-cache build", plain.cudaBuildBytes)
	}
	if cpu := fitCheckFor(gguf, "int4", quantInt4, Options{Quant: "int4", Backend: "cpu", MoECacheExperts: true}); cpu.cudaBuildBytes != 0 {
		t.Errorf("cudaBuildBytes = %d on the cpu backend", cpu.cudaBuildBytes)
	}
	if st := fitCheckFor(gguf, "int4", quantInt4, Options{Quant: "int4", Backend: "cuda", MoECacheExperts: true, StreamWeights: true}); st.cudaBuildBytes != 0 {
		t.Errorf("cudaBuildBytes = %d with StreamWeights (that load goes through a .giw)", st.cudaBuildBytes)
	}

	// gpt-oss-20b's measured numbers on a 64 GB box: 12.2 GB weights, 11.1 GB of them experts,
	// 11.3 GB file, 6 GB worst-case KV. CPU path: 29.5 GB. CUDA build: 35.5 GB. Budget: between.
	f := probe
	f.weightBytes, f.expertBytes = 12<<30, 11<<30
	f.srcFileBytes, f.kvBytes = 11<<30, 6<<30
	f.cudaBuildBytes = 2*f.weightBytes + f.expertBytes
	budgetGB := 33.0
	f.availBytes = int64(budgetGB * fitGB / fitMemFraction) // budget 33 GB: fits 29 GB, not 35 GB
	if cpuNeed := f.weightBytes + f.kvBytes + f.srcFileBytes; f.need() != f.cudaBuildBytes || cpuNeed >= f.budget() || f.cudaBuildBytes <= f.budget() {
		t.Fatalf("test setup: need=%d build=%d cpu=%d budget=%d", f.need(), f.cudaBuildBytes, cpuNeed, f.budget())
	}
	if f.fits() {
		t.Fatal("fits() is true with the budget below the CUDA build peak")
	}
	if !strings.Contains(f.arithmetic(), "CUDA expert cache") {
		t.Errorf("arithmetic does not name the CUDA expert-cache term:\n%s", f.arithmetic())
	}
	// The peak has no KV in it, so shrinking the context cannot fix it; guardFit must refuse
	// instead of auto-pinning a smaller context and letting the load through.
	if ctx, err := guardFit(f); err == nil {
		t.Fatalf("guardFit passed (pinned ctx %d) although the CUDA expert-cache build peak exceeds the budget", ctx)
	}
	// And with headroom above the peak it is silent.
	budgetGB = 80.0
	f.availBytes = int64(budgetGB * fitGB / fitMemFraction)
	if _, err := guardFit(f); err != nil {
		t.Errorf("guardFit refused a machine with ample headroom: %v", err)
	}
}

// TestFitGuard_streamWeightsSourceIsNotDoubleCounted: the srcFileBytes term must NOT apply when
// StreamWeights is set on a .gguf path — that request is resolved by transcoding to a .giw and
// re-loading from THAT (a separate Load call this guard never sees), so pricing the source .gguf's
// size here would price a load that is not actually about to happen through this path. Uses the
// same RAM figure the test above proves refuses WITHOUT this guard, to show StreamWeights changes
// the outcome.
func TestFitGuard_streamWeightsSourceIsNotDoubleCounted(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	f := fitCheckFor(gguf, "int4", quantInt4, Options{StreamWeights: true})
	if f.srcFileBytes != 0 {
		t.Errorf("srcFileBytes = %d, want 0 when StreamWeights is set (the .gguf is not what actually gets resident-loaded)", f.srcFileBytes)
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

// The same bypass through Options.Knobs, for one model, with the environment clean — the per-model route
// (docs/tasks/task-env-config-2026-09.md, phase 2b). The guard runs inside Load before the model's knob
// snapshot exists, so it reads the knob through loadKnob; this pins that the override reaches it. The
// refusing half is TestFitGuard_envOverrideLoads' own premise (same fixture, same injected RAM).
func TestFitGuard_optionsKnobLoads(t *testing.T) {
	restore := injectHostRAM(t, 128<<10)
	defer restore()
	t.Setenv("GOINFER_NO_FIT_GUARD", "")
	os.Unsetenv("GOINFER_NO_FIT_GUARD")

	if m, err := Load("testdata/gptoss_tiny.gguf", Options{Quant: "int4"}); err == nil {
		m.Close()
		t.Fatal("control arm loaded: with no override the guard must refuse this fixture at this RAM, or the arm below proves nothing")
	}
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Quant: "int4", Knobs: &Knobs{"GOINFER_NO_FIT_GUARD": "1"}})
	if err != nil {
		t.Fatalf("Options.Knobs GOINFER_NO_FIT_GUARD=1 did not bypass the guard: %v", err)
	}
	if m.knobs.get(knobNoFitGuard) != "1" {
		t.Error("the model's snapshot must carry the override too, for the per-request guard (AdmitPrefillMemory)")
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

// The load-time guard must price against CURRENTLY AVAILABLE memory, not total RAM: ample total RAM with tight
// availability is the shape of the live failure (docs/measurements/cold-user-2026-09-07-macbook-arm64.md, second live
// re-run).
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

// An UNPINNED load must price KV at the model's maximum context: a load that fits at idle can otherwise swap the
// machine on its first real request (docs/measurements/cold-user-2026-09-07-macbook-arm64.md, R13). Driven with a
// 7B-class shape whose weights fit comfortably and whose KV at the full context window does not.
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

// Through the real Load() path on the tiny GGUF fixture: an ample-RAM load is neither capped nor pinned. The auto-pin
// wiring (guardFit's return value into opts.ResidentContext) is NOT asserted here; see the comment in the body.
func TestFitGuard_unpinnedLoadAutoPinsASmallerContextRatherThanRefusing(t *testing.T) {
	const gguf = "testdata/gptoss_tiny.gguf"
	// This fixture's ggufConfig has MaxPositions 0 (context_length does not parse on this synthetic build), so fitCheckFor's
	// "unknown proceeds" branch fires and the auto-pin path is not exercised. The arithmetic is covered by
	// TestFitCheck_unpinnedPricesKVAtTheModelsMaximum, which does not depend on a fixture's dimensions; this test pins the
	// other half: a comfortable-weights load does not regress into capping or refusing.
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
	if m.ResidentContextPinned() {
		t.Error("ResidentContextPinned() = true for a load that asked for no context")
	}
}

// The estimator must not drift from the accountant. It prices the model from GGUF metadata BEFORE the load;
// ResidentWeightBytes sums the matrices AFTER it. They answer the same question from opposite sides, so a large
// disagreement means one of them is wrong.
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
			// TIGHT on purpose: quantBytesPerElem MEASURES through quantizeWM, so a wide band would hide a platform divergence
			// (the arm64 W4A8 row4 repack keeps a second buffer, so int4 costs about twice its encoding there). Some slack
			// remains because the estimator prices every tensor uniformly while the accountant reads the real backing slices.
			if ratio < 0.85 || ratio > 1.25 {
				t.Errorf("estimate %d vs accounted %d (ratio %.2f) — the pre-load estimate has "+
					"drifted from ResidentWeightBytes", est, actual, ratio)
			}
			t.Logf("%s: estimate %d, accounted %d, ratio %.2f", q.name, est, actual, ratio)
		})
	}
}

// TestFitEstimate_safetensorsAgreesWithResidentWeightBytes is TestFitEstimate_agreesWithResidentWeightBytes's safetensors
// twin (docs/multimodal.md, P9b): estimateSafetensorsWeightBytes's shape-based, quant-priced estimate must track a real
// load's resident weight bytes, on a real (tracked, non-gitignored) safetensors checkpoint. A safetensors path used to
// return a zero weight estimate, so fits() was always true.
func TestFitEstimate_safetensorsAgreesWithResidentWeightBytes(t *testing.T) {
	const dir = "testdata/internlm2-tiny"
	for _, q := range []struct {
		name string
		mode quantMode
	}{{"int4", quantInt4}, {"int8int8", quantInt8I8}} {
		t.Run(q.name, func(t *testing.T) {
			est := estimateSafetensorsWeightBytes(dir, q.mode)
			if est <= 0 {
				t.Fatal("estimator returned 0 for a real safetensors checkpoint")
			}
			m, err := Load(dir, Options{Quant: q.name})
			if err != nil {
				t.Skipf("cannot load fixture at %s: %v", q.name, err)
			}
			defer m.Close()
			actual := m.ResidentWeightBytes()
			if actual <= 0 {
				t.Skip("accountant reported 0 for this fixture")
			}
			ratio := float64(est) / float64(actual)
			// Same band as the GGUF twin, same reasoning: quantBytesPerElem measures through the
			// real quantizeWM path, so remaining slack is the estimator pricing every tensor
			// uniformly while the accountant reads the real backing slices.
			if ratio < 0.85 || ratio > 1.25 {
				t.Errorf("estimate %d vs accounted %d (ratio %.2f) — the pre-load safetensors "+
					"estimate has drifted from ResidentWeightBytes", est, actual, ratio)
			}
			t.Logf("%s: estimate %d, accounted %d, ratio %.2f", q.name, est, actual, ratio)
		})
	}
}

// TestFitCheckFor_pricesSafetensorsNotJustGGUF pins that fitCheckFor no longer returns a zero-weight (always-fits)
// check for a non-.gguf path: weightBytes and kvBytes are both non-zero for a real safetensors directory with a
// resolvable context.
func TestFitCheckFor_pricesSafetensorsNotJustGGUF(t *testing.T) {
	const dir = "testdata/internlm2-tiny"
	f := fitCheckFor(dir, "int4", quantInt4, Options{})
	if f.weightBytes <= 0 {
		t.Fatalf("fitCheckFor(%s).weightBytes = %d, want > 0 — safetensors path still unpriced", dir, f.weightBytes)
	}
	if f.cfg == nil {
		t.Fatal("fitCheckFor(dir).cfg = nil — config.json was not read for the safetensors path")
	}
	if f.effCtx <= 0 {
		t.Fatalf("fitCheckFor(%s).effCtx = %d, want > 0 (a real config with max_position_embeddings)", dir, f.effCtx)
	}
	if f.kvBytes <= 0 {
		t.Fatalf("fitCheckFor(%s).kvBytes = %d, want > 0", dir, f.kvBytes)
	}
}

// TestFitCheckFor_unresolvableSafetensorsProceedsUnknown pins the "every unknown proceeds"
// discipline for the NEW branch specifically: a directory that is not a real safetensors
// checkpoint (no config.json, no model.safetensors) must return a zero-weight, unknown check —
// never an error, never a panic — exactly like an unreadable .gguf already does.
func TestFitCheckFor_unresolvableSafetensorsProceedsUnknown(t *testing.T) {
	f := fitCheckFor(t.TempDir(), "int4", quantInt4, Options{})
	if f.weightBytes != 0 {
		t.Errorf("weightBytes = %d, want 0 (unresolvable directory)", f.weightBytes)
	}
	if !f.fits() {
		t.Error("an unresolvable directory must proceed (fits() == true, the unknown case), not refuse")
	}
}

// TestFitGuard_remedyNamesPrequantNotStreamWeightsForDirectory (M-30): the remedy for a refused safetensors DIRECTORY
// must not recommend -stream-weights, which is a no-op for a directory (decoder.Load ignores it for anything but a
// .giw). It must name the escape hatch that works for this source (GOINFER_NO_FIT_GUARD=1) and the permanent fix
// (cmd/prequant).
func TestFitGuard_remedyNamesPrequantNotStreamWeightsForDirectory(t *testing.T) {
	restore := injectHostRAM(t, 128<<10) // far below anything internlm2-tiny needs — forces refusal
	defer restore()

	m, err := Load("testdata/internlm2-tiny", Options{Quant: "int4"})
	if err == nil {
		if m != nil {
			m.Close()
		}
		t.Fatal("Load succeeded on a machine too small to hold the model — the guard did not fire")
	}
	msg := err.Error()
	if strings.Contains(msg, "Re-run goinfer-serve with -stream-weights") {
		t.Errorf("refusal for a safetensors DIRECTORY still RECOMMENDS -stream-weights (a no-op for it):\n%s", msg)
	}
	for _, want := range []string{"GOINFER_NO_FIT_GUARD", "cmd/prequant", "safetensors"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal does not mention %q:\n%s", want, msg)
		}
	}
}

// injectHostRAM replaces BOTH the total-RAM figure AND the currently-available figure with the same value, for one
// test. A test that cares about the difference calls injectHostRAMAvailable (prefill_budget_test.go) afterwards to
// override just the available figure. Returned as a restore func rather than only t.Cleanup so the intent reads at the
// call site.
func injectHostRAM(t *testing.T, bytes int64) func() {
	t.Helper()
	prevTotal, prevAvail := hostRAM, hostRAMAvailable
	hostRAM = func() int64 { return bytes }
	hostRAMAvailable = func() int64 { return bytes }
	resetAvailProbeCache()
	restore := func() {
		hostRAM, hostRAMAvailable = prevTotal, prevAvail
		resetAvailProbeCache()
	}
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

// declineErr must carry fitCheck.denseStreamable through to FitDeclineError.DenseStreamable unchanged in both
// directions: main.go's retry decision reads only the returned error, so a silent drop would offer a retry for MoE (the
// path that never completes) or withhold one for a dense model that would have streamed.
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

	// int4 has exactly TWO legitimate costs, and which applies is a property of the host, not of the encoding:
	//
	//   ~0.5625 canonical nibbles + one binary16 scale per group of 32, and no repack
	//   ~1.125  the same, PLUS a second repacked buffer the loader keeps beside it (RepackInt4Row4 on arm64-with-dotprod)
	//
	// Pinning the pair rather than a range is the point: a wrong measurement usually lands BETWEEN them, and a range wide enough
	// to hold both would accept it. Do not assert int4 < int8: on Apple Silicon int4 occupies MORE resident RAM than int8int8,
	// because int8 gets no repack (docs/tasks/task-first-hour.md).
	for _, name := range []struct {
		label string
		mode  quantMode
	}{{"int4", quantInt4}, {"int4mix", quantInt4Mix}} {
		t.Run(name.label, func(t *testing.T) {
			got := quantBytesPerElem(name.mode)
			const enc, repacked = 0.5625, 1.125
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

// TestResolveWeightCacheBudget is S4 item 2's own gate (task-never-swap-2026-09.md): an explicit
// request passes through unchanged, an "auto" (0) request resolves to half of the LIVE probe (not
// aikit's Linux-only /proc probe or its fixed 8 GB darwin fallback), and only falls through to 0
// (aikit's own AutoBudget, now genuinely the last resort) when this platform's live probe is
// itself unavailable.
func TestResolveWeightCacheBudget(t *testing.T) {
	const gb = int64(1) << 30
	t.Run("explicit request passes through unchanged", func(t *testing.T) {
		defer injectHostRAMAvailable(t, 16*gb)()
		if got := resolveWeightCacheBudget(3 * gb); got != 3*gb {
			t.Errorf("resolveWeightCacheBudget(3 GB) = %d, want 3 GB unchanged", got)
		}
	})
	t.Run("auto (0) resolves to half the live probe", func(t *testing.T) {
		defer injectHostRAMAvailable(t, 16*gb)()
		if got, want := resolveWeightCacheBudget(0), 8*gb; got != want {
			t.Errorf("resolveWeightCacheBudget(0) with 16 GB available = %d, want %d (half)", got, want)
		}
	})
	t.Run("live probe unavailable falls through to 0", func(t *testing.T) {
		defer injectHostRAMAvailable(t, 0)()
		if got := resolveWeightCacheBudget(0); got != 0 {
			t.Errorf("resolveWeightCacheBudget(0) with no live probe = %d, want 0 (aikit's own fallback applies)", got)
		}
	})
}
