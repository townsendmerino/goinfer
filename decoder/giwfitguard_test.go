package decoder

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// task-never-swap-2026-09.md S4 item 1: a .giw load's weights are file-backed (fitCheckFor is
// never called for one, by design — its own srcFileBytes doc comment), but KV and prefill scratch
// ARE real anonymous cost this path priced nowhere before. guardGIWFit is the load-time guard;
// these are its own gates, mirroring fitguard_test.go's existing conventions for the .gguf side
// (injectHostRAM, the same Config field shape) rather than inventing new ones.

func TestGuardGIWFit_fitsComfortably(t *testing.T) {
	defer injectHostRAM(t, 16<<30)()
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 8192}
	ctx, err := guardGIWFit(cfg, Options{})
	if err != nil {
		t.Fatalf("a load with ample headroom must not be refused: %v", err)
	}
	if ctx != 0 {
		t.Errorf("ctx = %d, want 0 (no pin needed)", ctx)
	}
}

func TestGuardGIWFit_unpinnedAutoPinsToFloor(t *testing.T) {
	// A huge MaxPositions with tight availability: KV at the model's own max overflows, but
	// ctxFloor's own KV is small enough to fit comfortably — forcing the binary search to do
	// real work rather than degenerate to either boundary trivially.
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	perPos := kvBytesPerPosition(cfg, false, false)
	// budget = avail - 1GB margin. Set avail so budget covers roughly ctxFloor*4 positions —
	// comfortably between ctxFloor and MaxPositions, so the binary search's result is checkable.
	budget := perPos*int64(ctxFloor*4) + prefillAttnScratchBudget
	avail := budget + giwMemMargin
	defer injectHostRAM(t, avail)()

	ctx, err := guardGIWFit(cfg, Options{})
	if err != nil {
		t.Fatalf("an unpinned load that fits at SOME smaller context must auto-pin, not refuse: %v", err)
	}
	if ctx <= 0 {
		t.Fatal("want a nonzero auto-pinned context")
	}
	if ctx >= cfg.MaxPositions {
		t.Errorf("ctx = %d did not actually shrink from MaxPositions %d", ctx, cfg.MaxPositions)
	}
	// The chosen ctx must fit, and ctx+1 (if under MaxPositions) must not — otherwise the binary
	// search picked something other than the LARGEST fitting context.
	needAt := func(c int) int64 { return kvBytesPerPosition(cfg, false, false)*int64(c) + prefillAttnScratchBudget }
	if needAt(ctx) > budget {
		t.Errorf("chosen ctx %d needs %d bytes, exceeds budget %d", ctx, needAt(ctx), budget)
	}
	if ctx+1 < cfg.MaxPositions && needAt(ctx+1) <= budget {
		t.Errorf("ctx %d fits but so does ctx+1 (%d) — did not find the LARGEST fitting context", ctx, ctx+1)
	}
}

func TestGuardGIWFit_unpinnedRefusesWhenEvenFloorDoesNotFit(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	// Budget too small even for ctxFloor's own KV + scratch.
	defer injectHostRAM(t, giwMemMargin+prefillAttnScratchBudget/2)()
	ctx, err := guardGIWFit(cfg, Options{})
	if err == nil {
		t.Fatalf("want a refusal when even ctxFloor does not fit, got ctx=%d nil error", ctx)
	}
	if !strings.Contains(err.Error(), "GOINFER_NO_FIT_GUARD") {
		t.Errorf("refusal %q does not name the escape hatch", err.Error())
	}
}

func TestGuardGIWFit_pinnedContextThatDoesNotFitIsRefusedNotDowngraded(t *testing.T) {
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	pinnedCtx := 1 << 18
	needed := kvBytesPerPosition(cfg, false, false)*int64(pinnedCtx) + prefillAttnScratchBudget
	// Budget covers ctxFloor comfortably but NOT the pinned request — an explicit pin that
	// cannot be honoured must refuse (G-07), never silently shrink to something smaller.
	defer injectHostRAM(t, needed/2+giwMemMargin)()
	ctx, err := guardGIWFit(cfg, Options{ResidentContext: pinnedCtx})
	if err == nil {
		t.Fatalf("a pinned -ctx that does not fit must be refused, got ctx=%d nil error", ctx)
	}
	if strings.Contains(err.Error(), "auto-pin") {
		t.Error("a pinned request's refusal must not suggest it was auto-pinned")
	}
}

func TestGuardGIWFit_envOverrideBypasses(t *testing.T) {
	t.Setenv("GOINFER_NO_FIT_GUARD", "1")
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	defer injectHostRAM(t, 1)() // absurdly tight — would refuse without the override
	ctx, err := guardGIWFit(cfg, Options{})
	if err != nil || ctx != 0 {
		t.Fatalf("GOINFER_NO_FIT_GUARD=1 must bypass entirely, got (%d, %v)", ctx, err)
	}
}

func TestGuardGIWFit_unknownAvailableProceeds(t *testing.T) {
	defer injectHostRAM(t, 0)() // 0 = "could not be determined" (HostRAMAvailableBytes' own contract)
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	ctx, err := guardGIWFit(cfg, Options{})
	if err != nil || ctx != 0 {
		t.Fatalf("an unreadable live probe must proceed unguarded, got (%d, %v)", ctx, err)
	}
}

func TestGuardGIWFit_noConfigProceeds(t *testing.T) {
	defer injectHostRAM(t, 1)()
	ctx, err := guardGIWFit(nil, Options{})
	if err != nil || ctx != 0 {
		t.Fatalf("a nil Config must proceed unguarded (can't price KV), got (%d, %v)", ctx, err)
	}
}

func TestGuardGIWFit_unpinnedNoMaxPositionsProceeds(t *testing.T) {
	defer injectHostRAM(t, 1)()
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128} // MaxPositions unset
	ctx, err := guardGIWFit(cfg, Options{})
	if err != nil || ctx != 0 {
		t.Fatalf("unpinned with no known MaxPositions can't price KV, must proceed: got (%d, %v)", ctx, err)
	}
}

// TestGuardGIWFit_marginActuallyApplied mutation-checks the margin itself: without it, a load
// sized to fit EXACTLY at avail-bytes (no margin subtracted) would pass; WITH the registered 1 GB
// margin, that same load must be refused. Pins giwMemMargin's own value rather than assuming it.
func TestGuardGIWFit_marginActuallyApplied(t *testing.T) {
	if giwMemMargin != 1<<30 {
		t.Fatalf("giwMemMargin = %d, want 1 GiB (1<<30) — the brief's own registered rule; update this test if the rule changes", giwMemMargin)
	}
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	need := kvBytesPerPosition(cfg, false, false)*int64(cfg.MaxPositions) + prefillAttnScratchBudget
	// avail sized to exactly cover `need` with NOTHING left for a margin: without giwMemMargin
	// being subtracted, this would fit exactly (ctx == MaxPositions, no pin); with it applied,
	// the budget is smaller than `need` by exactly the margin, so SOME reduction must happen —
	// either an auto-pin to a smaller context, or (if even ctxFloor no longer fits) a refusal.
	defer injectHostRAM(t, need)()
	ctx, err := guardGIWFit(cfg, Options{})
	switch {
	case err != nil:
		// Refused — the margin ate the entire slack. Also a valid sign the margin was applied.
	case ctx == 0:
		t.Fatal("ctx=0 (no pin) with no error — a load sized to consume ALL available memory " +
			"with zero left for the margin fit at its full context anyway; the 1 GB margin was not applied")
	case ctx >= cfg.MaxPositions:
		t.Errorf("auto-pinned ctx %d did not actually shrink from MaxPositions %d — the margin was not applied", ctx, cfg.MaxPositions)
	}
}

func TestGuardGIWFit_marginTooLargeForBudgetProceedsAsRefused(t *testing.T) {
	// avail below the margin itself: budget must clamp to 0, not go negative and wrap/misbehave.
	defer injectHostRAM(t, giwMemMargin/2)()
	cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 20}
	if _, err := guardGIWFit(cfg, Options{}); err == nil {
		t.Fatal("available memory below the margin itself must refuse (budget clamps to 0), not silently permit everything")
	}
}

// TestLoad_giwAutoPinsUnderTightMemory is the real end-to-end wiring proof: a genuine .giw file,
// loaded through the real Load function (not guardGIWFit called standalone), auto-pins to a
// smaller context under an injected tight memory reading — same pattern
// eos_giw_roundtrip_test.go's own TestGIWRoundTrip_preservesMergedEOSIDs uses to build a real
// on-disk bundle (giw.WriteStream(SerializeWeightsToForTarget)), combined with
// tinyNormRopeGGUF's synthetic source (gguf_granite_permute_test.go).
func TestLoad_giwAutoPinsUnderTightMemory(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	// tinyNormRopeGGUF's fixture carries a real MaxPositions (its context_length KV, 64) — too
	// small to exercise a meaningful auto-pin range on its own, so widen it directly on the
	// already-built Weights before serializing, matching how a real large-context model would
	// look to this guard.
	w.Cfg.MaxPositions = 1 << 20

	out := filepath.Join(t.TempDir(), "tight.giw")
	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create bundle: %v", err)
	}
	werr := giw.WriteStream(f, nil, func(dst io.Writer) (int64, error) {
		return SerializeWeightsToForTarget(dst, w, "tight-fixture", GIWTargetNone)
	})
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		t.Fatalf("write bundle: %v", werr)
	}

	// &w.Cfg directly — the actual struct Load will use internally — rather than hand-copying
	// fields into a fresh Config, which silently dropped HeadDim to 0 the first time this was
	// written (caught here: HeadDim=0 makes estimateKVBytes return 0 regardless of context,
	// so the "floor vs max" need never actually differed and this test could not have told
	// an auto-pin from a no-op).
	floorNeed := estimateKVBytes(&w.Cfg, ctxFloor, false, false) + prefillAttnScratchBudget
	maxNeed := estimateKVBytes(&w.Cfg, w.Cfg.MaxPositions, false, false) + prefillAttnScratchBudget
	if maxNeed <= floorNeed {
		t.Fatalf("test fixture problem: needAt(MaxPositions)=%d <= needAt(ctxFloor)=%d — KV does not "+
			"actually grow with context for this fixture, so no budget exists that would force a "+
			"real auto-pin rather than a no-op", maxNeed, floorNeed)
	}
	// Budget strictly between the two: too small for MaxPositions, comfortably above ctxFloor.
	budget := (floorNeed + maxNeed) / 2
	defer injectHostRAM(t, budget+giwMemMargin)()

	m, err := Load(out, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load(.giw) under a tight memory reading must auto-pin, not refuse or crash: %v", err)
	}
	defer m.Close()
	if m.resCtxReq <= 0 {
		t.Fatal("Load did not record an auto-pinned context on the Model — guardGIWFit's result never reached opts.ResidentContext")
	}
	if m.resCtxReq >= w.Cfg.MaxPositions {
		t.Errorf("m.resCtxReq = %d did not actually shrink from MaxPositions %d", m.resCtxReq, w.Cfg.MaxPositions)
	}
}

func TestGuardGIWFit_scratchBudgetIsPositive(t *testing.T) {
	// Guards the assumption every other test in this file leans on: a zero or negative scratch
	// term would make several "needs > budget" comparisons above vacuously true for the wrong
	// reason.
	if prefillAttnScratchBudget <= 0 {
		t.Fatalf("prefillAttnScratchBudget = %d, want > 0", prefillAttnScratchBudget)
	}
}
