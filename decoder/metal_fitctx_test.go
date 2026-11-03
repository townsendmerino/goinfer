package decoder

import (
	"strings"
	"testing"
)

// fitCtxModel is llama-tiny at int4 with its context window widened to 131072, so MetalFitCtx's candidate is
// MetalFitDefaultCtx itself and not the fixture's 128 positions. It also returns the bytes Plan("metal") holds fixed
// (weights and host copy) and the KV one position costs, the two figures a case sizes its budget from.
func fitCtxModel(t *testing.T, opts Options) (m *Model, fixed, perBlock int64) {
	t.Helper()
	if opts.Quant == "" {
		opts.Quant = "int4"
	}
	m, err := Load("../testdata/llama-tiny", opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	m.w.Cfg.MaxPositions = 131072
	p := m.Plan("metal", 1<<40, PlanRequest{Ctx: MetalCtxDefault, CtxPinned: true})
	if p.Placement != PlacementResident {
		t.Fatalf("llama-tiny does not plan resident on metal: %s", p.Reason)
	}
	perBlock = m.MetalKVSlotBytes(KVBlockRows)
	if perBlock <= 0 {
		t.Fatal("llama-tiny prices no KV on metal; every case below would pass on nothing")
	}
	null := m.ResidentKVBytes("metal", KVBlockRows, false, false) - perBlock // the pools' null blocks, once per load
	return m, p.DenseBytes + p.HostCopyBytes + null, perBlock
}

// budgetFor is the probe figure at which exactly `positions` positions of KV fit beside the weights and the margin.
func budgetFor(fixed, perBlock int64, positions int) int64 {
	return MetalFitMarginBytes + fixed + perBlock*int64(positions)/KVBlockRows
}

// TestMetalFitCtx_growsWhenItFitsAndShrinksWhenItDoesNot: an unpinned Metal load asks for MetalFitDefaultCtx and takes
// it when memory holds it; a budget that holds less gets the whole blocks it holds, and never less than
// MetalCtxDefault.
func TestMetalFitCtx_growsWhenItFitsAndShrinksWhenItDoesNot(t *testing.T) {
	m, fixed, perBlock := fitCtxModel(t, Options{})
	for _, c := range []struct {
		name      string
		positions int // KV positions the budget holds
		want      int
	}{
		{"room for the whole candidate", 40000, MetalFitDefaultCtx},
		{"exactly the candidate", MetalFitDefaultCtx, MetalFitDefaultCtx},
		{"room for 6000: the whole blocks in it", 6000, 5952},
		{"room for 6016: exactly its blocks", 6016, 6016},
		{"room for the default only", MetalCtxDefault, MetalCtxDefault},
		{"room for less than the default: the default, and the guard judges it", 1000, MetalCtxDefault},
	} {
		f := m.MetalFitCtx(MetalFitInput{FreeBytes: budgetFor(fixed, perBlock, c.positions), FreeKnown: true, KVSlots: 1})
		if f.Ctx != c.want {
			t.Errorf("%s: ctx %d (%s), want %d", c.name, f.Ctx, f.Why, c.want)
		}
		if want := m.ResidentKVBytes("metal", f.Ctx, false, false); f.KVBytes != want {
			t.Errorf("%s: KVBytes %d, want %d (one slot at ctx %d)", c.name, f.KVBytes, want, f.Ctx)
		}
		if grew := f.Ctx > MetalCtxDefault; grew != strings.HasPrefix(f.Why, "fit: 1 slot,") {
			t.Errorf("%s: ctx %d with reason %q; a grown context says \"fit: 1 slot, ...\" and no other does", c.name, f.Ctx, f.Why)
		}
	}
}

// TestMetalFitCtx_slotsTradeContext: with more than one KV slot asked for, the context gives way until every slot fits,
// and stops at MetalCtxDefault (the build then grants the slots that fit there).
func TestMetalFitCtx_slotsTradeContext(t *testing.T) {
	m, fixed, perBlock := fitCtxModel(t, Options{})
	free := budgetFor(fixed, perBlock, 20000)
	prev := MetalFitDefaultCtx + 1
	for _, c := range []struct{ slots, want int }{{1, MetalFitDefaultCtx}, {2, 9984}, {4, 4992}, {8, MetalCtxDefault}, {64, MetalCtxDefault}} {
		f := m.MetalFitCtx(MetalFitInput{FreeBytes: free, FreeKnown: true, KVSlots: c.slots})
		if f.Ctx != c.want {
			t.Errorf("%d slots: ctx %d (%s), want %d", c.slots, f.Ctx, f.Why, c.want)
		}
		if f.Ctx > prev {
			t.Errorf("%d slots took ctx %d, more than fewer slots did (%d)", c.slots, f.Ctx, prev)
		}
		if f.Ctx < MetalCtxDefault {
			t.Errorf("%d slots: ctx %d is under the default %d", c.slots, f.Ctx, MetalCtxDefault)
		}
		prev = f.Ctx
		if f.KVSlots != c.slots || f.KVBytes != m.MetalKVBytes(c.slots, f.Ctx) {
			t.Errorf("%d slots: answer prices %d slots, %d B of KV", c.slots, f.KVSlots, f.KVBytes)
		}
		if f.Ctx > MetalCtxDefault {
			// The choice must really hold every slot: one more grid step must not.
			need := func(ctx int) int64 {
				return MetalFitMarginBytes + fixed + int64(c.slots)*m.MetalKVSlotBytes(ctx) // fixed holds the null blocks, once
			}
			if need(f.Ctx) > free {
				t.Errorf("%d slots at ctx %d need %d B, over the %d B budget", c.slots, f.Ctx, need(f.Ctx), free)
			}
			if f.Ctx < MetalFitDefaultCtx && need(f.Ctx+metalKVPad) <= free {
				t.Errorf("%d slots: ctx %d, but %d would also fit", c.slots, f.Ctx, f.Ctx+metalKVPad)
			}
		}
	}
}

// TestMetalFitCtx_aLimitOnlyLowers: a context the load-time guard pinned (or a memory decline shrank to) caps the
// candidate and takes the floor down with it. It never raises the choice above what an unpinned load takes.
func TestMetalFitCtx_aLimitOnlyLowers(t *testing.T) {
	m, fixed, perBlock := fitCtxModel(t, Options{})
	roomy, tight := budgetFor(fixed, perBlock, 40000), budgetFor(fixed, perBlock, 6000)
	for _, c := range []struct {
		name  string
		free  int64
		lower int
		want  int
	}{
		{"a limit above the candidate changes nothing", roomy, 20000, MetalFitDefaultCtx},
		{"a limit above the ceiling changes nothing", roomy, 40000, MetalFitDefaultCtx},
		{"a limit between the default and the candidate caps it", roomy, 8192, 8192},
		{"a limit at the default", roomy, MetalCtxDefault, MetalCtxDefault},
		{"a limit under the default takes the floor down", roomy, 2000, 2000},
		{"a limit above what memory holds: memory decides", tight, 8192, 5952},
		{"a limit under the default on a tight machine", tight, 3000, 3000},
	} {
		f := m.MetalFitCtx(MetalFitInput{FreeBytes: c.free, FreeKnown: true, KVSlots: 1, Lower: c.lower})
		if f.Ctx != c.want {
			t.Errorf("%s: ctx %d (%s), want %d", c.name, f.Ctx, f.Why, c.want)
		}
		if free := m.MetalFitCtx(MetalFitInput{FreeBytes: c.free, FreeKnown: true, KVSlots: 1}); f.Ctx > free.Ctx {
			t.Errorf("%s: the limit raised the context to %d, above the unlimited %d", c.name, f.Ctx, free.Ctx)
		}
	}
}

// TestMetalFitCtx_optOuts: a paged-MoE load, --fit=off and an unreadable memory budget each keep MetalCtxDefault on a
// machine with room for the whole candidate, and say which it was. The first case is the control: without it the
// others would pass on a model that never grows.
func TestMetalFitCtx_optOuts(t *testing.T) {
	m, fixed, perBlock := fitCtxModel(t, Options{})
	off, _, _ := fitCtxModel(t, Options{DisableFit: true})
	roomy := budgetFor(fixed, perBlock, 40000)
	if f := m.MetalFitCtx(MetalFitInput{FreeBytes: roomy, FreeKnown: true, KVSlots: 2}); f.Ctx != MetalFitDefaultCtx {
		t.Fatalf("control: ctx %d (%s), want %d; the opt-out cases below prove nothing", f.Ctx, f.Why, MetalFitDefaultCtx)
	}
	for _, c := range []struct {
		name string
		m    *Model
		in   MetalFitInput
		why  string
	}{
		{"paged MoE", m, MetalFitInput{FreeBytes: roomy, FreeKnown: true, KVSlots: 2, PagedMoE: true}, "paged MoE keeps the default"},
		{"--fit=off", off, MetalFitInput{FreeBytes: roomy, FreeKnown: true, KVSlots: 2}, "--fit=off keeps the default"},
		{"unknown memory", m, MetalFitInput{FreeBytes: roomy, KVSlots: 2}, "memory could not be read"},
	} {
		f := c.m.MetalFitCtx(c.in)
		if f.Ctx != MetalCtxDefault {
			t.Errorf("%s: ctx %d, want the default %d", c.name, f.Ctx, MetalCtxDefault)
		}
		if !strings.Contains(f.Why, c.why) {
			t.Errorf("%s: reason %q does not say %q", c.name, f.Why, c.why)
		}
	}
	// A limit under the default still applies to an opted-out load: it is the guard's, not the fit's.
	if f := m.MetalFitCtx(MetalFitInput{FreeBytes: roomy, FreeKnown: true, KVSlots: 1, PagedMoE: true, Lower: 2000}); f.Ctx != 2000 {
		t.Errorf("paged MoE under a 2000-position limit: ctx %d, want 2000", f.Ctx)
	}
}

// TestMetalFitCtx_theModelsWindowCapsTheCandidate: a model whose own window is under MetalFitDefaultCtx gets its window,
// and one whose window is under MetalCtxDefault keeps the default it always had.
func TestMetalFitCtx_theModelsWindowCapsTheCandidate(t *testing.T) {
	m, fixed, perBlock := fitCtxModel(t, Options{})
	roomy := MetalFitInput{FreeBytes: budgetFor(fixed, perBlock, 40000), FreeKnown: true, KVSlots: 1}
	for _, c := range []struct{ window, want int }{{8192, 8192}, {MetalFitDefaultCtx, MetalFitDefaultCtx}, {128, MetalCtxDefault}, {0, MetalFitDefaultCtx}} {
		m.w.Cfg.MaxPositions = c.window
		if f := m.MetalFitCtx(roomy); f.Ctx != c.want {
			t.Errorf("window %d: ctx %d (%s), want %d", c.window, f.Ctx, f.Why, c.want)
		}
	}
}

// TestMetalFitCtx_aCompanionIsPriced: bytes a companion claims beside the resident (Options.ExtraResidentBytes: a
// vision tower, a drafter) come out of what the context may grow into.
func TestMetalFitCtx_aCompanionIsPriced(t *testing.T) {
	_, fixed, perBlock := fitCtxModel(t, Options{})
	extra := perBlock * 10000 / KVBlockRows
	m, _, _ := fitCtxModel(t, Options{ExtraResidentBytes: extra})
	free := budgetFor(fixed, perBlock, 16000)
	if f := m.MetalFitCtx(MetalFitInput{FreeBytes: free, FreeKnown: true, KVSlots: 1}); f.Ctx != 5952 {
		t.Errorf("a companion holding 10000 positions' worth of a 16000-position budget: ctx %d (%s), want 5952", f.Ctx, f.Why)
	}
}
