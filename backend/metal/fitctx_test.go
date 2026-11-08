//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/giw"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// fitBudget sizes a memory figure for m: the bytes at which exactly `positions` positions of KV fit beside the weights
// and decoder.MetalFitMarginBytes.
func fitBudget(t *testing.T, m *decoder.Model, positions int) int64 {
	t.Helper()
	p := m.Plan("metal", 1<<40, fit.PlanRequest{Ctx: metalCtxCapDefault, CtxPinned: true})
	if p.Placement != fit.PlacementResident {
		t.Fatalf("the tiny model does not plan resident on metal: %s", p.Reason)
	}
	kv := m.ResidentKVBytes("metal", positions, false, false)
	if kv <= 0 {
		t.Fatal("the tiny model prices no KV; every case would pass on nothing")
	}
	return decoder.MetalFitMarginBytes + p.DenseBytes + p.HostCopyBytes + p.ScratchBytes + kv
}

// TestResolveMetalCtxCap_fitByDefault: what an unpinned Metal load resolves under a forced memory figure. It grows to
// decoder.MetalFitDefaultCtx when that fits and takes what fits when it does not; KV slots trade context, an explicit
// count like Metal's default two; a load that pages MoE experts, --fit=off and an unreadable memory figure keep the
// default; an explicit -ctx is the caller's and no memory figure moves it. No Metal device is needed.
func TestResolveMetalCtxCap_fitByDefault(t *testing.T) {
	dir := wideWindowTiny(t)
	for _, c := range []struct {
		name      string
		opts      decoder.Options
		positions int // KV positions the forced budget holds; 0 forces an unknown probe
		want      int
	}{
		{"roomy: the candidate", decoder.Options{}, 40000, decoder.MetalFitDefaultCtx},
		{"room for 6000 positions, in whole blocks", decoder.Options{}, 6000, 6016},
		{"room for less than the default", decoder.Options{}, 100, metalCtxCapDefault},
		{"unknown memory", decoder.Options{}, 0, metalCtxCapDefault},
		{"--fit=off", decoder.Options{DisableFit: true}, 40000, metalCtxCapDefault},
		{"-moe-cache-experts", decoder.Options{MoECacheExperts: true}, 40000, metalCtxCapDefault},
		{"-moe-cache-slots", decoder.Options{MoECacheExperts: true, MoECacheSlots: 8}, 40000, metalCtxCapDefault},
		{"four slots asked for by the operator", decoder.Options{ResidentKVSlots: 4}, 20000, 4992},
		{"four slots as the caller's default: Metal keeps two", decoder.Options{ResidentKVSlots: 4, ResidentKVSlotsDefault: true}, 20000, 9984},
		{"sixteen slots: the default, never under it", decoder.Options{ResidentKVSlots: 16}, 20000, metalCtxCapDefault},
		{"-ctx 8192 on a roomy machine", decoder.Options{ResidentContext: 8192}, 40000, 8192},
		{"-ctx 32768 on a tight machine: untouched", decoder.Options{ResidentContext: 32768, ResidentKVSlots: 4}, 100, 32768},
		{"-ctx 2048 on a roomy machine: untouched", decoder.Options{ResidentContext: 2048}, 40000, 2048},
		{"-ctx 8192 with --fit=off", decoder.Options{ResidentContext: 8192, DisableFit: true}, 40000, 8192},
	} {
		t.Run(c.name, func(t *testing.T) {
			c.opts.Quant = "int8int8"
			m, err := decoder.Load(dir, c.opts)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			free, known := int64(0), false
			if c.positions > 0 {
				free, known = fitBudget(t, m, c.positions), true
			}
			defer decoder.SetMemoryProbeForTest("metal", free, known)()
			got, err := resolveMetalCtxCap(m)
			if err != nil || got != c.want {
				t.Errorf("resolveMetalCtxCap = %d, %v; want %d (%s)", got, err, c.want, metalFitFor(m).Why)
			}
		})
	}
}

// TestMetalBuildResident_fitByDefault builds the resident a default serve load asks for (four KV slots as the caller's
// default, no -ctx) under two memory figures. Roomy, it holds decoder.MetalFitDefaultCtx positions in two slots, says so
// in its ContextNote, and decodes a prompt of a coding agent's first-request length, which the old 4096 default refused.
// With room for 20000 positions of KV it holds two slots of 10000. In both, the context is the one `goinfer-chat fit`
// computes for the same figure (decoder.Model.MetalFitCtx with Metal's default slots) as each slot's share, and one conversation's
// cap is the pool the two shares make.
func TestMetalBuildResident_fitByDefault(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := wideWindowTiny(t)
	probe, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", Backend: "metal"})
	if err != nil {
		t.Fatal(err)
	}
	roomy, interior := fitBudget(t, probe, 40000), fitBudget(t, probe, 20000)
	probe.Close()
	live := metalLiveAvailable
	t.Cleanup(func() { metalLiveAvailable = live })
	for _, c := range []struct {
		name    string
		budget  int64
		wantCtx int
		prompt  int
	}{
		{"roomy", roomy, decoder.MetalFitDefaultCtx, 11137},
		{"room for 20000 positions", interior, 9984, 6000},
	} {
		t.Run(c.name, func(t *testing.T) {
			metalLiveAvailable = func() int64 { return c.budget } // the guard's budget and the probe's: both read it
			m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", Backend: "metal", ResidentKVSlots: 4, ResidentKVSlotsDefault: true})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Fatalf("the build declined: %s", m.ResidentDecline())
			}
			// The fit sizes each slot's share; one conversation's cap is the pool the shares make together (kvRaise).
			pool := metalDefaultKVSlots * kvBlocksFor(c.wantCtx) * decoder.KVBlockRows
			if got := a.ContextCap(); a.r.kvShareCtx != c.wantCtx || got != min(pool, metalCtxCapMax) {
				t.Fatalf("share %d, ContextCap %d; want %d and the pool's %d", a.r.kvShareCtx, got, c.wantCtx, min(pool, metalCtxCapMax))
			}
			if got := a.r.kvSlotCount(); got != metalDefaultKVSlots {
				t.Errorf("%d KV slots, want Metal's default %d", got, metalDefaultKVSlots)
			}
			wantNote := decoder.MetalFitWhy(metalDefaultKVSlots, m.MetalKVBytes(metalDefaultKVSlots, c.wantCtx))
			if got := m.ResidentContextNote(); got != wantNote || !strings.HasPrefix(got, "fit: 2 slots, ") {
				t.Errorf("ContextNote = %q, want %q", got, wantNote)
			}
			fit := m.MetalFitCtx(decoder.MetalFitInput{FreeBytes: c.budget, FreeKnown: true, KVSlots: decoder.MetalDefaultKVSlots})
			if fit.Ctx != a.r.kvShareCtx {
				t.Errorf("fit plans ctx %d for this memory figure; the load's share is %d", fit.Ctx, a.r.kvShareCtx)
			}
			// The KV the fit priced is the KV the build leased: a ledger line rounds each lease up by under a page, no more.
			nL := m.Config().NumLayers
			charged := metalLedgerOf(a.r.d).Usage().ByLine[mem.LineKV]
			if over := charged - fit.KVBytes; over < 0 || over > int64(2*nL+1)*16<<10 {
				t.Errorf("fit prices %d B of KV for %d slots, and the build leased %d B", fit.KVBytes, metalDefaultKVSlots, charged)
			}
			vocab := m.Config().VocabSize
			prompt := make([]int, c.prompt)
			for i := range prompt {
				prompt[i] = (i*7 + 3) % vocab
			}
			out, gen := m.Generate(context.Background(), prompt, 2, decoder.SamplingParams{Temperature: 0})
			n := 0
			for range out {
				n++
			}
			if err := gen.Err(); err != nil || n == 0 {
				t.Errorf("a %d-token prompt in a %d-position context: %d tokens, err %v", c.prompt, c.wantCtx, n, err)
			}
		})
	}
}

// TestMetalBuildResident_pagedMoEKeepsTheDefault builds a tiny MoE model twice on a machine with room for the whole
// candidate. Every expert resident, it grows to decoder.MetalFitDefaultCtx like a dense model (the control: this model
// can grow). Asked to page its experts, it keeps metalCtxCapDefault and its ContextNote says why: the pager lives on
// the memory a longer context would take.
func TestMetalBuildResident_pagedMoEKeepsTheDefault(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := t.TempDir()
	writeMoEIdentical(t, dir, genTinyWeights(rand.New(rand.NewSource(5))))
	widenWindow(t, dir)
	live := metalLiveAvailable
	metalLiveAvailable = func() int64 { return 8 << 30 }
	t.Cleanup(func() { metalLiveAvailable = live })
	for _, c := range []struct {
		name     string
		opts     decoder.Options
		wantCtx  int
		wantNote string
	}{
		{"every expert resident", decoder.Options{}, decoder.MetalFitDefaultCtx, "fit: 1 slot, "},
		{"-moe-cache-experts -moe-cache-slots", decoder.Options{MoECacheExperts: true, MoECacheSlots: max(tmTopK, 2)}, metalCtxCapDefault, "paged MoE keeps the default"},
	} {
		c.opts.Quant, c.opts.Backend = "int4", "metal"
		m, err := decoder.Load(dir, c.opts)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("%s: the build declined: %s", c.name, m.ResidentDecline())
		}
		if got := a.ContextCap(); got != c.wantCtx {
			t.Errorf("%s: ContextCap = %d, want %d", c.name, got, c.wantCtx)
		}
		if got := m.ResidentContextNote(); !strings.HasPrefix(got, c.wantNote) {
			t.Errorf("%s: ContextNote = %q, want it to start %q", c.name, got, c.wantNote)
		}
		m.Close()
	}
}

// TestMetalFit_agreesWithSlotPricing: the context MetalFitCtx picks for S slots and the slots metalKVSlots grants at that context price the
// same KV. A budget of exactly the fit's bytes grants all S, and one byte less grants S-1.
func TestMetalFit_agreesWithSlotPricing(t *testing.T) {
	dir := wideWindowTiny(t)
	for _, slots := range []int{2, 4} {
		t.Run(strconv.Itoa(slots), func(t *testing.T) {
			m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", ResidentKVSlots: slots})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			defer decoder.SetMemoryProbeForTest("metal", fitBudget(t, m, 20000), true)()
			fit := metalFitFor(m)
			ctx, err := resolveMetalCtxCap(m)
			if err != nil || ctx != fit.Ctx || ctx <= metalCtxCapDefault {
				t.Fatalf("ctx %d (%v), fit %d: want them equal and past the default %d", ctx, err, fit.Ctx, metalCtxCapDefault)
			}
			// The guard adds every slot's table at the raised cap to its fixed need (metalKVSlots), as the fit adds it to the load's KV.
			need, each, raise := residentNeedBytes(m), metalSlotBytes(m), metalRaisedTableBytes(m, slots)
			further := fit.KVBytes - m.ResidentKVBytes("metal", ctx, false, false)
			if raise <= 0 || further != int64(slots-1)*each+raise {
				t.Errorf("the fit prices %d B past the first slot, metalSlotBytes %d B each and the raised tables %d B", further, each, raise)
			}
			if got := kvSlotsWithin(slots, need+further, need+raise, each); got != slots {
				t.Errorf("a budget of exactly the fit's bytes grants %d slots", got)
			}
			if got := kvSlotsWithin(slots, need+further-1, need+raise, each); got != slots-1 {
				t.Errorf("a byte under the fit's bytes grants %d slots, want %d", got, slots-1)
			}
		})
	}
}

// TestMetalLoadPlan_isTheGuardsPrice: fit's Metal line (decoder.Model.MetalLoadPlan, on the file opened as fitcmd's fitLoadOptions
// opens it) is the build's own price. The build is given a budget (its memory ceiling, and the Metal probe an unpinned context is fit
// to), and MetalLoadPlan the same one. Its NeedBytes equals the build guard's, to the byte: residentNeedBytes, plus the raised block
// tables of every slot asked (metalRaisedTableBytes), plus metalSlotBytes for each slot kept after the first. Its slot count is the
// build's (kvSlotCount) and the banner's (decoder.Model.ResidentKVSlots). Each case runs at a budget with room for every slot and
// at one a byte short of the second. The cases: a dense model pinned within its window and past it (which the build clamps to the
// window), and unpinned (which raises the tables); the metal sidecar of testdata/phi3-q4k-tiny.gguf; and a recurrent model's metal
// .giw asked for 2 slots (whose further slot carries its state). The .giw cases read canonical int4, as the build holds it. With
// GOINFER_HEAVY_TESTS=1 the Phi-3 mini q4 and Qwen2.5-Coder 0.5B q4_k_m sidecars in ~/models are two more, and a missing one fails.
func TestMetalLoadPlan_isTheGuardsPrice(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/phi3-q4k-tiny.gguf")
	if err != nil {
		t.Fatal(err)
	}
	src := filepath.Join(t.TempDir(), "phi3.gguf")
	if err := os.WriteFile(src, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	sidecar := decoder.SidecarPath(src, "q4k", "metal", false)
	if err := decoder.TranscodeGIW(context.Background(), src, sidecar, "q4k", false, decoder.GIWTargetForBackend("metal")); err != nil {
		t.Fatal(err)
	}
	type guardCase struct {
		name, path, quant string
		ctx, slots        int // ctx 0 is unpinned; slots 0 is a serve load's default request, else an explicit count
	}
	q35, err := decoder.Load("../../testdata/qwen35-tiny", decoder.Options{Quant: "int4"}) // canonical int4, for a metal-target .giw
	if err != nil {
		t.Fatal(err)
	}
	blob, err := decoder.SerializeWeightsForTarget(q35.Weights(), "qwen35-tiny", decoder.GIWTargetMetal)
	q35.Close()
	if err != nil {
		t.Fatal(err)
	}
	q35GIW := filepath.Join(t.TempDir(), "qwen35-tiny.int4.metal.giw")
	if err := os.WriteFile(q35GIW, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []guardCase{
		{"llama-tiny int8int8", "../../testdata/llama-tiny", "int8int8", 128, 0},
		{"llama-tiny, -ctx past its window", "../../testdata/llama-tiny", "int8int8", 256, 0},
		{"wide-window llama-tiny, unpinned", wideWindowTiny(t), "int8int8", 0, 0},
		{"phi3-q4k-tiny's metal sidecar", sidecar, "q4k", 256, 0},
		{"qwen35-tiny's metal .giw asked for 2 slots", q35GIW, "int4", 64, 2},
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		home := os.Getenv("HOME")
		cases = append(cases,
			guardCase{"Phi-3 mini q4", filepath.Join(home, "models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.q4k.metal.v17.giw"), "q4k", 2048, 0},
			guardCase{"Qwen2.5-Coder 0.5B q4_k_m", filepath.Join(home, "models/qwen2.5-coder-0.5b-instruct-q4_k_m.int4.metal.v17.giw"), "int4", 2048, 0})
	}
	slotOpts := func(c guardCase, o decoder.Options) decoder.Options {
		o.ResidentContext = c.ctx
		if c.slots > 0 {
			o.ResidentKVSlots = c.slots
		} else {
			o.ResidentKVSlots, o.ResidentKVSlotsDefault = 4, true // serve's -kv-sessions default
		}
		return o
	}
	// build loads c on Metal at budget and returns the slots the build keeps and the banner reads, the guard's price for them, and the
	// fixed need and per-slot cost the guard's slot decision uses.
	build := func(t *testing.T, c guardCase, budget int64) (kept, banner int, guard, base, perSlot int64) {
		t.Helper()
		defer func(was int64) { metalMemoryCeilingForTest = was }(metalMemoryCeilingForTest)
		metalMemoryCeilingForTest = budget
		defer decoder.SetMemoryProbeForTest("metal", budget, true)()
		m, err := decoder.Load(c.path, slotOpts(c, decoder.Options{Backend: "metal", Quant: c.quant, KVPrecision: "auto"}))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("not resident on metal at a budget of %d B: %s", budget, m.DecodePath())
		}
		kept, banner = rf.(*metalResident).r.kvSlotCount(), m.ResidentKVSlots()
		base, perSlot = residentNeedBytes(m)+metalRaisedTableBytes(m, metalKVSlotsWanted(m)), metalSlotBytes(m)
		return kept, banner, base + int64(kept-1)*perSlot, base, perSlot
	}
	plan := func(t *testing.T, c guardCase, budget int64) decoder.MetalLoad {
		t.Helper()
		fm, err := decoder.Load(c.path, slotOpts(c, decoder.Options{Quant: c.quant, DirectLoad: true}))
		if err != nil {
			t.Fatal(err)
		}
		defer fm.Close()
		return fm.MetalLoadPlan(budget, fit.PlanRequest{Ctx: c.ctx, CtxPinned: c.ctx > 0})
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.path); err != nil {
				t.Fatalf("%v (a metal sidecar is built by a Metal serve of its .gguf)", err)
			}
			const roomy = int64(1) << 40
			kept, _, _, base, perSlot := build(t, c, roomy)
			budgets := []struct {
				name   string
				budget int64
			}{{"room for every slot", roomy}}
			if kept > 1 {
				// A byte short of a second slot at this budget's context: an unpinned load is fit to the budget, so the tight budget
				// is priced at the context it gives (the floor).
				tight := base + perSlot - 1
				if c.ctx == 0 {
					_, _, _, b, s := build(t, c, base) // at a budget of one slot's need the context falls to the floor
					tight = b + s - 1
				}
				budgets = append(budgets, struct {
					name   string
					budget int64
				}{"a byte short of a second slot", tight})
			}
			for _, b := range budgets {
				kept, banner, guard, _, _ := build(t, c, b.budget)
				l := plan(t, c, b.budget)
				if l.Plan.Placement != fit.PlacementResident || l.Plan.NeedBytes() != guard {
					t.Errorf("%s: fit plans %v needing %d B (%s); the build's guard prices %d B", b.name, l.Plan.Placement, l.Plan.NeedBytes(),
						l.Plan.Reason, guard)
				}
				if l.KVSlots != kept || banner != kept {
					t.Errorf("%s: KV slots: fit %d, the build %d, the banner %d", b.name, l.KVSlots, kept, banner)
				}
				t.Logf("%s, %s: the build keeps %d KV slots, and its guard prices %d B (%.1f MiB)", c.name, b.name, kept, guard, float64(guard)/(1<<20))
			}
		})
	}
}
