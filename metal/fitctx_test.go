//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// fitBudget sizes a memory figure for m: the bytes at which exactly `positions` positions of KV fit beside the weights
// and decoder.MetalFitMarginBytes.
func fitBudget(t *testing.T, m *decoder.Model, positions int) int64 {
	t.Helper()
	p := m.Plan("metal", 1<<40, decoder.PlanRequest{Ctx: metalCtxCapDefault, CtxPinned: true})
	if p.Placement != decoder.PlacementResident {
		t.Fatalf("the tiny model does not plan resident on metal: %s", p.Reason)
	}
	perPos := m.ResidentKVBytes("metal", 8, false, false) / 8
	if perPos <= 0 {
		t.Fatal("the tiny model prices no KV; every case would pass on nothing")
	}
	return decoder.MetalFitMarginBytes + p.DenseBytes + p.HostCopyBytes + perPos*int64(positions)
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
		{"room for 6000 positions", decoder.Options{}, 6000, 6000},
		{"room for less than the default", decoder.Options{}, 100, metalCtxCapDefault},
		{"unknown memory", decoder.Options{}, 0, metalCtxCapDefault},
		{"--fit=off", decoder.Options{DisableFit: true}, 40000, metalCtxCapDefault},
		{"-moe-cache-experts", decoder.Options{MoECacheExperts: true}, 40000, metalCtxCapDefault},
		{"-moe-cache-slots", decoder.Options{MoECacheExperts: true, MoECacheSlots: 8}, 40000, metalCtxCapDefault},
		{"four slots asked for by the operator", decoder.Options{ResidentKVSlots: 4}, 20000, 5000},
		{"four slots as the caller's default: Metal keeps two", decoder.Options{ResidentKVSlots: 4, ResidentKVSlotsDefault: true}, 20000, 10000},
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
// computes for the same figure (decoder.Model.MetalFitCtx with Metal's default slots).
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
		{"room for 20000 positions", interior, 10000, 6000},
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
			if got := a.ContextCap(); got != c.wantCtx {
				t.Fatalf("ContextCap = %d, want %d", got, c.wantCtx)
			}
			if got := a.r.kvSlotCount(); got != metalDefaultKVSlots {
				t.Errorf("%d KV slots, want Metal's default %d", got, metalDefaultKVSlots)
			}
			wantNote := decoder.MetalFitWhy(metalDefaultKVSlots, int64(metalDefaultKVSlots)*m.ResidentKVBytes("metal", c.wantCtx, false, false))
			if got := m.ResidentContextNote(); got != wantNote || !strings.HasPrefix(got, "fit: 2 slots, ") {
				t.Errorf("ContextNote = %q, want %q", got, wantNote)
			}
			fit := m.MetalFitCtx(decoder.MetalFitInput{FreeBytes: c.budget, FreeKnown: true, KVSlots: decoder.MetalDefaultKVSlots})
			if fit.Ctx != a.ContextCap() {
				t.Errorf("fit plans ctx %d for this memory figure; the load took %d", fit.Ctx, a.ContextCap())
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
