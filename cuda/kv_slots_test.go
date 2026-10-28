//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/mem"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"sort"
	"strings"
)

// kvSlotTurn is one generation of the MC1 identity scenario: its token ids and PrefillReused.
type kvSlotTurn struct {
	ids    []int
	reused int
}

// kvSlotsScenario is MC1's identity gate on CUDA (docs/tasks/task-concurrency-2026-09.md; the Metal twin is
// TestMetalKVSlots_interleavedMatchesAlone): with two resident KV slots, two conversations that share only a lead,
// interleaved turn by turn through the production Generate path, must each emit exactly the token ids they emit
// when served alone, and reuse exactly as much of their own history on every turn. A control — the same interleaving
// on one slot — must thrash, so the scenario can see the failure it guards against. With GOINFER_CUDA_GRAPHS set it
// also requires the 2-slot resident to have kept graphs on, so a graphs run cannot quietly pass on the live path.
func kvSlotsScenario(t *testing.T, dir string, opts decoder.Options, maxTok, base int) {
	t.Helper()
	lead := []int{base + 1, base + 2, base + 3} // the shared "chat template" lead
	first := map[string][]int{
		"A": append(slices.Clone(lead), base+10, base+11, base+12, base+13),
		"B": append(slices.Clone(lead), base+20, base+21, base+22, base+23),
	}
	run := func(m *decoder.Model, order []string) map[string][]kvSlotTurn {
		prompts := map[string][]int{"A": slices.Clone(first["A"]), "B": slices.Clone(first["B"])}
		out := map[string][]kvSlotTurn{}
		for _, c := range order {
			ch, gen := m.Generate(context.Background(), prompts[c], maxTok, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("%s turn %d: %v", c, len(out[c])+1, err)
			}
			out[c] = append(out[c], kvSlotTurn{ids, gen.PrefillReused})
			// the next user turn extends the conversation with its reply and two new ids
			prompts[c] = append(append(slices.Clone(prompts[c]), ids...), base+30+len(out[c]), base+40+len(out[c]))
		}
		return out
	}
	load := func(slots int) *decoder.Model {
		o := opts
		o.Backend, o.ResidentKVSlots = "cuda", slots
		m, err := decoder.Load(dir, o)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
			m.Close()
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		if got := m.ResidentKVSlots(); got != slots {
			m.Close()
			t.Fatalf("ResidentKVSlots() = %d with %d requested — the resident did not allocate them", got, slots)
		}
		return m
	}
	interleaved := []string{"A", "B", "A", "B", "A", "B"}

	mTwo := load(2)
	if os.Getenv("GOINFER_CUDA_GRAPHS") != "" {
		if r := mTwo.ResidentForwardForTest().(*cudaResident); !r.graphs {
			mTwo.Close()
			t.Fatal("GOINFER_CUDA_GRAPHS was set but the resident runs live — this run would not exercise graph replay")
		}
	}
	inter := run(mTwo, interleaved)
	mTwo.Close()
	alone := map[string][]kvSlotTurn{}
	for _, c := range []string{"A", "B"} { // each conversation alone on a fresh one-slot model
		m := load(1)
		alone[c] = run(m, []string{c, c, c})[c]
		m.Close()
	}
	mOne := load(1)
	thrash := run(mOne, interleaved) // the control
	mOne.Close()

	for _, c := range []string{"A", "B"} {
		for i := range alone[c] {
			a, b := alone[c][i], inter[c][i]
			if !slices.Equal(a.ids, b.ids) {
				t.Errorf("%s turn %d: interleaved on 2 slots emitted %v, alone %v — not bit-identical", c, i+1, b.ids, a.ids)
			}
			if a.reused != b.reused {
				t.Errorf("%s turn %d: interleaved on 2 slots reused %d, alone %d", c, i+1, b.reused, a.reused)
			}
			if i > 0 && b.reused <= len(lead) {
				t.Errorf("%s turn %d: reused only %d (the lead is %d) — its own history was not kept", c, i+1, b.reused, len(lead))
			}
			if i > 0 && thrash[c][i].reused > len(lead) {
				t.Errorf("control: %s turn %d reused %d on ONE slot interleaved — the control no longer thrashes, so "+
					"this test cannot see a slot regression", c, i+1, thrash[c][i].reused)
			}
		}
		t.Logf("%s: ids per turn alone %v; reused per turn alone %v, interleaved on 2 slots %v, interleaved on 1 slot %v",
			c, idsOf(alone[c]), reusedOf(alone[c]), reusedOf(inter[c]), reusedOf(thrash[c]))
	}
}

func reusedOf(ts []kvSlotTurn) []int {
	out := make([]int, len(ts))
	for i, t := range ts {
		out[i] = t.reused
	}
	return out
}

func idsOf(ts []kvSlotTurn) [][]int {
	out := make([][]int, len(ts))
	for i, t := range ts {
		out[i] = t.ids
	}
	return out
}

// TestCUDAKVSlots_interleavedMatchesAlone runs kvSlotsScenario on one tiny fixture per KV layout the resident
// allocates: dense (llama-tiny), sliding window (mistral-tiny-window), and MLA (deepseek-tiny: one latent buffer per
// layer and no V). GOINFER_CUDA_KVSLOTS_MODEL=<checkpoint> runs it on a real checkpoint instead (int4, 48 tokens per
// turn, prompts of ids from a fixed range rather than text).
func TestCUDAKVSlots_interleavedMatchesAlone(t *testing.T) {
	if p := os.Getenv("GOINFER_CUDA_KVSLOTS_MODEL"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("GOINFER_CUDA_KVSLOTS_MODEL=%s: %v", p, err)
		}
		requireCUDADevice(t)
		kvSlotsScenario(t, p, decoder.Options{Quant: "int4"}, 48, 1000)
		return
	}
	for _, f := range []string{"llama-tiny", "mistral-tiny-window", "deepseek-tiny"} {
		t.Run(f, func(t *testing.T) {
			dir := filepath.Join("..", "testdata", f)
			requireDeviceAndFixture(t, dir)
			kvSlotsScenario(t, dir, decoder.Options{Quant: "int4"}, 8, 0)
		})
	}
}

// TestCUDAKVSlots_graphs is the scenario with CUDA graphs forced on (GOINFER_CUDA_GRAPHS + _UNSAFE: this box runs the
// Default compute mode, where the tenancy gate would otherwise decline them). A captured segment bakes in its buffer
// arguments, so a slot switch is safe under graphs only because segA/B/C touch no KV — rope_kv and attention run
// live and bind the bound slot's buffers when issued. This pins that: a KV buffer captured into a segment would make
// a switched-in slot read or write the previous slot's cache, and the interleaved ids would diverge.
func TestCUDAKVSlots_graphs(t *testing.T) {
	dir := filepath.Join("..", "testdata", "mistral-tiny-window")
	requireDeviceAndFixture(t, dir)
	t.Setenv("GOINFER_CUDA_GRAPHS", "1")
	t.Setenv("GOINFER_CUDA_GRAPHS_UNSAFE", "1")
	kvSlotsScenario(t, dir, decoder.Options{Quant: "int4"}, 8, 0)
}

// TestCUDAKVSlots_recurrentKeepsOne confirms nothing on CUDA bypasses the decoder's one-slot rule for a family with
// recurrent state: a Gated-DeltaNet hybrid asked for 4 slots allocates 1, and generations choose among 1.
func TestCUDAKVSlots_recurrentKeepsOne(t *testing.T) {
	dir := filepath.Join("..", "testdata", "qwen35-tiny")
	requireDeviceAndFixture(t, dir)
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
	}
	if r.dnet == nil {
		t.Fatal("qwen35-tiny built without DeltaNet layers — this test no longer covers a recurrent family")
	}
	if len(r.kvTabs) != 1 || r.KVSlots() != 1 || m.ResidentKVSlots() != 1 {
		t.Errorf("recurrent resident: %d slot tables, KVSlots %d, model reports %d — want one slot",
			len(r.kvTabs), r.KVSlots(), m.ResidentKVSlots())
	}
}

// kvSlotsProbe is one armed cudaFreeVRAM stub of TestCUDAKVSlots_pricedAgainstWhatIsLeft: it reports live minus
// the real bytes the device has allocated since it was armed (or the real free figure, live == 0), and records what
// it saw at the build's first read, checkKVFits'. A later read (the prefill width, after the KV) is answered the same
// way and recorded only in calls.
type kvSlotsProbe struct {
	free0   uint64   // the real free VRAM when the probe was armed
	live    int64    // the fictional free figure the probe falls from; 0: report the real one
	sawNeed int64    // one slot's KV at the probed resident's context, as the pools charge it
	sawKV   [4]int64 // the pools' charged bytes for 0..3 slots at that context
	sawRes  int64    // the build's context margin
	sawDrop int64    // real bytes allocated between arming and the probe call
	sawFree int64    // what the probe returned
	calls   int
}

// requireCUDADevice skips without a CUDA device (requireDeviceAndFixture without the fixture).
func requireCUDADevice(t *testing.T) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
}

// TestCUDAKVSlots_clampedBuild runs a real build whose free VRAM is stubbed to hold exactly 2 slots' KV beside the
// reserve, with 4 requested: the build must allocate 2, report 2, and generate on both. This is the clamping load
// Metal never ran; serve's banner then names the clamp (internal/serveapp's banner test covers the wording).
func TestCUDAKVSlots_clampedBuild(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-tiny")
	requireDeviceAndFixture(t, dir)
	orig := cudaFreeVRAM
	t.Cleanup(func() { cudaFreeVRAM = orig })
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		two, three := kvChargedBytes(r.ctxCap, 2, r.layers), kvChargedBytes(r.ctxCap, 3, r.layers)
		return uint64((two+three)/2 + r.extraBytes + r.ctxMargin()), nil
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 4})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
	}
	if len(r.kvTabs) != 2 || m.ResidentKVSlots() != 2 {
		t.Fatalf("clamped build: %d slot tables, model reports %d slots — want 2 of 4", len(r.kvTabs), m.ResidentKVSlots())
	}
	for l := range r.layers {
		if want := r.kvPoolBlocks(r.kvGroupOf(l), 2) * kvBlock * r.layers[l].kvDim; r.kc[l].Len() != want {
			t.Errorf("layer %d: K pool holds %d floats, want two slots' blocks and the null block = %d", l, r.kc[l].Len(), want)
		}
	}
	if err := r.UseKVSlot(2); err == nil {
		t.Error("UseKVSlot(2) on a 2-slot resident did not fail")
	}
	for _, s := range []int{1, 0, 1} {
		if err := r.UseKVSlot(s); err != nil {
			t.Fatalf("UseKVSlot(%d): %v", s, err)
		}
		if r.kvSlot != s || r.kvTabSlot != s {
			t.Errorf("after UseKVSlot(%d) the bound tables are slot %d's (read: %d)", s, r.kvSlot, r.kvTabSlot)
		}
		if _, err := r.Forward(m.EmbedResidentForTest(5), 0); err != nil {
			t.Fatalf("forward on slot %d: %v", s, err)
		}
	}
}

// TestCUDAKVSlots_pricedAgainstWhatIsLeft pins how the slot count is priced — the CUDA side of the Metal double-count
// bug, where a weights-inclusive base was compared with a live figure the weights had already left. The
// free-VRAM probe is stubbed to fall by exactly what the device really allocates from a fictional starting figure,
// chosen (from a calibration build) so that precisely `want` slots fit beside everything the build puts on the device
// before its KV. One more slot than that is requested, so the test is two-sided:
//   - pricing that counts the weights twice (a weights-inclusive base against the post-weights probe) grants fewer
//     than `want`;
//   - pricing KV alone against a figure read before the weights are on the device grants `want`+1, which on a real
//     card would not fit.
//
// A double count is visible only where the build's pre-KV bytes exceed half a slot, so a fixture where they do not
// skips with the numbers. GOINFER_CUDA_KVSLOTS_MODEL runs it on a real checkpoint.
func TestCUDAKVSlots_pricedAgainstWhatIsLeft(t *testing.T) {
	dir, quant := os.Getenv("GOINFER_CUDA_KVSLOTS_MODEL"), "int4"
	if dir == "" {
		dir = filepath.Join("..", "testdata", "llama-tiny")
		requireDeviceAndFixture(t, dir)
	} else {
		requireCUDADevice(t)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	realFree := func() uint64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return f
	}
	orig := cudaFreeVRAM
	t.Cleanup(func() { cudaFreeVRAM = orig })

	// build loads with the probe armed at `live` (0: report the real figure) and returns what the probe saw.
	build := func(live int64, slots int) (granted int, p kvSlotsProbe) {
		p = kvSlotsProbe{free0: realFree(), live: live}
		cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
			p.calls++
			now := realFree()
			if p.calls > 1 {
				if p.live > 0 {
					return uint64(p.live - (int64(p.free0) - int64(now))), nil
				}
				return now, nil
			}
			p.sawNeed, p.sawDrop = kvChargedBytes(r.ctxCap, 1, r.layers), int64(p.free0)-int64(now)
			for n := 1; n < len(p.sawKV); n++ {
				p.sawKV[n] = kvChargedBytes(r.ctxCap, n, r.layers)
			}
			p.sawRes = r.ctxMargin()
			p.sawFree = int64(now)
			if p.live > 0 {
				p.sawFree = p.live - p.sawDrop
			}
			return uint64(p.sawFree), nil
		}
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: quant, ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		if p.calls == 0 {
			t.Fatal("the free-VRAM probe never ran during the build; checkKVFits reads it first")
		}
		return r.KVSlots(), p
	}

	_, cal := build(0, 1) // calibration: what the build allocates before its KV, and one slot's KV
	const want = 2
	reserve := cal.sawRes // no companion attach in this test
	if cal.sawDrop < cal.sawNeed/2 {
		t.Skipf("the build allocates %d B before its KV against %d B per slot: a double count of that could not "+
			"change the slot count here (run with GOINFER_CUDA_KVSLOTS_MODEL on a real checkpoint)", cal.sawDrop, cal.sawNeed)
	}
	live := cal.sawDrop + reserve + (cal.sawKV[want]+cal.sawKV[want+1])/2
	got, p := build(live, want+1)
	t.Logf("pre-KV bytes %d (calibration) / %d (this build), one slot %d B, probe returned %d B: %d of %d slots granted",
		cal.sawDrop, p.sawDrop, p.sawNeed, p.sawFree, got, want+1)
	if got != want {
		t.Errorf("granted %d slots where exactly %d fit beside what the build had already allocated: fewer means the "+
			"price counts the weights again; more means it was read before they were on the device", got, want)
	}
}

// TestResolveCtxCapFit_slotsShrinkTheContext pins the owner's decision (docs/tasks/task-concurrency-2026-09.md
// MC1 on CUDA): an unpinned load that asks for N resident KV slots gives up context until all N fit, instead of taking
// the one-slot context and clamping the slots. The free-VRAM budget is forced (ExtraResidentBytes, from a real probe,
// the way TestResolveCtxCapFit_agreesWithCheckKVFits does it) so that one slot fits the full candidate while 4 fit only
// at about 5000 positions. It checks the planning-time choice against the build-time one: at the chosen context
// checkKVFits must grant all 4 slots, and at the one-slot context it must not — the thrash the decision removes. A
// budget where not even the floor holds 4 must return the floor, and an explicit -ctx must be untouched.
func TestResolveCtxCapFit_slotsShrinkTheContext(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-tiny")
	requireDeviceAndFixture(t, dir)
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no cuda device: %v", err)
	}
	defer dev.ReleaseObjects()
	probe, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load (probe): %v", err)
	}
	_, nLayers, _, nKV, hd, _, _ := probe.Dims()
	layers := make([]cudaLayer, nLayers)
	for i := range layers {
		layers[i].kvDim = nKV * hd
	}
	dense := probe.ResidentDenseWeightBytesFor("cuda") // what Plan prices for CUDA
	probe.Close()
	free, ok := decoder.FreeBytesFor("cuda")
	if !ok {
		t.Skip("no cuda free-bytes probe available")
	}
	const slots = 4
	for _, c := range []struct {
		name      string
		target    int // the context at which exactly `slots` slots' KV fits the forced budget
		want      func(got int) bool
		wantSlots int // what checkKVFits grants at the chosen context
	}{
		{"interior: 4 slots fit at ~5000", 5000, func(got int) bool { return got > 4800 && got <= 5064 }, slots},
		{"not even the floor holds 4: the floor, and the build clamps", 3000, func(got int) bool { return got == cudaCtxCapDefault }, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			// The budget holds exactly `slots` slots at the target as the pools charge them (kvChargedBytes): on this fixture the
			// 2 MiB rounding of each pool buffer is a large share of the KV, so a linear budget would not hold them at all.
			round := kvChargedBytes(c.target, slots, layers) - (int64(slots)*kvBytesForCap(c.target, layers) + kvPoolNullBytes(layers))
			forced := free - testCtxMargin - dense - kvChargedBytes(c.target, slots, layers)
			if forced <= 0 {
				t.Skipf("free VRAM %d B too small to force this budget", free)
			}
			m, err := decoder.Load(dir, decoder.Options{Quant: "int4", ExtraResidentBytes: forced, ResidentKVSlots: slots})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			// A GENUINE pin (item 28, docs/prompts/nobara-mc1-webgpu-2026-09.md §4): ResidentContext set
			// in Options, not just a bare 8192 handed to resolveCtxCapFit with an otherwise-unpinned m —
			// the latter is now a guard-pin scenario (m.ResidentContextPinned() false) and correctly runs
			// fit-by-default clamped to it, which would shrink for these slots, not the "never shrinks"
			// invariant this case exists to check.
			mPinned, err := decoder.Load(dir, decoder.Options{Quant: "int4", ExtraResidentBytes: forced, ResidentKVSlots: slots, ResidentContext: 8192})
			if err != nil {
				t.Fatalf("Load (pinned): %v", err)
			}
			defer mPinned.Close()
			// Plan prices the KV linearly; it reserves the rounding at the target beside the margin, the exact figure where the build
			// reserves its upper bound (kvRoundingBound), which on this fixture exceeds the whole KV.
			one, _ := resolveCtxCapFitSlack(m, 0, 1<<20, 1, 0, fixedMargin(testCtxMargin), round)
			got, _ := resolveCtxCapFitSlack(m, 0, 1<<20, slots, 0, fixedMargin(testCtxMargin), round)
			pinned, _ := resolveCtxCapFitSlack(mPinned, mPinned.ResidentContextRequest(), 1<<20, slots, 0, fixedMargin(testCtxMargin), round)
			t.Logf("one slot: ctx %d; %d slots: ctx %d (target %d); explicit 8192 with %d slots: %d", one, slots, got, c.target, slots, pinned)
			// One slot gets the smaller of the candidate and what the forced budget holds (slots*target positions of KV beside the weights).
			// With the candidate at 8192 that was always the candidate; at 16384 the 3000-position case holds only ~12000, which is the
			// budget talking, not a shrink the slots rule did.
			room := slots * c.target
			wantOne := min(room, fitDefaultCtx)
			if d := one - wantOne; d < -wantOne/50 || d > wantOne/50 {
				t.Fatalf("one slot chose ctx %d, want ~%d (the candidate %d or the forced budget's %d positions, whichever is less)", one, wantOne, fitDefaultCtx, room)
			}
			if one <= got {
				t.Fatalf("one slot (ctx %d) did not get more context than %d slots (ctx %d) — the slots rule shrank nothing, so this case shows no trade", one, slots, got)
			}
			if !c.want(got) {
				t.Errorf("%d slots chose ctx %d, target %d", slots, got, c.target)
			}
			if pinned != 8192 {
				t.Errorf("an explicit -ctx 8192 became %d with slots requested — a pinned context must never shrink", pinned)
			}
			grant := func(ctx int, pinned bool) (int, int) {
				r := &cudaResident{dev: dev, ctxCap: ctx, ctxExplicit: pinned, layers: layers, kvSlotsReq: slots, extraBytes: m.ExtraResidentBytes(), ctxMarginBytes: testCtxMargin}
				if err := r.checkKVFits(); err != nil {
					t.Fatalf("checkKVFits at ctx %d: %v", ctx, err)
				}
				return r.kvSlotsN, r.ctxCap
			}
			if g, _ := grant(got, false); g != c.wantSlots {
				t.Errorf("at the chosen ctx %d checkKVFits grants %d slots, want %d", got, g, c.wantSlots)
			}
			// The trade the rule removes: pinned at the one-slot context, fewer slots fit.
			if g, _ := grant(one, true); g >= slots {
				t.Errorf("pinned at the one-slot ctx %d checkKVFits still grants %d slots — the forced budget does not show the trade", one, g)
			}
			// Unpinned there, the build trims the context itself (TestCUDAKVSlots_buildTrimsTheContext) and fits them.
			// Unpinned there, the build trims the context itself (TestCUDAKVSlots_buildTrimsTheContext) to the largest context whose
			// charged pools for every requested slot fit what is free, never below cudaCtxCapDefault, where it clamps the slots instead.
			// On this fixture the pools' rounding makes that context longer than the plan's linear figure for the same bytes.
			f, _, e := dev.Context().MemInfo()
			if e != nil {
				t.Fatalf("MemInfo: %v", e)
			}
			budget := int64(f) - m.ExtraResidentBytes() - testCtxMargin
			exact, ok := kvChargedCapForBytes(budget, slots, cudaCtxCapDefault, one, layers)
			if !ok {
				exact = cudaCtxCapDefault
			}
			if g, ctx := grant(one, false); g != c.wantSlots || ctx != exact || ctx < got {
				t.Errorf("unpinned at the one-slot ctx %d the build gave %d slots at ctx %d, want %d slots at ctx %d (the charged pools' edge; the plan chose %d)", one, g, ctx, c.wantSlots, exact, got)
			}
		})
	}
}

// planOverPriceToleranceBytes is how far Plan's dense figure plus the allocation slack may exceed what the build put on the
// device before its KV. It was one allocation quantum, and the 1.5B then passed only when it ran alone: the fall in free VRAM
// includes the kernel modules and context state a FIRST load of this process pays (1054 MB for the 1.5B alone, 1040 MB after
// the suite's earlier tests had loaded them, 14 MB apart), and that cold-start memory hid an over-pricing of about 8 MB (0.8%)
// that the rest of the suite exposes. Over-pricing is the safe direction (it shrinks the planned context; the 384 MiB margin
// is for UNDER-pricing, which the second check below still bounds), and a host-resident table that Plan wrongly counted is
// 100 MB or more (the untied 7B's embedding), so 16 MiB keeps the check sharp against the defect it exists for. Where the
// 8 MB comes from is not established (the slack's rounding rule and Plan's dense figure each could carry it).
const planOverPriceToleranceBytes = 16 << 20

// TestResidentDenseBytes_matchesCUDADevice pins Plan's dense-weight figure for CUDA
// (decoder.Model.ResidentDenseWeightBytesFor("cuda")) against what a real build puts on the device before its KV: the
// fall in free VRAM from before the load to checkKVFits' probe. That fall is the weights plus the build's scratch and
// kernel modules, so the estimate must not exceed it by more than an allocation quantum per matrix-ish slack, and it
// must fall short of it by less than the build's context margin (ctxMarginFor), which Plan and checkKVFits both reserve. The
// test holds the primary context for its readings, so the startup self-test's pool, which serve releases with the fixtures'
// context, survives here: a warm-up load pays it before the first drop is read. The untied 7B's host-side embedding table must not be priced as device weights
// (docs/measurements/concurrency-mc1-cuda-2026-09-27.md). GOINFER_HEAVY_TESTS adds the 7B, the one untied model here.
func TestResidentDenseBytes_matchesCUDADevice(t *testing.T) {
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	models := []string{
		filepath.Join("..", "testdata", "llama-tiny"),
		filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
		filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		models = append(models, filepath.Join(home, "models", "qwen2.5-7b-instruct-q4_k_m.gguf"))
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	realFree := func() int64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return int64(f)
	}
	orig := cudaFreeVRAM
	t.Cleanup(func() { cudaFreeVRAM = orig })
	if w, err := decoder.Load(models[0], decoder.Options{Backend: "cuda", Quant: "int4"}); err == nil {
		_ = w.Close() // the warm-up: the self-test's pool, before any drop is read
	}
	ran := 0
	for _, p := range models {
		t.Run(filepath.Base(p), func(t *testing.T) {
			if _, err := os.Stat(p); err != nil {
				t.Skipf("no model at %s", p)
			}
			var reads []int64 // every probe read in the build: checkKVFits' first, before the KV
			free0 := realFree()
			cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
				f, _, err := r.dev.Context().MemInfo()
				reads = append(reads, int64(f))
				return f, err
			}
			m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			res, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || len(reads) == 0 {
				t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			drop := free0 - reads[0]
			for i, f := range reads[1:] {
				t.Logf("probe read %d: %.0f MB more on the device than at checkKVFits; the KV pools charge %.0f MB (%d slots at context %d)",
					i+2, mb(reads[0]-f), mb(kvChargedBytes(res.ctxCap, res.KVSlots(), res.layers)), res.KVSlots(), res.ctxCap)
			}
			est, all := m.ResidentDenseWeightBytesFor("cuda"), m.ResidentDenseWeightBytes()
			// Plan prices the requested bytes; the driver rounds each buffer of a quantum or more up (allocRoundSlack), which the build prices into the plan on its own
			// (res.allocSlackBytes). The margin is for what neither knows, so the bound below is on Plan + that slack, not on Plan alone: the rounding alone can exceed the whole margin on the 7B
			// (docs/tasks/task-multimodal-support-2026-10.md).
			t.Logf("device before KV %.0f MB; Plan's CUDA dense %.0f MB (all dense, host tables included, %.0f MB) + allocation slack %.0f MB; "+
				"under by %.0f MB (margin %.0f MB)", mb(drop), mb(est), mb(all), mb(res.allocSlackBytes), mb(drop-est-res.allocSlackBytes), mb(res.ctxMargin()))
			est += res.allocSlackBytes
			if est > drop+planOverPriceToleranceBytes {
				t.Errorf("Plan prices %.0f MB of dense weights for CUDA, but the build put only %.0f MB on the device "+
					"before its KV (tolerance %.0f MB): it counts something CUDA keeps on the host", mb(est), mb(drop), mb(planOverPriceToleranceBytes))
			}
			if drop-est >= res.ctxMargin() {
				t.Errorf("the build put %.0f MB on the device before its KV, %.0f MB more than Plan's %.0f MB: more than "+
					"the %.0f MB margin reserved for scratch", mb(drop), mb(drop-est), mb(est), mb(res.ctxMargin()))
			}
		})
	}
	if ran == 0 {
		t.Skip("no model built a CUDA resident here")
	}
}

func mb(b int64) float64 { return float64(b) / (1 << 20) }

// TestCUDAKVSlots_buildTrimsTheContext pins checkKVFits' half of the slots-before-context rule: when the planned
// unpinned context (Plan cannot see the build's scratch) leaves room for fewer than the requested slots, the build
// trims the context against the real free figure, never below cudaCtxCapDefault, so all of them fit. The free probe is
// stubbed at the KV site to hold exactly 4 slots at 5000 positions. It needs a model whose own window exceeds 8192 so
// the unpinned context starts above the floor (the tiny fixtures' windows are 64-512), so it uses the 0.5B. An explicit
// -ctx is never trimmed: it keeps the context and clamps the slots.
func TestCUDAKVSlots_buildTrimsTheContext(t *testing.T) {
	requireCUDADevice(t)
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no model at %s", p)
	}
	orig := cudaFreeVRAM
	t.Cleanup(func() { cudaFreeVRAM = orig })
	const slots, target = 4, 5000
	cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
		return uint64(kvChargedBytes(target, slots, r.layers) + r.extraBytes + r.ctxMargin()), nil
	}
	for _, c := range []struct {
		name               string
		ctxReq             int
		wantCtx, wantSlots int
	}{
		{"unpinned: trimmed to fit all 4", 0, kvBlocksFor(target) * kvBlock, slots}, // a slot's cost steps every block, so the trim lands on a block edge
		{"explicit -ctx 8192: kept, slots clamped", 8192, 8192, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: slots, ResidentContext: c.ctxReq})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
			}
			// The plan's per-slot context is each slot's guaranteed share (kvShareCtx); an unpinned load lets one conversation's cap
			// (ctxCap) grow to the pool, up to the model's window (kvCapFor), and an explicit -ctx keeps it.
			t.Logf("share %d, cap %d, %d slots", r.kvShareCtx, r.ctxCap, r.KVSlots())
			if r.kvShareCtx != c.wantCtx || r.KVSlots() != c.wantSlots {
				t.Errorf("built a share of %d with %d slots, want %d with %d", r.kvShareCtx, r.KVSlots(), c.wantCtx, c.wantSlots)
			}
			wantCap := c.wantCtx
			if c.ctxReq == 0 {
				wantCap = min(c.wantSlots*kvBlocksFor(c.wantCtx)*kvBlock, r.modelCtx)
			}
			if r.ctxCap != wantCap {
				t.Errorf("one conversation's cap is %d, want %d", r.ctxCap, wantCap)
			}
			if b, want := r.kc[0].Len(), r.kvPoolBlocks(r.kvGroupOf(0), r.KVSlots())*kvBlock*r.layers[0].kvDim; b != want {
				t.Errorf("layer 0 K pool holds %d floats, want %d: the pools were sized before the trim", b, want)
			}
			if err := r.checkCap(r.ctxCap-1, 1); err != nil {
				t.Errorf("checkCap at the last position: %v", err)
			}
			if err := r.checkCap(r.ctxCap, 1); err == nil {
				t.Error("checkCap past the trimmed cap did not fail")
			}
		})
	}
}

// TestCUDAKVSlots_oomKeepsTheSlotsThatFit is S18's CUDA gate (docs/tasks/task-multimodal-support-2026-10.md, "G-S18a"): when the device runs out of memory allocating a later KV slot (checkKVFits sizes the
// count against the free VRAM read before the build's own scratch, with the margin the only slack, so the last slot can miss by the scratch), the resident keeps the slots that fit instead of declining to the CPU.
// Three slots requested, the seam fails slot 2: the model stays CUDA-resident with two, no device buffer leaks (the ledger equals a plain two-slot build's), and both granted slots decode the same tokens as a plain
// two-slot build's. The control, no seam: all three slots. Without the recover the build declines and this goes red.
func TestCUDAKVSlots_oomKeepsTheSlotsThatFit(t *testing.T) {
	dir := filepath.Join("..", "testdata", "mistral-tiny-window")
	requireDeviceAndFixture(t, dir)
	load := func(slots, failAt int) (*decoder.Model, *cudaResident) {
		kvSlotAllocFailAtForTest = failAt
		defer func() { kvSlotAllocFailAtForTest = 0 }()
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: slots})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("not CUDA-resident with %d slots requested (seam at %d): %s", slots, failAt, m.ResidentDecline())
		}
		return m, r
	}
	gen := func(m *decoder.Model) []int {
		ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 8, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return ids
	}
	m3, r3 := load(3, 0)
	if got := m3.ResidentKVSlots(); got != 3 {
		t.Fatalf("control: ResidentKVSlots() = %d, want 3", got)
	}
	_ = r3
	m3.Close()

	m2, r2 := load(2, 0)
	want := gen(m2)
	a2, _ := r2.dev.LedgerLen()
	m2.Close()

	md, rd := load(3, 2) // three requested, slot 2 fails
	defer md.Close()
	if got := md.ResidentKVSlots(); got != 2 {
		t.Fatalf("degraded: ResidentKVSlots() = %d, want 2 (the slots that fit)", got)
	}
	if n := len(rd.kvTabs); n != 2 {
		t.Fatalf("degraded: %d slot table sets, want 2", n)
	}
	if ad, _ := rd.dev.LedgerLen(); ad != a2 {
		t.Errorf("degraded build holds %d device allocations, a plain two-slot build %d: the partial slot leaked", ad, a2)
	}
	if got := gen(md); !slices.Equal(got, want) {
		t.Errorf("degraded build generated %v, a plain two-slot build %v", got, want)
	}
}

var (
	planSlackUpdate = flag.Bool("planslack.update", false, "rewrite testdata/mm2b_realsize.json.gz from this run")
	planSlackOut    = flag.String("planslack.out", "", "write the real-size before/after lines and the diff summary into this directory")
)

// realSizeModels are the real checkpoints the rounding diff and the device check read, from this box's local disk (CLAUDE.md:
// never the archive).
func realSizeModels(t *testing.T, names ...string) []string {
	t.Helper()
	home, _ := os.UserHomeDir()
	var out []string
	for _, n := range names {
		p := filepath.Join(home, "models", n)
		real, err := filepath.EvalSymlinks(p) // a link into the archive reads the archive
		if err != nil {
			continue
		}
		if strings.HasPrefix(real, "/srv/models") || strings.HasPrefix(real, "/Volumes/") {
			t.Fatalf("%s resolves to %s, the archive, not a bench path", p, real)
		}
		out = append(out, p)
	}
	return out
}

// TestPlanSlack_matchesTheBuild (MM2b gate G-MM2b-2): on real models, the rounding the plan prices equals packedAllocSlack over
// the real build's own buffers to the byte. The dense figure against the device is TestResidentDenseBytes_matchesCUDADevice's.
func TestPlanSlack_matchesTheBuild(t *testing.T) {
	requireCUDADevice(t)
	models := realSizeModels(t, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if len(models) < 2 {
		t.Skip("the 0.5B and 1.5B checkpoints are not on this box")
	}
	defer func() { allocSlackSeenForTest = nil }()
	for _, p := range models {
		var planned, built int64 = -1, -1
		allocSlackSeenForTest = func(pl, b int64) { planned, built = pl, b }
		m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("%s: not resident: %s", p, m.ResidentDecline())
		}
		plan := m.Plan("cuda", 64<<30, decoder.PlanRequest{Ctx: 4096})
		t.Logf("%s: plan slack %d, build packedAllocSlack %d, build's slack %d", filepath.Base(p), planned, built, r.allocSlackBytes)
		if planned != built || planned <= 0 || plan.AllocSlackBytes != built || r.allocSlackBytes != built {
			t.Errorf("%s: the plan prices %d (Plan %d), the build's buffers round by %d", filepath.Base(p), planned, plan.AllocSlackBytes, built)
		}
		_ = m.Close()
	}
}

// TestPlanSlack_realSizeDiff (MM2b gate G-MM2b-1): D3's Plan("cuda") cells on real-size models, with the rounding priced and with
// it forced to 0. It writes both and the diff with -planslack.out, and holds the priced side to its golden.
func TestPlanSlack_realSizeDiff(t *testing.T) {
	requireCUDADevice(t)
	models := realSizeModels(t, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "qwen2.5-7b-instruct-q4_k_m.gguf")
	if len(models) < 3 {
		t.Skip("the real-size checkpoints are not on this box: a capture run")
	}
	frees := []int64{7630553088, 7630553088 + 17<<20} // nobara's cuda probe with and without the desktop's context (testdata/machines)
	type cell struct{ group, line string }
	var before, after []cell
	var summary []string
	groups, changed, toDecline := 0, 0, 0
	for _, p := range models {
		m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		name := filepath.Base(p)
		summary = append(summary, fmt.Sprintf("%s: rounding %d bytes (%.1f MB)", name, m.PlanSlack("cuda"), float64(m.PlanSlack("cuda"))/(1<<20)))
		for _, ctx := range []int{4096, 16384, 32768} {
			for _, pinned := range []bool{false, true} {
				for _, kv := range []string{"f32", "f16", "i8"} {
					for _, extra := range []int64{0, 256 << 20} {
						if extra > 0 && (pinned || kv != "f32") {
							continue
						}
						req := decoder.PlanRequest{Ctx: ctx, CtxPinned: pinned, KVF16: kv == "f16", KVI8: kv == "i8", ExtraBytes: extra}
						g := fmt.Sprintf("D3 %s | cuda ctx=%d pinned=%v kv=%s extra=%d", name, ctx, pinned, kv, extra)
						groups++
						sum := func(pl decoder.Plan) string { return fmt.Sprintf("%s %d %d", pl.Placement, pl.Ctx, pl.Slots) }
						edges := map[int64]bool{}
						for _, f := range frees {
							edges[f] = true
						}
						for _, side := range []func(int64) decoder.Plan{
							func(f int64) decoder.Plan { return m.PlanWithoutSlackForTest("cuda", f, req) },
							func(f int64) decoder.Plan { return m.Plan("cuda", f, req) },
						} {
							for _, e := range changePointsOf(0, 1<<36, 8, func(f int64) string { return sum(side(f)) }) {
								edges[e], edges[e-1] = true, true
							}
						}
						var fs []int64
						for f := range edges {
							fs = append(fs, f)
						}
						sort.Slice(fs, func(i, j int) bool { return fs[i] < fs[j] })
						diff := false
						for _, f := range fs {
							b, a := m.PlanWithoutSlackForTest("cuda", f, req), m.Plan("cuda", f, req)
							bl, al := fmt.Sprintf("free %d: %s", f, planLineOf(b)), fmt.Sprintf("free %d: %s", f, planLineOf(a))
							before, after = append(before, cell{g, bl}), append(after, cell{g, al})
							if bl != al {
								diff = true
							}
							if a.Placement == decoder.PlacementDecline && b.Placement != decoder.PlacementDecline {
								toDecline++
								summary = append(summary, fmt.Sprintf("TO DECLINE: %s at free %d: was %s ctx %d, now %s", g, f, b.Placement, b.Ctx, a.Reason))
							}
						}
						if diff {
							changed++
						}
					}
				}
			}
		}
		_ = m.Close()
	}
	summary = append([]string{fmt.Sprintf("%d groups, %d changed, %d cells move to a decline (a cell is a group at one free reading; the readings are nobara's two and each side's change points)", groups, changed, toDecline)}, summary...)
	t.Log(strings.Join(summary, "\n"))
	if *planSlackOut != "" {
		write := func(name string, cs []cell) {
			var sb strings.Builder
			for _, c := range cs {
				sb.WriteString(c.group + " || " + c.line + "\n")
			}
			if err := os.WriteFile(filepath.Join(*planSlackOut, name), []byte(sb.String()), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write("before.txt", before)
		write("after.txt", after)
		if err := os.WriteFile(filepath.Join(*planSlackOut, "summary.txt"), []byte(strings.Join(summary, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	byGroup := map[string][]string{}
	for _, c := range after {
		byGroup[c.group] = append(byGroup[c.group], c.line)
	}
	hashes := map[string]string{}
	for g, ls := range byGroup {
		h := sha256.Sum256([]byte(strings.Join(ls, "\n")))
		hashes[g] = hex.EncodeToString(h[:8])
	}
	golden := filepath.Join("testdata", "mm2b_realsize.json.gz")
	if *planSlackUpdate {
		f, err := os.Create(golden)
		if err != nil {
			t.Fatal(err)
		}
		zw := gzip.NewWriter(f)
		b, _ := json.MarshalIndent(hashes, "", " ")
		_, _ = zw.Write(b)
		_ = zw.Close()
		_ = f.Close()
		return
	}
	f, err := os.Open(golden)
	if err != nil {
		t.Skipf("no golden yet (%v): a capture run", err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var want map[string]string
	if err := json.NewDecoder(zr).Decode(&want); err != nil {
		t.Fatal(err)
	}
	for g, h := range hashes {
		if want[g] != h {
			t.Errorf("real-size group changed: %s", g)
		}
	}
	if len(want) != len(hashes) {
		t.Errorf("the golden has %d groups, this run %d", len(want), len(hashes))
	}
}

func planLineOf(p decoder.Plan) string {
	return fmt.Sprintf("%s ctx %d slots %d dense %d slack %d expertUsed %d kv %d extra %d free %d need %d | %s",
		p.Placement, p.Ctx, p.Slots, p.DenseBytes, p.AllocSlackBytes, p.ExpertBytesUsed, p.KVBytes, p.ExtraBytes, p.FreeBytes, p.NeedBytes(), p.Reason)
}

// changePointsOf is the readings in [lo, hi] at which outcome changes, by bisection, at most max of them.
func changePointsOf(lo, hi int64, max int, outcome func(int64) string) []int64 {
	var out []int64
	var walk func(a, b int64, oa, ob string)
	walk = func(a, b int64, oa, ob string) {
		if oa == ob || len(out) >= max {
			return
		}
		if b-a <= 1 {
			out = append(out, b)
			return
		}
		mid := a + (b-a)/2
		om := outcome(mid)
		walk(a, mid, oa, om)
		walk(mid, b, om, ob)
	}
	walk(lo, hi, outcome(lo), outcome(hi))
	return out
}

// TestBuildDemand_registered (G-GF-4): decoder's registry holds this package's report for cuda, not decoder's default copy, so the
// agreement test's comparison is the one the fit guard reads. It first shows the check tells two registered reports apart.
func TestBuildDemand_registered(t *testing.T) {
	ptr := func(f decoder.BuildDemandFunc) uintptr { return reflect.ValueOf(f).Pointer() }
	if o := decoder.RegisteredBuildDemandForTest("metal"); o == nil || ptr(o) == ptr(buildDemand) {
		t.Fatal("the metal report is missing or is this package's: the identity check cannot tell reports apart")
	}
	if f := decoder.RegisteredBuildDemandForTest("cuda"); f == nil || ptr(f) != ptr(buildDemand) {
		t.Fatal("decoder's registry does not hold this package's cuda report")
	}
}

// TestLedger_closesAgainstDevice (G-MM3a-4, I3): for a tiny model and a real 0.5B (where it is on disk), the device's free VRAM
// just after the resident's device is created, less its free VRAM after the build and three identical 8-token requests, is at
// least the ledger's charged VRAM and at most 64 MiB above it (modules, streams and the driver's shared pages for small buffers).
// I2: the charged bytes do not grow from one request to the next, and after Close the ledger holds no lease.
func TestLedger_closesAgainstDevice(t *testing.T) {
	requireCUDADevice(t)
	models := append([]string{filepath.Join("..", "testdata", "llama-tiny")}, realSizeModels(t, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")...)
	if len(models) < 2 {
		t.Log("the 0.5B checkpoint is not on this box: llama-tiny only")
	}
	const band = 64 << 20
	real := CreateSystemDefaultDevice
	defer func() { CreateSystemDefaultDevice = real }() // the probe below reads free VRAM as each device is created
	created := map[*Device]uint64{}
	CreateSystemDefaultDevice = func() (*Device, error) {
		d, err := real()
		if err == nil {
			free, _, e := d.Context().MemInfo()
			if e != nil {
				return nil, e
			}
			created[d] = free
		}
		return d, err
	}
	for _, p := range models {
		m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("%s: not resident: %s", p, m.ResidentDecline())
		}
		// I2 within the model's life: a request leaves nothing behind, so the charged bytes after a second and a third identical
		// request equal those after the first (a buffer leaked per request grows them; the sweep at Close would hide it).
		var after []int64
		for range 3 {
			ch, gen := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 8, decoder.SamplingParams{})
			for range ch {
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("%s: generate: %v", p, err)
			}
			after = append(after, ledgerOf(r.dev).Usage().Charged)
		}
		if after[1] != after[0] || after[2] != after[0] {
			t.Errorf("%s: the charged bytes grow across identical requests: %v", filepath.Base(p), after)
		}
		free0, seen := created[r.dev]
		if !seen {
			t.Fatalf("%s: the resident's device was not created through CreateSystemDefaultDevice", p)
		}
		var free1 uint64
		if err := r.do(func() error { f, _, e := r.dev.Context().MemInfo(); free1 = f; return e }); err != nil {
			t.Fatalf("%s: free VRAM: %v", p, err)
		}
		u := ledgerOf(r.dev).Usage()
		measured, charged := int64(free0)-int64(free1), u.ByTier[mem.VRAM]
		t.Logf("%s: measured %d bytes on the device, the ledger charges %d (requested %d); gap %d; by line %v", filepath.Base(p),
			measured, charged, u.Requested, measured-charged, u.ByLine)
		if measured < charged || measured > charged+band {
			t.Errorf("%s: measured %d, charged %d: outside [charged, charged+%d]", filepath.Base(p), measured, charged, band)
		}
		dev := r.dev
		_ = m.Close()
		if u := ledgerOf(dev).Usage(); u.Leases != 0 || u.Charged != 0 {
			t.Errorf("%s: after Close the ledger holds %d leases, %d bytes, by line %v", filepath.Base(p), u.Leases, u.Charged, u.ByLine)
		}
	}
}

// TestLedger_unplannedIsReported (G-MM3a-2's plant): an allocation on the empty line, and a lease left open, are each what the
// suite's release check (ledgerBreak) reports; a tracked pinned host buffer closes with the device; a clean device reports nothing.
func TestLedger_unplannedIsReported(t *testing.T) {
	requireCUDADevice(t)
	real := gpu.ReleasedForTest
	defer func() { gpu.ReleasedForTest = real }()
	var got *mem.Ledger
	gpu.ReleasedForTest = func(d *Device) { got = ledgerOf(d) }
	for _, plant := range []string{"unplanned", "open", "clean"} {
		d, err := CreateSystemDefaultDevice()
		if err != nil {
			t.Skip(err)
		}
		if _, err := gpu.NewHostBufferOn[float32](d, mem.LineScratch, 16); err != nil {
			t.Fatal(err)
		}
		_ = d.NewBufferLenOn(mem.LineKV, 1<<20)
		switch plant {
		case "unplanned":
			_ = d.MustBufOn("", 3<<20, 3<<20, "planted")
		case "open":
			ledgerOf(d).Reserve(mem.HostAnon, mem.LinePinnedExperts, 4096, 4096) // never released
		}
		d.ReleaseObjects()
		b := ledgerBreak(got)
		switch {
		case plant == "unplanned" && !strings.Contains(b, "unplanned peak 4194304"):
			t.Errorf("an unplanned 3 MiB (charged 4) gives %q", b)
		case plant == "open" && !strings.Contains(b, "1 leases (4096 bytes) open"):
			t.Errorf("an open lease gives %q", b)
		case plant == "clean" && b != "":
			t.Errorf("a clean device gives %q", b)
		}
	}
	if b := ledgerBreak(nil); b == "" {
		t.Error("a device with no ledger is not reported")
	}
}

var sb11Update = flag.Bool("sb11.update", false, "rewrite testdata/sb11_decode_llama_tiny.json from this card (review the diff)")

// sb11Token is one counted decode token's per-operation launches, syncs and allocations (time is not compared).
type sb11Token map[string][3]int64

func sb11Counts(r cost.Report) sb11Token {
	out := sb11Token{}
	for name, c := range r.Decode.Ops {
		if name == "token" {
			continue
		}
		out[name] = [3]int64{c.Launches, c.Syncs, c.Allocs}
	}
	return out
}

// countedDecode runs one decode token at pos on r, counted into a fresh counter.
func countedDecode(t *testing.T, m *decoder.Model, r *cudaResident, id, pos int) (cost.Report, []float32) {
	t.Helper()
	c := cost.New()
	var lg []float32
	if err := r.costToken(c, cost.Decode, func() error {
		var e error
		lg, e = r.Forward(m.EmbedResidentForTest(id), pos)
		return e
	}); err != nil {
		t.Fatal(err)
	}
	return c.Report(), append([]float32(nil), lg...)
}

// TestCost_cudaDecodeExact (G-SB11-8, G-SB11-9, part of G-SB11-11): llama-tiny resident on CUDA. Tokens 2 to 4 each count the
// same per-kernel launches, syncs and allocations, equal to the committed golden; their per-kernel launches sum to the device's
// own count (no launch_other); a steady token allocates nothing; logits match the uncounted token's bit for bit; and the two plants
// (an extra launch, an extra sync) are seen.
func TestCost_cudaDecodeExact(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not resident: %s", m.ResidentDecline())
	}
	if _, err := r.Forward(m.EmbedResidentForTest(1), 0); err != nil {
		t.Fatal(err)
	}
	var first sb11Token
	for pos := 1; pos <= 3; pos++ {
		rep, _ := countedDecode(t, m, r, 2+pos, pos)
		got := sb11Counts(rep)
		if _, other := got["launch_other"]; other {
			t.Errorf("token %d: %d launches not made through r.launch", pos, got["launch_other"][0])
		}
		if got["alloc"][2] != 0 {
			t.Errorf("token %d allocated %d leases", pos, got["alloc"][2])
		}
		if pos == 1 {
			first = got
			continue
		}
		if !reflect.DeepEqual(got, first) {
			t.Errorf("token %d counts differ from token 1's:\n%v\n%v", pos, got, first)
		}
	}
	golden := filepath.Join("testdata", "sb11_decode_llama_tiny.json")
	if *sb11Update {
		b, _ := json.MarshalIndent(first, "", " ")
		if err := os.WriteFile(golden, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("%v (write it with -sb11.update)", err)
	}
	var want sb11Token
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, want) {
		t.Errorf("a decode token's counts differ from the golden:\n got  %v\n want %v", first, want)
	}
	var launches int64
	for _, v := range first {
		launches += v[0]
	}
	t.Logf("a decode token: %d kernels, %d launches, %d syncs", len(first)-2, launches, first["sync"][1])

	// Counting changes nothing: the same token at the same position, uncounted and counted.
	off, err := r.Forward(m.EmbedResidentForTest(9), 4)
	if err != nil {
		t.Fatal(err)
	}
	off = append([]float32(nil), off...)
	_, on := countedDecode(t, m, r, 9, 4)
	for i := range on {
		if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
			t.Fatalf("logit %d differs with the counter on", i)
		}
	}

	// The plants (G-SB11-9).
	defer func() { costPlantForTest = 0 }()
	costPlantForTest, r.costNamed = 1, 0
	rep, _ := countedDecode(t, m, r, 10, 5)
	planted := sb11Counts(rep)
	diff := 0
	for name, v := range planted {
		if v[0] == want[name][0]+1 {
			diff++
		} else if v != want[name] {
			t.Errorf("the extra-launch plant moved %s from %v to %v", name, want[name], v)
		}
	}
	if _, other := planted["launch_other"]; diff != 1 || other {
		t.Errorf("the extra-launch plant moved %d kernels by one (want 1), launch_other %v", diff, planted["launch_other"])
	}
	costPlantForTest = 2
	rep, _ = countedDecode(t, m, r, 11, 6)
	if got := sb11Counts(rep)["sync"][1]; got != want["sync"][1]+1 {
		t.Errorf("the extra-sync plant gives %d syncs, want %d", got, want["sync"][1]+1)
	}
}

// costOffDevice and costOffLaunch are the off path one launch pays, at the registered bound: two of the device's atomic adds and
// the resident's nil check.
type costOffDevice struct{ launches, waits atomic.Int64 }

//go:noinline
func costOffLaunch(d *costOffDevice, c *cost.Counter) bool {
	d.launches.Add(1)
	d.waits.Add(1)
	return c != nil
}

// TestCost_cudaOffShare (G-SB11-10): the off path a launch pays (costOffLaunch, at its registered bound) × the launches of a
// decode token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on the card, is under 0.5%. Heavy; run it under the
// timing lock and note the card's performance state, power limit and the box's load.
func TestCost_cudaOffShare(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	requireCUDADevice(t)
	models := realSizeModels(t, "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if len(models) == 0 {
		t.Skip("no 0.5B on this box")
	}
	m, err := decoder.Load(models[0], decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := m.ResidentForwardForTest().(*cudaResident)
	for pos := range 8 {
		if _, err := r.Forward(m.EmbedResidentForTest(100+pos), pos); err != nil {
			t.Fatal(err)
		}
	}
	rep, _ := countedDecode(t, m, r, 300, 8)
	launches := rep.Decode.Totals.Launches
	const tokens = 64
	start := time.Now()
	for i := range tokens {
		if _, err := r.Forward(m.EmbedResidentForTest(400+i), 9+i); err != nil {
			t.Fatal(err)
		}
	}
	perToken := time.Since(start) / tokens
	var dev costOffDevice
	hook := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			if costOffLaunch(&dev, nil) {
				b.Fatal("on")
			}
		}
	})
	perLaunch := float64(hook.T.Nanoseconds()) / float64(hook.N) // NsPerOp truncates to whole nanoseconds
	share := perLaunch * float64(launches) / float64(perToken.Nanoseconds())
	load, _ := os.ReadFile("/proc/loadavg")
	t.Logf("decode %v a token, %d launches a token, the off path %.2f ns a launch (%d runs): %.4f%% of a token; load %s",
		perToken, launches, perLaunch, hook.N, 100*share, strings.TrimSpace(string(load)))
	if share >= 0.005 {
		t.Errorf("the off path costs %.3f%% of a decode token, at or over the registered 0.5%%", 100*share)
	}
}
