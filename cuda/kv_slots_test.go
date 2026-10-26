//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
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
// it saw at the call.
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
		if p.calls != 1 {
			t.Fatalf("the free-VRAM probe ran %d times during the build, want once (at checkKVFits)", p.calls)
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
			var atKV int64
			free0 := realFree()
			cudaFreeVRAM = func(r *cudaResident) (uint64, error) {
				f, _, err := r.dev.Context().MemInfo()
				atKV = int64(f)
				return f, err
			}
			m, err := decoder.Load(p, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			res, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok || atKV == 0 {
				t.Skipf("not CUDA-resident: %s", m.ResidentDecline())
			}
			ran++
			drop := free0 - atKV
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
