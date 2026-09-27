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
	if len(r.kvSlotBufs) != 0 || r.KVSlots() != 1 || m.ResidentKVSlots() != 1 {
		t.Errorf("recurrent resident: %d slot buffers, KVSlots %d, model reports %d — want one slot",
			len(r.kvSlotBufs), r.KVSlots(), m.ResidentKVSlots())
	}
}

// TestKVSlotsFit pins MC1's clamp arithmetic on CUDA (kvSlotsFit) without allocating a real clamp's worth of KV.
func TestKVSlotsFit(t *testing.T) {
	const mb = int64(1 << 20)
	for _, tc := range []struct {
		name                string
		want                int
		free, slot, reserve int64
		got                 int
	}{
		{"all fit", 4, 4000 * mb, 470 * mb, 384 * mb, 4},
		{"clamped to what fits", 4, 2000 * mb, 470 * mb, 384 * mb, 3}, // 3*470+384 = 1794 <= 2000; 4*470+384 > 2000
		{"exactly at the edge fits", 2, 940*mb + 384*mb, 470 * mb, 384 * mb, 2},
		{"only the first fits", 4, 900 * mb, 470 * mb, 384 * mb, 1},
		{"even the first over budget: still 1, the refusal is checkKVFits'", 4, 100 * mb, 470 * mb, 384 * mb, 1},
		{"request of 0 means 1", 0, 4000 * mb, 470 * mb, 384 * mb, 1},
	} {
		if got := kvSlotsFit(tc.want, tc.free, tc.slot, tc.reserve); got != tc.got {
			t.Errorf("%s: kvSlotsFit(%d, ...) = %d, want %d", tc.name, tc.want, got, tc.got)
		}
	}
}

// kvSlotsProbe is one armed cudaFreeVRAM stub of TestCUDAKVSlots_pricedAgainstWhatIsLeft: it reports live minus
// the real bytes the device has allocated since it was armed (or the real free figure, live == 0), and records what
// it saw at the call.
type kvSlotsProbe struct {
	free0   uint64 // the real free VRAM when the probe was armed
	live    int64  // the fictional free figure the probe falls from; 0: report the real one
	sawNeed int64  // one slot's KV at the probed resident's context
	sawDrop int64  // real bytes allocated between arming and the probe call
	sawFree int64  // what the probe returned
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
		need := kvBytesForCap(r.ctxCap, r.layers)
		return uint64(2*need + r.extraBytes + ctxCapMarginBytes + need/2), nil
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
	if len(r.kvSlotBufs) != 2 || m.ResidentKVSlots() != 2 {
		t.Fatalf("clamped build: %d slot buffers, model reports %d slots — want 2 of 4", len(r.kvSlotBufs), m.ResidentKVSlots())
	}
	for s := range 2 {
		for l := range r.layers {
			if b := r.kvSlotBufs[s].kc[l]; b.Len() != r.ctxCap*r.layers[l].kvDim {
				t.Errorf("slot %d layer %d: K buffer holds %d floats, want ctxCap*kvDim = %d", s, l, b.Len(), r.ctxCap*r.layers[l].kvDim)
			}
		}
	}
	if err := r.UseKVSlot(2); err == nil {
		t.Error("UseKVSlot(2) on a 2-slot resident did not fail")
	}
	for _, s := range []int{1, 0, 1} {
		if err := r.UseKVSlot(s); err != nil {
			t.Fatalf("UseKVSlot(%d): %v", s, err)
		}
		if &r.kc[0] != &r.kvSlotBufs[s].kc[0] {
			t.Errorf("after UseKVSlot(%d) the bound K buffers are not slot %d's", s, s)
		}
		if _, err := r.Forward(m.EmbedResidentForTest(5), 0); err != nil {
			t.Fatalf("forward on slot %d: %v", s, err)
		}
	}
}

// TestCUDAKVSlots_pricedAgainstWhatIsLeft pins how the slot count is priced — the CUDA side of the Metal double-count
// bug (6807ab95), where a weights-inclusive base was compared with a live figure the weights had already left. The
// free-VRAM probe is stubbed to fall by exactly what the device really allocates from a fictional starting figure,
// chosen (from a calibration build) so that precisely `want` slots fit beside everything the build puts on the device
// before its KV. One more slot than that is requested, so the test is two-sided:
//   - pricing that counts the weights twice (a weights-inclusive base against the post-weights probe) grants fewer
//     than `want`;
//   - pricing KV alone against a figure read before the weights are on the device grants `want`+1, which on a real
//     card would not fit.
//
// A double count is visible only where the build's pre-KV bytes exceed half a slot, so a fixture where they do not
// skips with the numbers. GOINFER_CUDA_KVSLOTS_MODEL runs it on a real checkpoint (the 1.5B: ~1 GB before KV
// against ~0.46 GB per slot).
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
			p.sawNeed, p.sawDrop = kvBytesForCap(r.ctxCap, r.layers), int64(p.free0)-int64(now)
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
	reserve := int64(ctxCapMarginBytes) // no companion attach in this test
	if cal.sawDrop < cal.sawNeed/2 {
		t.Skipf("the build allocates %d B before its KV against %d B per slot: a double count of that could not "+
			"change the slot count here (run with GOINFER_CUDA_KVSLOTS_MODEL on a real checkpoint)", cal.sawDrop, cal.sawNeed)
	}
	live := cal.sawDrop + reserve + want*cal.sawNeed + cal.sawNeed/2
	got, p := build(live, want+1)
	t.Logf("pre-KV bytes %d (calibration) / %d (this build), one slot %d B, probe returned %d B: %d of %d slots granted",
		cal.sawDrop, p.sawDrop, p.sawNeed, p.sawFree, got, want+1)
	if got != want {
		t.Errorf("granted %d slots where exactly %d fit beside what the build had already allocated: fewer means the "+
			"price counts the weights again; more means it was read before they were on the device", got, want)
	}
}

// TestResolveCtxCapFit_slotsShrinkTheContext pins the owner's 2026-09-27 decision (docs/tasks/task-concurrency-2026-09.md
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
	perPos := kvBytesForCap(1, layers)
	dense := probe.ResidentDenseWeightBytes()
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
		{"interior: 4 slots fit at ~5000", 5000, func(got int) bool { return got > 4800 && got <= 5000 }, slots},
		{"not even the floor holds 4: the floor, and the build clamps", 3000, func(got int) bool { return got == cudaCtxCapDefault }, 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			forced := free - ctxCapMarginBytes - dense - int64(slots)*perPos*int64(c.target)
			if forced <= 0 {
				t.Skipf("free VRAM %d B too small to force this budget", free)
			}
			m, err := decoder.Load(dir, decoder.Options{Quant: "int4", ExtraResidentBytes: forced, ResidentKVSlots: slots})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			one := resolveCtxCapFit(m, 0, 1<<20, 1)
			got := resolveCtxCapFit(m, 0, 1<<20, slots)
			pinned := resolveCtxCapFit(m, 8192, 1<<20, slots)
			t.Logf("one slot: ctx %d; %d slots: ctx %d (target %d); explicit 8192 with %d slots: %d", one, slots, got, c.target, slots, pinned)
			if one != fitDefaultCtx {
				t.Fatalf("one slot chose ctx %d, want the full candidate %d — the forced budget did not leave one slot room", one, fitDefaultCtx)
			}
			if !c.want(got) {
				t.Errorf("%d slots chose ctx %d, target %d", slots, got, c.target)
			}
			if pinned != 8192 {
				t.Errorf("an explicit -ctx 8192 became %d with slots requested — a pinned context must never shrink", pinned)
			}
			grant := func(ctx int) int {
				r := &cudaResident{dev: dev, ctxCap: ctx, layers: layers, kvSlotsReq: slots, extraBytes: m.ExtraResidentBytes()}
				if err := r.checkKVFits(); err != nil {
					t.Fatalf("checkKVFits at ctx %d: %v", ctx, err)
				}
				return r.kvSlotsN
			}
			if g := grant(got); g != c.wantSlots {
				t.Errorf("at the chosen ctx %d checkKVFits grants %d slots, want %d", got, g, c.wantSlots)
			}
			if g := grant(one); g >= slots {
				t.Errorf("at the one-slot ctx %d checkKVFits still grants %d slots — the forced budget does not show the trade", one, g)
			}
		})
	}
}
