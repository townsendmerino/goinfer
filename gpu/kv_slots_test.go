//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"errors"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kvSlotTurn is one generation of the MC1 identity scenario: its token ids and PrefillReused.
type kvSlotTurn struct {
	ids    []int
	reused int
}

// loadWebGPUResident loads dir on webgpu with `slots` resident KV slots requested and returns the model and its
// resident. It skips when a tiny fixture's weights are not built on this machine (they are gitignored) or when there
// is no device, and fails when the model does not go resident.
func loadWebGPUResident(t *testing.T, dir string, opts decoder.Options, slots int) (*decoder.Model, *residentDecoder) {
	t.Helper()
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
			t.Skipf("fixture %s has no model.safetensors on this machine (gitignored; regenerate it): %v", dir, err)
		}
	}
	o := opts
	o.Backend, o.ResidentKVSlots = "webgpu", slots
	m, err := decoder.Load(dir, o)
	if err != nil {
		if !GPUEverAvailable() {
			t.Skipf("no webgpu device: %v", err)
		}
		t.Fatalf("load %s: %v", dir, err)
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		m.Close()
		t.Fatalf("%s did not go webgpu-resident: %s", dir, m.ResidentDecline())
	}
	requireGPU(t, nil)
	return m, rd
}

// kvSlotsScenario is MC1's identity gate on WebGPU (docs/tasks/task-concurrency-2026-09.md; the twins are
// TestMetalKVSlots_interleavedMatchesAlone and TestCUDAKVSlots_interleavedMatchesAlone): with two resident KV slots,
// two conversations that share only a lead, interleaved turn by turn through the production Generate path, must each
// emit exactly the token ids they emit when served alone, and reuse exactly as much of their own history on every
// turn. A control — the same interleaving on one slot — must thrash, so the scenario can see the failure it guards
// against.
//
// adapter names a compute-time LoRA adapter for conversation A (loaded on every model by load), "" for none. An
// adapter conversation runs through a fresh Session per turn — the resident path admits an adapter only at
// prefillFrom 0 — so generateInto binds the adapter BEFORE it acquires the slot, and a switch has to move it.
func kvSlotsScenario(t *testing.T, dir string, opts decoder.Options, maxTok, base int, adapter string, load func(*decoder.Model)) {
	t.Helper()
	lead := []int{base + 1, base + 2, base + 3} // the shared "chat template" lead
	first := map[string][]int{
		"A": append(slices.Clone(lead), base+10, base+11, base+12, base+13),
		"B": append(slices.Clone(lead), base+20, base+21, base+22, base+23),
	}
	gen := func(m *decoder.Model, c string, prompt []int) kvSlotTurn {
		var ch <-chan int
		var g *decoder.Generation
		if c == "A" && adapter != "" {
			s := m.NewSession(0)
			if err := s.UseAdapter(adapter); err != nil {
				t.Fatalf("UseAdapter: %v", err)
			}
			ch, g = s.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		} else {
			ch, g = m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		}
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("%s: %v", c, err)
		}
		return kvSlotTurn{ids, g.PrefillReused}
	}
	run := func(m *decoder.Model, order []string) map[string][]kvSlotTurn {
		prompts := map[string][]int{"A": slices.Clone(first["A"]), "B": slices.Clone(first["B"])}
		out := map[string][]kvSlotTurn{}
		for _, c := range order {
			turn := gen(m, c, prompts[c])
			out[c] = append(out[c], turn)
			// the next user turn extends the conversation with its reply and two new ids
			prompts[c] = append(append(slices.Clone(prompts[c]), turn.ids...), base+30+len(out[c]), base+40+len(out[c]))
		}
		return out
	}
	loadN := func(slots int) *decoder.Model {
		m, rd := loadWebGPUResident(t, dir, opts, slots)
		if got := m.ResidentKVSlots(); got != slots || rd.KVSlots() != slots {
			m.Close()
			t.Fatalf("ResidentKVSlots() = %d (resident %d) with %d requested — the resident did not allocate them", got, rd.KVSlots(), slots)
		}
		if load != nil {
			load(m)
		}
		return m
	}
	interleaved := []string{"A", "B", "A", "B", "A", "B"}

	mTwo := loadN(2)
	inter := run(mTwo, interleaved)
	mTwo.Close()
	alone := map[string][]kvSlotTurn{}
	for _, c := range []string{"A", "B"} { // each conversation alone on a fresh one-slot model
		m := loadN(1)
		alone[c] = run(m, []string{c, c, c})[c]
		m.Close()
	}
	mOne := loadN(1)
	thrash := run(mOne, interleaved) // the control
	mOne.Close()
	if adapter != "" {
		// The adapter has to change A's output, or a switch that dropped it would pass unseen.
		m := loadN(1)
		plain := gen(m, "B", first["A"]) // A's first prompt through the base model
		m.Close()
		if slices.Equal(plain.ids, alone["A"][0].ids) {
			t.Fatalf("adapter %q does not change A's first reply (%v) — this scenario cannot see an adapter lost in a slot switch", adapter, plain.ids)
		}
	}

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

// TestWebGPUKVSlots_interleavedMatchesAlone runs kvSlotsScenario on one tiny fixture per KV layout the resident
// allocates: dense f32 (llama-tiny), dense int8 with its per-head scale buffers (llama-tiny, KVPrecision i8), dense f16,
// sliding window (mistral-tiny-window), and MLA (deepseek-tiny: one latent buffer per layer). GOINFER_WEBGPU_KVSLOTS_MODEL
// =<checkpoint> runs it on a real checkpoint instead (int4, 48 tokens per turn, prompts of ids from a fixed range).
func TestWebGPUKVSlots_interleavedMatchesAlone(t *testing.T) {
	if p := os.Getenv("GOINFER_WEBGPU_KVSLOTS_MODEL"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("GOINFER_WEBGPU_KVSLOTS_MODEL=%s: %v", p, err)
		}
		kvSlotsScenario(t, p, decoder.Options{Quant: "int4"}, 48, 1000, "", nil)
		return
	}
	for _, tc := range []struct {
		name, fixture string
		opts          decoder.Options
	}{
		{"llama-tiny", "llama-tiny", decoder.Options{Quant: "int4"}},
		{"llama-tiny-kv-i8", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "i8"}},
		{"llama-tiny-kv-f16", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "f16"}},
		{"mistral-tiny-window", "mistral-tiny-window", decoder.Options{Quant: "int4"}},
		{"deepseek-tiny", "deepseek-tiny", decoder.Options{Quant: "int4"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kvSlotsScenario(t, filepath.Join("..", "testdata", tc.fixture), tc.opts, 8, 0, "", nil)
		})
	}
}

// TestWebGPUKVSlots_adapterFollowsTheSwitch is the scenario with conversation A on a compute-time LoRA adapter and B on
// the base model. generateInto binds the adapter on whatever runner is bound, THEN acquires A's slot, so a switch that
// did not move the adapter would decode A on the base weights (or leave B's runner carrying A's delta).
func TestWebGPUKVSlots_adapterFollowsTheSwitch(t *testing.T) {
	adapterDir := buildLlamaTinyLoRAFixtureGPU(t)
	kvSlotsScenario(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Quant: "int4"}, 8, 0, "a",
		func(m *decoder.Model) {
			if err := m.LoadAdapter("a", adapterDir); err != nil {
				t.Fatalf("LoadAdapter: %v", err)
			}
		})
}

// TestWebGPUKVSlots_recurrentKeepsOne confirms nothing on WebGPU bypasses the one-slot rule for a family with recurrent
// state: a Gated-DeltaNet hybrid (qwen35-tiny) and a Mamba-2 hybrid (nemotron-tiny, resident at int4 by default)
// asked for 4 slots allocate 1.
func TestWebGPUKVSlots_recurrentKeepsOne(t *testing.T) {
	for _, f := range []string{"qwen35-tiny", "nemotron-tiny"} {
		t.Run(f, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", f), decoder.Options{Quant: "int4"}, 4)
			defer m.Close()
			if rd.rm.dnet == nil && rd.rm.mamba == nil {
				t.Fatalf("%s built without recurrent layers — this test no longer covers a recurrent family", f)
			}
			if len(rd.slots) != 0 || rd.KVSlots() != 1 || m.ResidentKVSlots() != 1 {
				t.Errorf("recurrent resident: %d slots, KVSlots %d, model reports %d — want one slot", len(rd.slots), rd.KVSlots(), m.ResidentKVSlots())
			}
		})
	}
}

// TestWebGPUKVSlots_slotOwnsItsKV pins what a slot is: every per-sequence buffer is its own and the same size as slot
// 0's, and every weight is slot 0's (a slot that copied a weight would double the model; one that shared a KV buffer
// would be the thrash back).
func TestWebGPUKVSlots_slotOwnsItsKV(t *testing.T) {
	for _, tc := range []struct {
		fixture string
		opts    decoder.Options
	}{
		{"llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "i8"}},
		{"deepseek-tiny", decoder.Options{Quant: "int4"}},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", tc.fixture), tc.opts, 3)
			defer m.Close()
			if len(rd.slots) != 3 {
				t.Fatalf("%d slots, want 3", len(rd.slots))
			}
			s0 := rd.slots[0].rm
			for s := 1; s < 3; s++ {
				rm := rd.slots[s].rm
				for l := range s0.layers {
					a, b := &s0.layers[l], &rm.layers[l]
					kv0, kv := slotKVBuffers(a), slotKVBuffers(b)
					nKV := 0
					for i := range kv0 {
						if (*kv0[i] == nil) != (*kv[i] == nil) {
							t.Fatalf("slot %d layer %d KV buffer %d: nil in one slot only", s, l, i)
						}
						if *kv0[i] == nil {
							continue
						}
						nKV++
						if *kv0[i] == *kv[i] {
							t.Errorf("slot %d layer %d shares KV buffer %d with slot 0", s, l, i)
						}
						if (*kv0[i]).GetSize() != (*kv[i]).GetSize() {
							t.Errorf("slot %d layer %d KV buffer %d: %d bytes, slot 0 %d", s, l, i, (*kv[i]).GetSize(), (*kv0[i]).GetSize())
						}
					}
					if nKV == 0 {
						t.Fatalf("layer %d has no KV buffer — the fixture no longer exercises this", l)
					}
					if a.attnNorm != b.attnNorm || a.mlpNorm != b.mlpNorm || a.invFreq != b.invFreq {
						t.Errorf("slot %d layer %d does not share slot 0's norms / RoPE table", s, l)
					}
				}
				if rm.finalNorm != s0.finalNorm {
					t.Errorf("slot %d does not share slot 0's final norm", s)
				}
				if kvBytesPerSlot(&rm) != kvBytesPerSlot(&s0) {
					t.Errorf("slot %d KV %d bytes, slot 0 %d", s, kvBytesPerSlot(&rm), kvBytesPerSlot(&s0))
				}
			}
		})
	}
}

// TestWebGPUKVSlots_forwardNFollowsTheSlot covers the ForwardN verify pool, which grows lazily and whose runners bake
// the KV they were built over: slot 0's pool must never serve slot 1. Rows on slot 0, then rows on slot 1, then one
// more row on slot 0 must equal the same calls on fresh one-slot models, bit for bit.
func TestWebGPUKVSlots_forwardNFollowsTheSlot(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-tiny")
	opts := decoder.Options{Quant: "int4"}
	m, rd := loadWebGPUResident(t, dir, opts, 2)
	defer m.Close()
	hidden, _, _, _, _, _, _ := m.Dims()
	rng := rand.New(rand.NewSource(7))
	emb := func(n int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = make([]float32, hidden)
			for j := range out[i] {
				out[i][j] = float32(rng.NormFloat64())
			}
		}
		return out
	}
	e0, e1 := emb(5), emb(4)

	got0, err := rd.ForwardN(e0[:4], 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := rd.UseKVSlot(1); err != nil {
		t.Fatal(err)
	}
	got1, err := rd.ForwardN(e1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := rd.UseKVSlot(0); err != nil {
		t.Fatal(err)
	}
	got0b, err := rd.Forward(e0[4], 4)
	if err != nil {
		t.Fatal(err)
	}

	ref, rref := loadWebGPUResident(t, dir, opts, 1)
	defer ref.Close()
	want0, err := rref.ForwardN(e0[:4], 0)
	if err != nil {
		t.Fatal(err)
	}
	want0b, err := rref.Forward(e0[4], 4)
	if err != nil {
		t.Fatal(err)
	}
	want1, err := rref.ForwardN(e1, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want0 {
		if !slices.Equal(got0[i], want0[i]) {
			t.Errorf("slot 0 ForwardN row %d differs from a one-slot model's", i)
		}
	}
	for i := range want1 {
		if !slices.Equal(got1[i], want1[i]) {
			t.Errorf("slot 1 ForwardN row %d differs from a one-slot model's — its pool read another slot's KV", i)
		}
	}
	if !slices.Equal(got0b, want0b) {
		t.Error("slot 0's next Forward after slot 1 ran differs — slot 1's rows reached slot 0's KV")
	}
}

// TestWebGPUKVSlots_clampedBuild is a real clamping build: slot 2 of 4 fails its allocation halfway through its layers,
// so 2 are granted, both usable, the half-built slot's buffers are released, and nothing is left allocated after Close.
func TestWebGPUKVSlots_clampedBuild(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-tiny")
	injected := errors.New("injected allocation failure")
	slotAllocHook = func(slot, layer int) error {
		if slot == 2 && layer == 1 {
			return injected
		}
		return nil
	}
	defer func() { slotAllocHook = nil }()
	before := LiveBufferBytes()
	m, rd := loadWebGPUResident(t, dir, decoder.Options{Quant: "int4"}, 4)
	if rd.KVSlots() != 2 || m.ResidentKVSlots() != 2 {
		m.Close()
		t.Fatalf("KVSlots %d, model reports %d — want the 2 that were allocated before the failure", rd.KVSlots(), m.ResidentKVSlots())
	}
	hidden, _, _, _, _, _, _ := m.Dims()
	x := make([]float32, hidden)
	x[0] = 1
	for s := range 2 {
		if err := rd.UseKVSlot(s); err != nil {
			t.Errorf("UseKVSlot(%d): %v", s, err)
		}
		if _, err := rd.Forward(x, 0); err != nil {
			t.Errorf("slot %d Forward: %v", s, err)
		}
	}
	if err := rd.UseKVSlot(2); err == nil {
		t.Error("UseKVSlot(2) succeeded on a 2-slot resident")
	}
	m.Close()
	if after := LiveBufferBytes(); after != before {
		t.Errorf("live device bytes %d before the load, %d after Close — the clamp or the release leaked %d", before, after, after-before)
	}
}

// TestWebGPUKVSlots_pricedAgainstMemory is the darwin (unified-memory) pricing on a real build, once per ceiling: with
// RAM stubbed to where exactly 2 slots fit the fit guard's share, and then with the memory available before the build
// stubbed to where exactly 2 fit, 4 requested grants 2 each time.
func TestWebGPUKVSlots_pricedAgainstMemory(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("the memory pricing applies on darwin only (unified memory); elsewhere a slot is clamped by its allocation")
	}
	dir := filepath.Join("..", "testdata", "llama-tiny")
	m, rd := loadWebGPUResident(t, dir, decoder.Options{Quant: "int4"}, 1)
	perSlot := kvBytesPerSlot(&rd.rm)
	weights, hostCopy := m.ResidentWeightBytes(), m.ResidentHostCopyBytes(0)
	m.Close()
	defer func(r, a func() int64) { hostRAMBytes, hostRAMAvailable = r, a }(hostRAMBytes, hostRAMAvailable)
	const plenty = int64(1) << 40
	for _, tc := range []struct {
		name       string
		ram, avail int64
	}{
		// room for slot 0 and 1.5 more under each ceiling, the other ceiling out of the way
		{"RAM share", int64(float64(weights+hostCopy+perSlot*5/2) / decoder.WeightsMemFraction), plenty},
		{"available before the build", plenty, weights + perSlot*5/2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hostRAMBytes = func() int64 { return tc.ram }
			hostRAMAvailable = func() int64 { return tc.avail }
			m2, rd2 := loadWebGPUResident(t, dir, decoder.Options{Quant: "int4"}, 4)
			defer m2.Close()
			if rd2.KVSlots() != 2 {
				t.Errorf("KVSlots %d with room for 2 (weights %d, host copy %d, %d per slot, RAM %d, available %d) — want 2",
					rd2.KVSlots(), weights, hostCopy, perSlot, tc.ram, tc.avail)
			}
		})
	}
}

// TestDarwinKVSlots pins the unified-memory clamp arithmetic without a device.
func TestDarwinKVSlots(t *testing.T) {
	const mb = int64(1 << 20)
	budget := 10000 * mb
	ram := int64(float64(budget)/decoder.WeightsMemFraction) + 1 // a 10,000 MB RAM share
	for _, tc := range []struct {
		name                       string
		want                       int
		ram, avail                 int64
		weights, hostCopy, perSlot int64
		got                        int
	}{
		{"all fit", 4, ram, 0, 2000 * mb, 2000 * mb, 1000 * mb, 4},                                         // 5000 + 3*1000 = 8000
		{"RAM share clamps", 4, ram, 0, 3500 * mb, 3500 * mb, 1000 * mb, 3},                                // 8000 + 2*1000 = 10000 fits
		{"available clamps, host copy not counted", 4, ram, 5000 * mb, 3000 * mb, 3000 * mb, 1000 * mb, 2}, // 3000+1000+1000 = 5000
		{"available unknown: RAM share only", 4, ram, 0, 3000 * mb, 3000 * mb, 1000 * mb, 4},
		{"only the first fits", 4, ram, 0, 4500 * mb, 4500 * mb, 1000 * mb, 1}, // 10000 + 1000 > 10000
		{"even the first over budget: still 1", 4, ram, 100 * mb, 20000 * mb, 0, mb, 1},
		{"one requested", 1, ram, 0, 100 * mb, 0, mb, 1},
		{"unreadable RAM grants one", 4, 0, 1 << 40, 100 * mb, 0, mb, 1},
	} {
		if got := darwinKVSlots(tc.want, tc.ram, tc.avail, tc.weights, tc.hostCopy, tc.perSlot); got != tc.got {
			t.Errorf("%s: darwinKVSlots(%d, …) = %d, want %d", tc.name, tc.want, got, tc.got)
		}
	}
}

// TestWebGPUKVSlots_perPositionMatchesTheBuild pins slotsBeforeContext's price (kvBytesPerPosition, computed before
// anything is allocated) against what the build really allocates for one slot, on each KV layout: a drift here would
// size the context for slots against the wrong figure.
func TestWebGPUKVSlots_perPositionMatchesTheBuild(t *testing.T) {
	for _, tc := range []struct {
		name, fixture string
		opts          decoder.Options
	}{
		{"llama-tiny-f32", "llama-tiny", decoder.Options{Quant: "int4"}},
		{"llama-tiny-f16", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "f16"}},
		{"llama-tiny-i8", "llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "i8"}},
		{"mistral-tiny-window", "mistral-tiny-window", decoder.Options{Quant: "int4"}},
		{"deepseek-tiny-mla", "deepseek-tiny", decoder.Options{Quant: "int4"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", tc.fixture), tc.opts, 1)
			defer m.Close()
			per := kvBytesPerPosition(m, m.KVCacheF16(), m.KVCacheI8())
			if got, want := kvBytesPerSlot(&rd.rm), per*int64(rd.ctxCap); got != want || got == 0 {
				t.Errorf("one slot allocates %d bytes at %d positions; kvBytesPerPosition prices %d/position = %d", got, rd.ctxCap, per, want)
			}
		})
	}
}

// TestCtxWhereSlotsFit pins slotsBeforeContext's search: unchanged when the slots already fit, the largest fitting
// context otherwise, and the floor when not even the floor fits.
func TestCtxWhereSlotsFit(t *testing.T) {
	under := func(limit int) func(int) bool { return func(ctx int) bool { return ctx <= limit } }
	for _, tc := range []struct {
		name          string
		oneSlot, edge int
		got           int
	}{
		{"all fit at the default", 16384, 20000, 16384},
		{"fits exactly at the default", 16384, 16384, 16384},
		{"shrinks to the largest that fits", 16384, 7999, 7999},
		{"one below the default", 16384, 16383, 16383},
		{"exactly the floor", 16384, webgpuSlotCtxFloor, webgpuSlotCtxFloor},
		{"not even the floor: the floor, and the slots clamp", 16384, 1000, webgpuSlotCtxFloor},
	} {
		if got := ctxWhereSlotsFit(tc.oneSlot, 4, under(tc.edge)); got != tc.got {
			t.Errorf("%s: ctxWhereSlotsFit(%d, fits ≤ %d) = %d, want %d", tc.name, tc.oneSlot, tc.edge, got, tc.got)
		}
	}
}

// TestWebGPUKVSlots_slotsBeforeContext is the owner's rule on a real build (darwin; heavy — qwen2.5-coder-1.5b from
// ~/models): with the memory available before the build stubbed to where 4 slots fit only up to 8000 positions, an
// unpinned load gives up context to exactly that and keeps all 4 slots, while an explicit 16384 keeps its context
// and gets the 1 slot that fits beside it.
func TestWebGPUKVSlots_slotsBeforeContext(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("slots before context applies on darwin only until the discrete-GPU clamp is measured")
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 (loads qwen2.5-coder-1.5b from ~/models)")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s: %v", path, err)
	}
	probe, rd := loadWebGPUResident(t, path, decoder.Options{Quant: "int4"}, 1)
	per := kvBytesPerPosition(probe, false, false)
	weights := probe.ResidentWeightBytes()
	if rd.ctxCap != 16384 {
		probe.Close()
		t.Fatalf("the one-slot build chose %d positions, not WebGPU's 16384 default — this test's arithmetic assumes it", rd.ctxCap)
	}
	probe.Close()
	const edge = 8000
	defer func(r, a func() int64) { hostRAMBytes, hostRAMAvailable = r, a }(hostRAMBytes, hostRAMAvailable)
	hostRAMBytes = func() int64 { return 16 << 30 }
	hostRAMAvailable = func() int64 { return weights + 4*per*edge }

	m, rd := loadWebGPUResident(t, path, decoder.Options{Quant: "int4"}, 4)
	if m.ResidentContextPinned() {
		m.Close()
		t.Fatal("an unrequested context reads as pinned — the rule would never run")
	}
	if rd.ctxCap != edge || rd.KVSlots() != 4 {
		t.Errorf("unpinned: %d positions and %d slots, want %d and 4", rd.ctxCap, rd.KVSlots(), edge)
	}
	m.Close()

	mx, rdx := loadWebGPUResident(t, path, decoder.Options{Quant: "int4", ResidentContext: 16384}, 4)
	defer mx.Close()
	if rdx.ctxCap != 16384 || rdx.KVSlots() != 1 {
		t.Errorf("explicit --ctx 16384: %d positions and %d slots, want 16384 and 1 (an explicit context is never shrunk)", rdx.ctxCap, rdx.KVSlots())
	}
}
