//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/cost"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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
			if rd.rm.Dnet == nil && rd.rm.Mamba == nil {
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
				for l := range s0.Layers {
					a, b := &s0.Layers[l], &rm.Layers[l]
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
					if a.AttnNorm != b.AttnNorm || a.MlpNorm != b.MlpNorm || a.InvFreq != b.InvFreq {
						t.Errorf("slot %d layer %d does not share slot 0's norms / RoPE table", s, l)
					}
				}
				if rm.FinalNorm != s0.FinalNorm {
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

// TestSlotsGranted is G-WS1's table (docs/tasks/task-subsystems-2026-10.md, "SB2 memory reads, step 4"): slotsGranted is
// darwinKVSlots on darwin, every slot on a discrete GPU with no figure, and the headroom rule with one; slotsBeforeContextOn leaves a
// discrete GPU's context alone and prices darwin's as ctxWhereSlotsFit over darwinKVSlots does.
func TestSlotsGranted(t *testing.T) {
	const mb = int64(1 << 20)
	ram := int64(16 << 30)
	for _, want := range []int{0, 1, 2, 4} {
		for _, avail := range []int64{0, 6 << 30} {
			for _, weights := range []int64{2000 * mb, 9000 * mb} {
				for _, hostCopy := range []int64{0, 2000 * mb} {
					for _, perSlot := range []int64{500 * mb, 3000 * mb} {
						for _, r := range []int64{ram, 0} {
							darwin := max(1, want)
							if want > 1 {
								darwin = darwinKVSlots(want, r, avail, weights, hostCopy, perSlot)
							}
							if got := slotsGranted(want, true, r, avail, weights, hostCopy, perSlot, 0, false); got != darwin {
								t.Errorf("darwin want %d ram %d avail %d weights %d host %d per %d: %d, want darwinKVSlots' %d", want, r, avail, weights, hostCopy, perSlot, got, darwin)
							}
							if got := slotsGranted(want, false, r, avail, weights, hostCopy, perSlot, 0, false); got != max(1, want) {
								t.Errorf("discrete, no figure, want %d: %d, want every slot", want, got)
							}
						}
					}
				}
			}
		}
	}
	const head = int64(webgpuSlotHeadroom)
	for _, tc := range []struct {
		free, per int64
		want, got int
	}{
		{2000 * mb, 500 * mb, 4, 4},         // 2000 - 384 = 1616 holds three more
		{1000 * mb, 500 * mb, 4, 2},         // 616 holds one more
		{300 * mb, 500 * mb, 4, 1},          // under the headroom: none
		{head + 500*mb, 500 * mb, 4, 2},     // exactly one slot and the headroom
		{head + 500*mb - 1, 500 * mb, 4, 1}, // a byte short
	} {
		if got := slotsGranted(tc.want, false, 0, 0, 0, 0, tc.per, tc.free, true); got != tc.got {
			t.Errorf("discrete, %d B free, %d B a slot: %d of %d, want %d", tc.free, tc.per, got, tc.want, tc.got)
		}
	}

	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cpu", Quant: "int4", ResidentKVSlots: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const oneSlot = 32768
	for _, r := range []int64{1 << 30, 64 << 30} {
		if got := slotsBeforeContextOn(false, func() int64 { return r }, m, oneSlot, false, false, 0); got != oneSlot {
			t.Errorf("discrete: context %d, want %d left alone", got, oneSlot)
		}
		perPos := kvBytesPerPosition(m, false, false)
		weights, hostCopy := m.ResidentWeightBytes(), m.ResidentHostCopyBytes(0)
		want := ctxWhereSlotsFit(oneSlot, 4, func(ctx int) bool { return darwinKVSlots(4, r, 0, weights, hostCopy, perPos*int64(ctx)) >= 4 })
		if got := slotsBeforeContextOn(true, func() int64 { return r }, m, oneSlot, false, false, 0); got != want {
			t.Errorf("darwin, %d B RAM: context %d, want %d", r, got, want)
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

// captureStderrGPU swaps os.Stderr for a pipe for the duration of fn and returns what was written.
// Drains the pipe concurrently (unlike a synchronous read-after-fn) because fn here is a real
// multi-GB checkpoint load that can print more than a pipe's buffer before returning. Restores
// os.Stderr and closes the writer via defer, not after a plain fn() call: fn calls t.Fatalf on a
// load failure, which unwinds the goroutine via runtime.Goexit() and would otherwise skip the
// restore, leaving every later test's stderr silently swallowed and the reader goroutine parked
// forever on an unclosed pipe.
func captureStderrGPU(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stderr
	os.Stderr = w
	out := make(chan string, 1)
	go func() {
		b, _ := io.ReadAll(r)
		out <- string(b)
	}()
	// fn runs inside its own deferred restore, not captureStderrGPU's: the restore (and the w.Close
	// that lets the reader goroutine see EOF) must happen BEFORE the return statement below reads
	// from out, and a defer on captureStderrGPU itself would not run until AFTER that read is
	// evaluated — deadlocking the ordinary, non-Fatalf path on every call.
	func() {
		defer func() {
			os.Stderr = orig
			w.Close()
		}()
		fn()
	}()
	return <-out
}

// TestWebGPUKVSlots_realClamp exercises the discrete-GPU clamp on real hardware and a real
// checkpoint (docs/prompts/nobara-mc1-webgpu-2026-09.md, §2, item 26). A slot whose allocation
// fails, or whose headroom probe after it fails, stops buildKVSlots; that path otherwise only
// runs under an injected failure (TestWebGPUKVSlots_clampedBuild). It loads with 4 slots
// requested, runs every granted slot, and checks the device comes back clean.
//
// It records the granted count rather than asserting an expected one, but DOES fail hard on any
// device-lost or validation error, on a slot that is not bit-identical to a fresh one-slot load,
// or on memory that does not return to baseline. A skip here is not a pass; only a missing
// checkpoint file skips.
func TestWebGPUKVSlots_realClamp(t *testing.T) {
	requireHeavyModel(t)
	if runtime.GOOS == "darwin" {
		t.Skip("the discrete-GPU allocation clamp is what this test measures; darwin prices slots against RAM before allocating anything (darwinKVSlots) and never reaches it")
	}
	home, _ := os.UserHomeDir()
	for _, tc := range []struct {
		name, envVar, defaultFile string
	}{
		// Weights plus one slot's KV at the default 16k f32 cache leave room for one slot on an 8 GB
		// card, with slot 2 failing. Checkpoints load from ~/models on NVMe only (CLAUDE.md's
		// model-storage rule: a path under /srv/models or /Volumes/ would measure a 5400 rpm SMR disk).
		{"qwen2.5-7b", "GOINFER_WEBGPU_KVSLOTS_REALCLAMP_7B", "qwen2.5-7b-instruct-q4_k_m.gguf"},
		// Small enough per slot that all 4 requested slots are expected to fit.
		{"qwen2.5-coder-1.5b", "GOINFER_WEBGPU_KVSLOTS_REALCLAMP_1_5B", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := os.Getenv(tc.envVar)
			if path == "" {
				path = filepath.Join(home, "models", tc.defaultFile)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no checkpoint at %s: %v", path, err)
			}
			webgpuRealClampScenario(t, path)
		})
	}
}

// webgpuRealClampScenario is TestWebGPUKVSlots_realClamp's body, factored out so each checkpoint gets its own
// before/after device-memory baseline and its own t.Run scope.
func webgpuRealClampScenario(t *testing.T, path string) {
	t.Helper()
	beforeBytes := LiveBufferBytes()
	beforeVRAM := gpuVRAMMiB(t)

	var m *decoder.Model
	stderr := captureStderrGPU(t, func() {
		var err error
		m, err = decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4", ResidentKVSlots: 4})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
	})
	t.Log(stderr) // t.Fatal inside captureStderrGPU's fn would otherwise lose whatever was already printed
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		m.Close()
		t.Fatalf("%s did not go webgpu-resident: %s", path, m.ResidentDecline())
	}
	n := rd.KVSlots()
	if got := m.ResidentKVSlots(); got != n {
		m.Close()
		t.Fatalf("KVSlots() = %d but ResidentKVSlots() reports %d — the decoder and the resident disagree", n, got)
	}
	switch {
	case strings.Contains(stderr, "MiB left after it"):
		t.Logf("granted %d of 4 requested KV slots — stopped by the %d MiB HEADROOM PROBE after the last one that fit (webgpuSlotHeadroom)", n, webgpuSlotHeadroom>>20)
	case strings.Contains(stderr, "did not fit"):
		t.Logf("granted %d of 4 requested KV slots — stopped by a raw BUFFER ALLOCATION failure (device out of memory before the headroom probe ran)", n)
	default:
		t.Logf("granted %d of 4 requested KV slots — no clamp message, so every requested slot fit", n)
	}

	// Run the same short conversation on every granted slot: >= 32 tokens over >= 2 turns, as
	// kvSlotsScenario's own turns do. A recurrent family would have declined multiple slots long
	// before this point (kvSlotsRequest), so every slot here owns independent KV.
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := []int{7 % vocab, 42 % vocab, 100 % vocab}
	const turns, maxTok = 2, 32
	runConversation := func(gen func(ctx context.Context, ids []int, n int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation)) [][]int {
		cur := slices.Clone(prompt)
		out := make([][]int, turns)
		for turn := range turns {
			ch, g := gen(context.Background(), cur, maxTok, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("turn %d: %v", turn, err)
			}
			out[turn] = ids
			cur = append(append(cur, ids...), 900+turn, 901+turn)
		}
		return out
	}

	got := make([][][]int, n)
	for s := range n {
		if err := rd.UseKVSlot(s); err != nil {
			t.Fatalf("UseKVSlot(%d): %v", s, err)
		}
		got[s] = runConversation(m.Generate)
	}
	m.Close()

	afterBytes := LiveBufferBytes()
	if afterBytes != beforeBytes {
		t.Errorf("live device bytes %d before the load, %d after Close — leaked %d", beforeBytes, afterBytes, afterBytes-beforeBytes)
	}
	if afterVRAM := gpuVRAMMiB(t); beforeVRAM > 0 && afterVRAM > 0 {
		const marginMiB = 128 // cuda/lifecycle_test.go's own post-Close margin, for driver/context bookkeeping
		t.Logf("nvidia-smi used MiB: %d before, %d after (delta %d)", beforeVRAM, afterVRAM, afterVRAM-beforeVRAM)
		if d := afterVRAM - beforeVRAM; d > marginMiB {
			t.Errorf("nvidia-smi memory did not return to baseline: %d MiB before, %d MiB after (delta %d, margin %d) — the device did not give the model back",
				beforeVRAM, afterVRAM, d, marginMiB)
		}
	} else {
		t.Log("nvidia-smi unavailable — device-memory-return check skipped, LiveBufferBytes above still covers the leak")
	}

	// Bit-identical against a fresh one-slot load of the SAME prompt: every slot is independent KV over
	// the same weights, so this is the single reference every granted slot must match, not one per slot.
	ref, _ := loadWebGPUResident(t, path, decoder.Options{Quant: "int4"}, 1)
	defer ref.Close()
	want := runConversation(ref.Generate)
	for s := range n {
		for turn := range turns {
			if !slices.Equal(got[s][turn], want[turn]) {
				t.Errorf("slot %d turn %d: %v, want (one-slot) %v — not bit-identical", s, turn, got[s][turn], want[turn])
			}
		}
	}
}

var (
	ledgerMu     sync.Mutex
	ledgerClosed []*kernels.Context
)

// TestMain runs the ledger's check (I1, I2) on every Context this package's tests close, once, after the tests.
func TestMain(m *testing.M) {
	kernels.ClosedForTest = func(c *kernels.Context) {
		ledgerMu.Lock()
		ledgerClosed = append(ledgerClosed, c)
		ledgerMu.Unlock()
	}
	code := m.Run()
	ledgerMu.Lock()
	bad := kernels.LedgerBreaks(ledgerClosed)
	fmt.Fprintf(os.Stderr, "ledger: %d Context(s) closed in this run\n", len(ledgerClosed))
	ledgerMu.Unlock()
	if len(bad) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d closed Context(s) broke the ledger's invariants (I1 nothing unplanned, I2 close to zero):\n", len(bad))
		for _, b := range bad {
			fmt.Fprintln(os.Stderr, "  "+b)
		}
		code = 1
	}
	os.Exit(code)
}

// TestLedger_webgpuAgreesWithRegistry (G-MM3b-3, G-MM3b-4): on llama-tiny resident on WebGPU, the ledger's open leases equal
// wgpu-native's own count of the buffers the user holds (I3, exact), and the charged bytes after a second and a third identical
// request equal those after the first (I2 within the model's life). The bytes are logged beside the process's VRAM from
// nvidia-smi, for context only: the Vulkan allocator sub-allocates buffers from large blocks.
func TestLedger_webgpuAgreesWithRegistry(t *testing.T) {
	m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Quant: "int4"}, 1)
	defer m.Close()
	var after []int64
	for range 3 {
		ch, g := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 8, decoder.SamplingParams{})
		for range ch {
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		after = append(after, rd.c.Ledger().Usage().Charged)
	}
	if after[1] != after[0] || after[2] != after[0] {
		t.Errorf("the charged bytes grow across identical requests: %v", after)
	}
	u, held := rd.c.Ledger().Usage(), rd.c.BuffersHeld()
	vram := "unread"
	if out, err := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output(); err == nil {
		for _, ln := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			if f := strings.Split(ln, ", "); len(f) == 2 && f[0] == strconv.Itoa(os.Getpid()) {
				vram = f[1] + " MiB"
			}
		}
	}
	t.Logf("ledger: %d open leases, %d bytes charged, by line %v; wgpu-native holds %d buffers; process VRAM %s (context only)",
		u.Leases, u.Charged, u.ByLine, held, vram)
	if uint64(u.Leases) != held {
		t.Errorf("the ledger holds %d leases, wgpu-native %d buffers", u.Leases, held)
	}
}

var sb11Update = flag.Bool("sb11.update", false, "rewrite this architecture's testdata/sb11_decode_llama_tiny.<goarch>.json from this adapter (review the diff)")

// sb11Token is one counted decode token's per-operation calls, launches, submits, syncs and allocations (time is not compared).
type sb11Token map[string][5]int64

func sb11Counts(r cost.Report) sb11Token {
	out := sb11Token{}
	for name, c := range r.Decode.Ops {
		out[name] = [5]int64{c.Calls, c.Launches, c.Submits, c.Syncs, c.Allocs}
	}
	return out
}

// sb11PhaseErrors grades a counted token's phase: its wall time is counted and reaches Timings, and Calls counts dispatches
// only, so it equals the launches (the device counts sit in their own fields).
func sb11PhaseErrors(r cost.Report) []string {
	var errs []string
	d := r.Decode
	if tm := r.Timings(); d.WallNs <= 0 || tm.DecodeMs == nil {
		errs = append(errs, fmt.Sprintf("the token's wall time is %d ns and decode_ms %v: want it counted", d.WallNs, tm.DecodeMs))
	}
	if d.Totals.Calls != d.Totals.Launches {
		errs = append(errs, fmt.Sprintf("totals: %d calls against %d launches; a device count is in Calls", d.Totals.Calls, d.Totals.Launches))
	}
	return errs
}

// countedDecode runs one decode token at pos on rd, counted into a fresh counter.
func countedDecode(t *testing.T, m *decoder.Model, rd *residentDecoder, id, pos int) (cost.Report, []float32) {
	t.Helper()
	var lg []float32
	r := countedWork(t, rd, func() error {
		var e error
		lg, e = rd.Forward(m.EmbedResidentForTest(id), pos)
		return e
	})
	return r, append([]float32(nil), lg...)
}

// countedWork runs work on rd counted into a fresh counter, as one token of the decode phase.
func countedWork(t *testing.T, rd *residentDecoder, work func() error) cost.Report {
	t.Helper()
	c := cost.New()
	if err := rd.costToken(c, cost.Decode, work); err != nil {
		t.Fatal(err)
	}
	return c.Report()
}

// sb11Attributed checks that every dispatch the Context counted was attributed (no dispatch_other) and that some were.
func sb11Attributed(t *testing.T, what string, r cost.Report) {
	t.Helper()
	got := sb11Counts(r)
	if v, other := got["dispatch_other"]; other {
		t.Errorf("%s: %d dispatches the Context counted were not attributed", what, v[1])
	}
	if r.Decode.Totals.Launches == 0 {
		t.Errorf("%s: no dispatch counted", what)
	}
}

// TestCost_webgpuDecodeExact (G-SB11-12, G-SB11-13, part of G-SB11-15): llama-tiny resident on WebGPU.
//   - Every counted path attributes all the dispatches the Context counts at its submits (no dispatch_other): the first token,
//     whose buffer is recorded in line (nothing encoded ahead yet), a prefill token without logits, a batched pass and an
//     on-device draw.
//   - Tokens 2 to 4 each count the same per-pipeline dispatches, submits, syncs and allocations, equal to this architecture's
//     committed golden; a steady token allocates nothing.
//   - Logits match the uncounted token's bit for bit.
//   - The plants: a step recorded twice into the buffer encoded during one token shows as one dispatch_other on the next token,
//     the one that submits it, and not on the token that recorded it; an extra blocking poll moves sync by one.
//
// An architecture without a golden runs every check but the golden's and then skips, naming the counts to record.
func TestCost_webgpuDecodeExact(t *testing.T) {
	m, rd := loadWebGPUResident(t, filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Quant: "int4"}, 1)
	defer m.Close()
	rep, _ := countedDecode(t, m, rd, 1, 0)
	sb11Attributed(t, "the first token", rep)
	var first sb11Token
	for pos := 1; pos <= 3; pos++ {
		rep, _ := countedDecode(t, m, rd, 2+pos, pos)
		for _, e := range sb11PhaseErrors(rep) {
			t.Errorf("token %d: %s", pos, e)
		}
		sb11Attributed(t, fmt.Sprintf("token %d", pos), rep)
		got := sb11Counts(rep)
		if got["alloc"][4] != 0 {
			t.Errorf("token %d allocated %d leases", pos, got["alloc"][4])
		}
		if pos == 1 {
			first = got
			continue
		}
		if !reflect.DeepEqual(got, first) {
			t.Errorf("token %d counts differ from token 1's:\n%v\n%v", pos, got, first)
		}
	}
	var dispatches int64
	for _, v := range first {
		dispatches += v[1]
	}
	t.Logf("a decode token: %d pipelines, %d dispatches, %d submits, %d syncs", len(first)-3, dispatches, first["submit"][2], first["sync"][3])

	// Counting changes nothing: the same token at the same position, uncounted and counted.
	off, err := rd.Forward(m.EmbedResidentForTest(9), 4)
	if err != nil {
		t.Fatal(err)
	}
	off = append([]float32(nil), off...)
	_, on := countedDecode(t, m, rd, 9, 4)
	for i := range on {
		if math.Float32bits(on[i]) != math.Float32bits(off[i]) {
			t.Fatalf("logit %d differs with the counter on", i)
		}
	}

	// The plants (G-SB11-13). The token at 5 encodes the buffer the token at 6 submits.
	defer func() { kernels.CostPlantForTest, costPlantForTest = 0, 0 }()
	kernels.CostPlantForTest = 1
	rep, _ = countedDecode(t, m, rd, 10, 5)
	kernels.CostPlantForTest = 0
	if got := sb11Counts(rep); !reflect.DeepEqual(got, first) {
		t.Errorf("the token that recorded the planted buffer counts differently from a steady one:\n got  %v\n want %v", got, first)
	}
	rep, _ = countedDecode(t, m, rd, 11, 6)
	planted := sb11Counts(rep)
	if got := planted["dispatch_other"]; got != [5]int64{1, 1, 0, 0, 0} {
		t.Errorf("the token that submitted the planted buffer gives dispatch_other %v, want one dispatch", got)
	}
	delete(planted, "dispatch_other")
	if !reflect.DeepEqual(planted, first) {
		t.Errorf("the second-recording plant moved the attribution:\n got  %v\n want %v", planted, first)
	}
	costPlantForTest = 2
	rep, _ = countedDecode(t, m, rd, 12, 7)
	costPlantForTest = 0
	if got := sb11Counts(rep)["sync"][3]; got != first["sync"][3]+1 {
		t.Errorf("the extra-sync plant gives %d syncs, want %d", got, first["sync"][3]+1)
	}

	// The other submitting paths.
	sb11Attributed(t, "a prefill token without logits", countedWork(t, rd, func() error {
		return rd.ForwardNoLogits(m.EmbedResidentForTest(13), 8)
	}))
	sb11Attributed(t, "a batched pass of three", countedWork(t, rd, func() error {
		_, e := rd.ForwardN([][]float32{m.EmbedResidentForTest(14), m.EmbedResidentForTest(15), m.EmbedResidentForTest(16)}, 9)
		return e
	}))
	if rd.SampleAvailable() {
		rep := countedWork(t, rd, func() error {
			_, e := rd.ForwardSample(m.EmbedResidentForTest(17), 12, 0.8, 1, 0)
			return e
		})
		sb11Attributed(t, "an on-device draw", rep)
		if got := rep.Decode.Ops["gumbel_sample"].Launches; got != 2 {
			t.Errorf("an on-device draw counts %d sampling dispatches, want 2", got)
		}
	}

	// One golden per architecture, as the admission census keeps (the adapter can select other pipelines).
	golden := filepath.Join("testdata", "sb11_decode_llama_tiny."+runtime.GOARCH+".json")
	b, _ := json.MarshalIndent(first, "", " ")
	if *sb11Update {
		if err := os.WriteFile(golden, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(golden)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("no %s golden recorded yet; this architecture's counts are %s (a session on such a box records them with "+
			"-sb11.update and reviews them; every other check above ran)", runtime.GOARCH, b)
	}
	if err != nil {
		t.Fatal(err)
	}
	var want sb11Token
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, want) {
		t.Errorf("a decode token's counts differ from the golden:\n got  %v\n want %v", first, want)
	}
}

// costOffContext and costOffDispatch are the off path one dispatch pays, at the registered bound: the Context's three atomic
// adds (one per recording, submit and wait in fact, so a token pays them once) and the runner's nil check.
type costOffContext struct{ dispatches, submits, waits atomic.Int64 }

//go:noinline
func costOffDispatch(c *costOffContext, k *cost.Counter) bool {
	c.dispatches.Add(1)
	c.submits.Add(1)
	c.waits.Add(1)
	return k != nil
}

// TestCost_webgpuOffShare (G-SB11-14): the off path a dispatch pays (costOffDispatch, at its registered bound) × the dispatches
// of a decode token, as a share of a measured decode token on Qwen2.5-Coder-0.5B on WebGPU, is under 0.5%. Heavy; run it under
// the timing lock and note the card's performance state and the box's load.
func TestCost_webgpuOffShare(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no %s", path)
	}
	m, rd := loadWebGPUResident(t, path, decoder.Options{Quant: "int4"}, 1)
	defer m.Close()
	for pos := range 8 {
		if _, err := rd.Forward(m.EmbedResidentForTest(100+pos), pos); err != nil {
			t.Fatal(err)
		}
	}
	rep, _ := countedDecode(t, m, rd, 300, 8)
	dispatches := rep.Decode.Totals.Launches
	const tokens = 64
	start := time.Now()
	for i := range tokens {
		if _, err := rd.Forward(m.EmbedResidentForTest(400+i), 9+i); err != nil {
			t.Fatal(err)
		}
	}
	perToken := time.Since(start) / tokens
	var ctx costOffContext
	hook := testing.Benchmark(func(b *testing.B) {
		for b.Loop() {
			if costOffDispatch(&ctx, nil) {
				b.Fatal("on")
			}
		}
	})
	perDispatch := float64(hook.T.Nanoseconds()) / float64(hook.N) // NsPerOp truncates to whole nanoseconds
	share := perDispatch * float64(dispatches) / float64(perToken.Nanoseconds())
	load, _ := os.ReadFile("/proc/loadavg")
	t.Logf("decode %v a token, %d dispatches a token, the off path %.2f ns a dispatch (%d runs): %.4f%% of a token; load %s",
		perToken, dispatches, perDispatch, hook.N, 100*share, strings.TrimSpace(string(load)))
	if share >= 0.005 {
		t.Errorf("the off path costs %.3f%% of a decode token, at or over the registered 0.5%%", 100*share)
	}
}
