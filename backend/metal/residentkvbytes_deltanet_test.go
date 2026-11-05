//go:build darwin

package metal

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// TestResidentKVBytes_excludesDeltaNetLayers gates N-36 (docs/audit-metal-2026-09-12.md): residentKVBytes must not charge every
// layer the model's default kvDim, because Gated-DeltaNet (linear-attention) layers allocate NO KV cache (metal/model.go's
// layer-build loop leaves r.kc[l]/r.vc[l] zero-value for them). testdata/qwen35-tiny is a real 3:1 hybrid (3 linear_attention
// layers, 1 full_attention), a fixture where the bug and the fix give distinctly different answers (4x vs 1x one layer's KV
// bytes).
func TestResidentKVBytes_excludesDeltaNetLayers(t *testing.T) {
	m, err := decoder.Load("../../testdata/qwen35-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
	if !dnetOK {
		t.Fatal("test setup: qwen35-tiny did not resolve to a DeltaNet-hybrid model")
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	var wantAttnLayers, wantLinearLayers int
	for l := range nLayers {
		if m.Qwen35LinearLayer(l) {
			wantLinearLayers++
		} else {
			wantAttnLayers++
		}
	}
	if wantLinearLayers == 0 {
		t.Fatal("test setup: qwen35-tiny has no linear-attention layers — fixture changed, this test no longer discriminates")
	}
	if wantAttnLayers == 0 {
		t.Fatal("test setup: qwen35-tiny has no full-attention layers to charge KV for")
	}
	t.Logf("qwen35-tiny: %d layers, %d linear-attention (no KV), %d full-attention (KV)",
		nLayers, wantLinearLayers, wantAttnLayers)

	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		t.Fatalf("resolveMetalCtxCap: %v", err)
	}

	// Hand-compute the expected total from ONLY the attention layers, using the exact same
	// per-layer accessors residentKVBytes itself reads — this is the independent oracle, not a
	// re-implementation of the function under test.
	const bytesPerElem = 2
	var want int64
	for l := range nLayers {
		if m.Qwen35LinearLayer(l) {
			continue
		}
		kvDim := int64(m.KVHeadsAtResident(l)) * int64(m.HeadDimAtResident(l))
		want += 2 * (int64(kvBlocksFor(ctxCap)) + 1) * decoder.KVBlockRows * kvDim * bytesPerElem // the layer's blocks, and its pool's null block
	}
	want += 4 * int64(kvBlocksFor(ctxCap)) // the slot's table
	if want <= 0 {
		t.Fatal("test setup: expected KV bytes computed as 0 — oracle itself is degenerate")
	}

	got := residentKVBytes(m)
	if got != want {
		// The pre-fix behavior charged every layer, including the linear-attention ones — if the
		// fix regresses, this is what it would look like: got > want by roughly the linear
		// layers' own share.
		t.Errorf("residentKVBytes = %d, want %d (excluding %d DeltaNet layer(s)' KV) — "+
			"got/want ratio %.2f suggests %s",
			got, want, wantLinearLayers, float64(got)/float64(want),
			map[bool]string{true: "DeltaNet layers are still being charged (N-36 regressed)", false: "an unrelated discrepancy"}[got > want])
	}
}

// deltaNetSlotFixtures are the tracked Gated-DeltaNet checkpoints the per-slot state is checked on (OP4m-d4a,
// docs/tasks/task-forward-architecture-2026-10.md §4.9).
var deltaNetSlotFixtures = []string{"../../testdata/qwen35-tiny", "../../testdata/qwen3next-tiny"}

// mambaSlotFixture is the tracked Mamba-2 hybrid (Nemotron-H) the per-slot state is checked on (OP4m-d6a).
const mambaSlotFixture = "../../testdata/nemotron-tiny"

// recurrentSlotFixtures is every tracked fixture whose recurrent state Metal keeps per KV slot.
var recurrentSlotFixtures = append(slices.Clone(deltaNetSlotFixtures), mambaSlotFixture)

// slotState is KV slot s's own conv window and state for every recurrent layer, Gated-DeltaNet's and Mamba-2's, in layer order.
func slotState(r *resident, s int) []Buffer {
	var out []Buffer
	for l := range r.layers {
		if D := r.layers[l].delta; D != nil {
			out = append(out, D.slotWin[s], D.slotState[s])
		}
		if M := r.layers[l].mamba; M != nil {
			out = append(out, M.slotWin[s], M.slotSSM[s])
		}
	}
	return out
}

// recurrentSlotsLoad loads dir at int4 with opts' slot request.
func recurrentSlotsLoad(t *testing.T, dir string, opts decoder.Options) *decoder.Model {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	opts.Quant, opts.ResidentContext = "int4", 256
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// recurrentSlotsBuild builds m's Metal resident.
func recurrentSlotsBuild(t *testing.T, m *decoder.Model) *metalResident {
	t.Helper()
	rf, ok, err := (&metalBackend{}).BuildResident(m)
	if err != nil || !ok {
		t.Fatalf("build: %v", err)
	}
	a := rf.(*metalResident)
	t.Cleanup(func() { a.r.Close() })
	return a
}

// recurrentSlotsResident builds dir's resident with `slots` KV slots asked for explicitly (not the caller's default).
func recurrentSlotsResident(t *testing.T, dir string, slots int) *metalResident {
	t.Helper()
	return recurrentSlotsBuild(t, recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: slots}))
}

// recurrentSeqs is two sequences of n embeddings each, from fixed token ids.
func recurrentSeqs(r *resident, n int) (a, b [][]float32) {
	seed := uint32(86420)
	rnd := func() int {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		return int(seed % uint32(min(20000, r.V)))
	}
	for range n {
		a, b = append(a, mc3Emb(r, rnd())), append(b, mc3Emb(r, rnd()))
	}
	return a, b
}

// recurrentSolo decodes embs as one fresh sequence on the bound slot and returns each position's logits.
func recurrentSolo(a *metalResident, embs [][]float32) [][]float32 {
	a.Reset()
	out := make([][]float32, len(embs))
	for i, e := range embs {
		out[i] = append([]float32(nil), a.r.ForwardEmb(e, i)...)
	}
	return out
}

// recurrentInterleaved decodes A on slot 0 and B on slot 1 one token each in turn, and counts the logits that differ from
// refA and refB bit for bit.
func recurrentInterleaved(t *testing.T, a *metalResident, A, B, refA, refB [][]float32) int {
	t.Helper()
	bind := func(s int) {
		if err := a.r.useKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	bind(0)
	a.Reset()
	bind(1)
	a.Reset()
	differ := 0
	count := func(got, want []float32) {
		for j := range want {
			if j >= len(got) || math.Float32bits(got[j]) != math.Float32bits(want[j]) {
				differ++
			}
		}
	}
	for i := range A {
		bind(0)
		count(a.r.ForwardEmb(A[i], i), refA[i])
		bind(1)
		count(a.r.ForwardEmb(B[i], i), refB[i])
	}
	return differ
}

// TestRecurrentSlots_sequencesKeepTheirOwnState is OP4m-d4a's D4a-1 and D4a-2, and OP4m-d6a's D6a-1 and D6a-2: on a
// Gated-DeltaNet or Mamba-2 model that asked for two KV slots, the resident has two, and two sequences decoded one token each
// in turn on slots 0 and 1 each equal the same sequence decoded alone on a one-slot resident, every logit of every position.
func TestRecurrentSlots_sequencesKeepTheirOwnState(t *testing.T) {
	for _, dir := range recurrentSlotFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			two := recurrentSlotsResident(t, dir, 2)
			if n := two.r.kvSlotCount(); n != 2 {
				t.Fatalf("%d KV slots on a recurrent model that asked for 2", n)
			}
			one := recurrentSlotsResident(t, dir, 1)
			A, B := recurrentSeqs(one.r, 12)
			refA, refB := recurrentSolo(one, A), recurrentSolo(one, B)
			if d := recurrentInterleaved(t, two, A, B, refA, refB); d != 0 {
				t.Errorf("interleaved on two slots: %d logits differ from each sequence's solo run", d)
			}
		})
	}
}

// TestRecurrentSlots_sharedStateGoesRed is D4a-2's and D6a-2's plant: both slots bound to slot 0's window and state, as one
// shared state would be, and the interleaved sequences no longer equal their solo runs.
func TestRecurrentSlots_sharedStateGoesRed(t *testing.T) {
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			two := recurrentSlotsResident(t, dir, 2)
			if n := two.r.kvSlotCount(); n != 2 {
				t.Fatalf("%d KV slots on a recurrent model that asked for 2", n)
			}
			for l := range two.r.layers {
				if D := two.r.layers[l].delta; D != nil {
					D.slotWin[1], D.slotState[1] = D.slotWin[0], D.slotState[0]
				}
				if M := two.r.layers[l].mamba; M != nil {
					M.slotWin[1], M.slotSSM[1] = M.slotWin[0], M.slotSSM[0]
				}
			}
			one := recurrentSlotsResident(t, dir, 1)
			A, B := recurrentSeqs(one.r, 12)
			refA, refB := recurrentSolo(one, A), recurrentSolo(one, B)
			d := recurrentInterleaved(t, two, A, B, refA, refB)
			t.Logf("planted: %d logits differ", d)
			if d == 0 {
				t.Error("with one state shared by both slots, the interleaved sequences still equal their solo runs: the check does not see a shared state")
			}
		})
	}
}

// TestRecurrentSlots_keepsOneSlotAtTheDefaultRequest is D4a-4 and D6a-4 on Metal: a Gated-DeltaNet or Mamba-2 model whose
// request is only the caller's default keeps one KV slot. A recurrent kind Metal's table does not list (KDA) is
// TestSupport_recurrentSlotsCover's alone, since no tracked fixture resident on Metal has one; a backend whose table lists
// none (WebGPU) is TestResidentKVSlotsRequestFor_recurrentSlotsByTable's.
func TestRecurrentSlots_keepsOneSlotAtTheDefaultRequest(t *testing.T) {
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		def := recurrentSlotsBuild(t, recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 4, ResidentKVSlotsDefault: true}))
		if n := def.r.kvSlotCount(); n != 1 {
			t.Errorf("%s at the caller's default request has %d KV slots; want 1", dir, n)
		}
	}
}

// TestRecurrentSlots_eachSlotPricesItsState is D4a-5 and D6a-5. A build of two or more slots prices every slot's recurrent
// state: slot 0's in the need (decoder.Model.ResidentNeedBytes), and each further slot's beside its KV (metalSlotBytes). The
// state's price equals the buffers allocRecurrentSlots allocates. At a ceiling one byte short of the second slot's price,
// slot 0's state included, the build keeps one slot.
func TestRecurrentSlots_eachSlotPricesItsState(t *testing.T) {
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		t.Run(filepath.Base(dir), func(t *testing.T) { recurrentSlotPrice(t, dir) })
	}
}

func recurrentSlotPrice(t *testing.T, dir string) {
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	m := recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 2})
	state := metalRecurrentSlotBytes(m)
	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		t.Fatal(err)
	}
	if state <= 0 || metalSlotBytes(m) != m.MetalKVSlotBytes(ctxCap)+state {
		t.Fatalf("per-slot price %d with a recurrent state of %d and KV of %d", metalSlotBytes(m), state, m.MetalKVSlotBytes(ctxCap))
	}
	if plan := m.Plan("metal", 1<<50, decoder.PlanRequest{Ctx: 64, CtxPinned: true}).StateBytes; plan != state {
		t.Errorf("Plan prices one slot's state at %d bytes; metalRecurrentSlotBytes prices %d", plan, state)
	}
	metalMemoryCeilingForTest = residentNeedBytes(m) + metalSlotBytes(m)
	if n := metalKVSlots(m); n != 2 {
		t.Fatalf("a ceiling of the need and one further slot's price: %d slots; want 2", n)
	}
	two := recurrentSlotsBuild(t, m)
	var got int64
	for _, b := range slotState(two.r, 1) {
		got += 4 * int64(b.Len())
	}
	if got != state {
		t.Errorf("slot 1's recurrent buffers are %d bytes; metalRecurrentSlotBytes prices %d", got, state)
	}
	m2 := recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 2})
	metalMemoryCeilingForTest = residentNeedBytes(m2) + metalSlotBytes(m2) - 1
	if n := metalKVSlots(m2); n != 1 {
		t.Errorf("a ceiling one byte short of a second slot's price: %d slots; want 1", n)
	}
}

// TestRecurrentState_everyBufferReadsTheList: slot 0, every further slot and the checkpoint are all sized from the one list. Moving
// both of its entries moves every buffer of each copy, and the state line then holds exactly those three copies.
func TestRecurrentState_everyBufferReadsTheList(t *testing.T) {
	const extra = 8
	stateSpecsOf = func(m *decoder.Model, l int) []decoder.StateSpec {
		specs := slices.Clone(m.StateSpecs(l))
		for i := range specs {
			specs[i].Floats += extra * (i + 1) // unequal, so a window sized as a state would show
		}
		return specs
	}
	t.Cleanup(func() { stateSpecsOf = (*decoder.Model).StateSpecs })
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			m := recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 2})
			a := recurrentSlotsBuild(t, m)
			if a.r.ckptBytes == 0 || a.r.kvSlotCount() != 2 {
				t.Fatalf("checkpoint %d bytes on %d slots; this case needs a checkpoint on two slots", a.r.ckptBytes, a.r.kvSlotCount())
			}
			var want []int
			for l := range a.r.layers {
				if a.r.layers[l].delta != nil || a.r.layers[l].mamba != nil {
					for _, spec := range stateSpecsOf(m, l) {
						want = append(want, spec.Floats)
					}
				}
			}
			lens := func(bufs []Buffer) []int {
				var out []int
				for _, b := range bufs {
					out = append(out, b.Len())
				}
				return out
			}
			for name, bufs := range map[string][]Buffer{"slot 0": slotState(a.r, 0), "slot 1": slotState(a.r, 1), "the checkpoint": a.r.ckptBufs} {
				if got := lens(bufs); !slices.Equal(got, want) {
					t.Errorf("%s's buffers are %v floats; the moved list holds %v", name, got, want)
				}
			}
			var leased int64
			for _, le := range metalLedgerOf(a.r.d).OpenLeases() {
				if le.Line == mem.LineState {
					leased += le.Requested
				}
			}
			if want := 3 * metalRecurrentSlotBytes(m); leased != want {
				t.Errorf("the state line leased %d bytes; three copies of the moved list are %d", leased, want)
			}
		})
	}
}

// TestRecurrentSlots_decoderChoosesAmongSlots: through decoder.Load on Metal, a Gated-DeltaNet model that asked for two slots
// reports two to the decoder's slot choice (Model.ResidentKVSlots, residentSlotCount), and so does a Mamba-2 hybrid (OP4m-d6a).
func TestRecurrentSlots_decoderChoosesAmongSlots(t *testing.T) {
	for _, c := range []struct {
		dir  string
		want int
	}{{deltaNetSlotFixtures[0], 2}, {deltaNetSlotFixtures[1], 2}, {mambaSlotFixture, 2}} {
		m := recurrentSlotsLoad(t, c.dir, decoder.Options{Backend: "metal", ResidentKVSlots: 2})
		if !m.ResidentActive() {
			t.Fatalf("%s: not resident (%s)", c.dir, m.ResidentDecline())
		}
		if got := m.ResidentKVSlots(); got != c.want {
			t.Errorf("%s: the decoder chooses among %d slots; want %d", c.dir, got, c.want)
		}
	}
}

// TestRecurrentSlots_resetZeroesTheBoundSlot: after a sequence leaves slot 1's recurrent windows and states non-zero, a reset
// with slot 1 bound zeroes them, and leaves slot 0's, dirtied by its own sequence, as they were.
func TestRecurrentSlots_resetZeroesTheBoundSlot(t *testing.T) {
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		t.Run(filepath.Base(dir), func(t *testing.T) { recurrentSlotReset(t, dir) })
	}
}

func recurrentSlotReset(t *testing.T, dir string) {
	two := recurrentSlotsResident(t, dir, 2)
	A, B := recurrentSeqs(two.r, 6)
	bind := func(s int) {
		if err := two.r.useKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	bind(0)
	recurrentSolo(two, A)
	bind(1)
	recurrentSolo(two, B)
	nonZero := func(bufs []Buffer) int {
		n := 0
		for _, b := range bufs {
			for _, v := range b.Floats() {
				if v != 0 {
					n++
				}
			}
		}
		return n
	}
	slot := func(s int) []Buffer { return slotState(two.r, s) }
	before0 := nonZero(slot(0))
	if before0 == 0 || nonZero(slot(1)) == 0 {
		t.Fatalf("after their sequences, slot 0 has %d and slot 1 %d non-zero state elements; want both dirty", before0, nonZero(slot(1)))
	}
	two.Reset() // slot 1 is bound
	if n := nonZero(slot(1)); n != 0 {
		t.Errorf("after a reset with slot 1 bound, %d of its state elements are non-zero", n)
	}
	if n := nonZero(slot(0)); n != before0 {
		t.Errorf("a reset with slot 1 bound changed slot 0's state: %d non-zero elements, %d before", n, before0)
	}
}

// TestRecurrentSlots_checkpointStaysWithItsSlot: the recurrent checkpoint is one copy, saved from the bound slot. Restoring it
// before any save, or with another slot bound, is refused, and restoring it on its own slot brings back that slot's state.
func TestRecurrentSlots_checkpointStaysWithItsSlot(t *testing.T) {
	for _, dir := range []string{deltaNetSlotFixtures[0], mambaSlotFixture} {
		t.Run(filepath.Base(dir), func(t *testing.T) { recurrentSlotCheckpoint(t, dir) })
	}
}

func recurrentSlotCheckpoint(t *testing.T, dir string) {
	two := recurrentSlotsResident(t, dir, 2)
	if two.RecurrentCheckpointBytes() <= 0 {
		t.Skipf("the load kept no recurrent checkpoint: %s", two.r.ckptDeclined)
	}
	A, B := recurrentSeqs(two.r, 6)
	if err := two.RestoreRecurrentCheckpoint(); err == nil { // slot 0 is bound, and nothing has been saved
		t.Error("restoring a checkpoint before any save was not refused")
	}
	if err := two.r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	recurrentSolo(two, A)
	if err := two.SaveRecurrentCheckpoint(); err != nil {
		t.Fatal(err)
	}
	saved := append([]float32(nil), two.r.ckptBufs[1].Floats()...) // the first layer's state
	for i, e := range B {                                          // slot 1's state moves on, with no reset, which would free the checkpoint
		two.r.ForwardEmb(e, len(A)+i)
	}
	if err := two.r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if err := two.RestoreRecurrentCheckpoint(); err == nil {
		t.Error("restoring slot 1's checkpoint with slot 0 bound was not refused")
	}
	if err := two.r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if err := two.RestoreRecurrentCheckpoint(); err != nil {
		t.Fatalf("restoring the checkpoint on its own slot: %v", err)
	}
	live := two.r.recurrentLive()[1].Floats()
	for i := range saved {
		if math.Float32bits(live[i]) != math.Float32bits(saved[i]) {
			t.Fatalf("after the restore, slot 1's state differs from the saved checkpoint at element %d", i)
		}
	}
	if two.RecurrentCheckpointHeld(-1) {
		t.Error("no slot, -1, holds a checkpoint")
	}
	if !two.RecurrentCheckpointHeld(1) || two.RecurrentCheckpointHeld(0) {
		t.Errorf("held: slot 0 %v, slot 1 %v; want slot 1 alone", two.RecurrentCheckpointHeld(0), two.RecurrentCheckpointHeld(1))
	}
	two.Reset() // slot 1 is bound: a new sequence, so its checkpoint is no request's
	if two.RecurrentCheckpointHeld(1) {
		t.Error("a reset left the slot holding its checkpoint")
	}
	if err := two.RestoreRecurrentCheckpoint(); err == nil {
		t.Error("restoring a checkpoint after the slot's reset was not refused")
	}
}
