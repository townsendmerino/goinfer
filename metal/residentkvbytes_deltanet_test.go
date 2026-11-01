//go:build darwin

package metal

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentKVBytes_excludesDeltaNetLayers gates N-36 (docs/audit-metal-2026-09-12.md): residentKVBytes must not charge every
// layer the model's default kvDim, because Gated-DeltaNet (linear-attention) layers allocate NO KV cache (metal/model.go's
// layer-build loop leaves r.kc[l]/r.vc[l] zero-value for them). testdata/qwen35-tiny is a real 3:1 hybrid (3 linear_attention
// layers, 1 full_attention), a fixture where the bug and the fix give distinctly different answers (4x vs 1x one layer's KV
// bytes).
func TestResidentKVBytes_excludesDeltaNetLayers(t *testing.T) {
	m, err := decoder.Load("../testdata/qwen35-tiny", decoder.Options{Quant: "int4"})
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
		want += 2 * int64(ctxCap) * kvDim * bytesPerElem
	}
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
var deltaNetSlotFixtures = []string{"../testdata/qwen35-tiny", "../testdata/qwen3next-tiny"}

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

// TestRecurrentSlots_deltaNetSequencesKeepTheirOwnState is OP4m-d4a's D4a-1 and D4a-2: on a Gated-DeltaNet model that asked
// for two KV slots, the resident has two, and two sequences decoded one token each in turn on slots 0 and 1 each equal the
// same sequence decoded alone on a one-slot resident, every logit of every position.
func TestRecurrentSlots_deltaNetSequencesKeepTheirOwnState(t *testing.T) {
	for _, dir := range deltaNetSlotFixtures {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			two := recurrentSlotsResident(t, dir, 2)
			if n := two.r.kvSlotCount(); n != 2 {
				t.Fatalf("%d KV slots on a Gated-DeltaNet model that asked for 2", n)
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

// TestRecurrentSlots_sharedStateGoesRed is D4a-2's plant: both slots bound to slot 0's window and state, as one shared
// state would be, and the interleaved sequences no longer equal their solo runs.
func TestRecurrentSlots_sharedStateGoesRed(t *testing.T) {
	dir := deltaNetSlotFixtures[0]
	two := recurrentSlotsResident(t, dir, 2)
	if n := two.r.kvSlotCount(); n != 2 {
		t.Fatalf("%d KV slots on a Gated-DeltaNet model that asked for 2", n)
	}
	for l := range two.r.layers {
		if D := two.r.layers[l].delta; D != nil {
			D.slotWin[1], D.slotState[1] = D.slotWin[0], D.slotState[0]
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
}

// TestRecurrentSlots_keepsOneSlotWhereNotDeclared is D4a-3 and D4a-4 on Metal: a Mamba-2 hybrid, whose mixer
// program.Metal.RecurrentSlots does not list, keeps one KV slot when it asks for two, and so does a Gated-DeltaNet model
// whose request is only the caller's default.
func TestRecurrentSlots_keepsOneSlotWhereNotDeclared(t *testing.T) {
	mamba := recurrentSlotsResident(t, "../testdata/nemotron-tiny", 2)
	if n := mamba.r.kvSlotCount(); n != 1 {
		t.Errorf("the Mamba-2 hybrid has %d KV slots; want 1", n)
	}
	def := recurrentSlotsBuild(t, recurrentSlotsLoad(t, deltaNetSlotFixtures[0], decoder.Options{ResidentKVSlots: 4, ResidentKVSlotsDefault: true}))
	if n := def.r.kvSlotCount(); n != 1 {
		t.Errorf("a Gated-DeltaNet model at the caller's default request has %d KV slots; want 1", n)
	}
}

// TestRecurrentSlots_eachSlotPricesItsState is D4a-5: each further KV slot is priced with its Gated-DeltaNet state beside its
// KV (metalSlotBytes), the state's price equals the buffers allocDeltaNetSlots allocates, and at a ceiling one byte short of a
// second slot's price the build keeps one slot.
func TestRecurrentSlots_eachSlotPricesItsState(t *testing.T) {
	t.Cleanup(func() { metalMemoryCeilingForTest = 0 })
	dir := deltaNetSlotFixtures[0]
	m := recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 2})
	state := metalRecurrentSlotBytes(m)
	if state <= 0 || metalSlotBytes(m) != residentKVBytes(m)+state {
		t.Fatalf("per-slot price %d with a recurrent state of %d and KV of %d", metalSlotBytes(m), state, residentKVBytes(m))
	}
	metalMemoryCeilingForTest = residentNeedBytes(m) + metalSlotBytes(m)
	if n := metalKVSlots(m); n != 2 {
		t.Fatalf("a ceiling of the need plus one slot's price: %d slots; want 2", n)
	}
	two := recurrentSlotsBuild(t, m)
	var got int64
	for l := range two.r.layers {
		if D := two.r.layers[l].delta; D != nil {
			got += 4 * int64(D.slotWin[1].Len()+D.slotState[1].Len())
		}
	}
	if got != state {
		t.Errorf("slot 1's Gated-DeltaNet buffers are %d bytes; metalRecurrentSlotBytes prices %d", got, state)
	}
	m2 := recurrentSlotsLoad(t, dir, decoder.Options{ResidentKVSlots: 2})
	metalMemoryCeilingForTest = residentNeedBytes(m2) + metalSlotBytes(m2) - 1
	if n := metalKVSlots(m2); n != 1 {
		t.Errorf("a ceiling one byte short of a second slot's price: %d slots; want 1", n)
	}
}

// TestRecurrentSlots_checkpointStaysWithItsSlot: the recurrent checkpoint is one copy, saved from the bound slot. Restoring it
// with another slot bound is refused, and restoring it on its own slot brings back that slot's state.
func TestRecurrentSlots_checkpointStaysWithItsSlot(t *testing.T) {
	two := recurrentSlotsResident(t, deltaNetSlotFixtures[0], 2)
	if two.RecurrentCheckpointBytes() <= 0 {
		t.Skipf("the load kept no recurrent checkpoint: %s", two.r.ckptDeclined)
	}
	A, B := recurrentSeqs(two.r, 6)
	if err := two.r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	recurrentSolo(two, A)
	if err := two.SaveRecurrentCheckpoint(); err != nil {
		t.Fatal(err)
	}
	saved := append([]float32(nil), two.r.ckptBufs[1].Floats()...) // the first layer's state
	recurrentSolo(two, B)                                          // slot 1's state moves on
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
}
