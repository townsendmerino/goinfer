//go:build darwin

package metal

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

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
