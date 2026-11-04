//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKV2_recordedClassIsTheClassThatRan is G4(a) on Metal (docs/measurements/kv2-2026-10-14.md): for each pass a
// generation ran through the production path, the class the KV manager recorded for its rows is the class the resident
// ran: a cold 1,000-token prompt and a 30-token prompt under the fast-prefill floor are replayed from row 0 on the lane
// their recorded class names (fast: the batched PrefillLast; exact: decode, token by token) and must equal production
// byte for byte; a 5-row continuation is re-run on production's own rows, on each lane (below).
func TestKV2_recordedClassIsTheClassThatRan(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(mc3FixtureWriter(t, 4096), decoder.Options{Backend: "metal", Quant: mc3TestQuant, ResidentContext: 2048, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Skipf("not resident: %s", m.ResidentDecline())
	}
	r := a.r
	rb := kv2RowBytes(r)
	gen := func(prompt []int, maxTok int) []int {
		ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return ids
	}
	embs := func(ids []int) [][]float32 {
		out := make([][]float32, len(ids))
		for i, id := range ids {
			out[i] = m.EmbedResidentForTest(id)
		}
		return out
	}
	expect := func(rows []string, lo, hi int, want string) {
		t.Helper()
		for p := lo; p < hi; p++ {
			if rows[p] != want {
				t.Fatalf("row %d recorded %q, want %q", p, rows[p], want)
			}
		}
	}
	other := func(slot int) int { return 1 - slot }

	// A cold 1,000-token prompt: recorded fast, and equal to the batched pass.
	p1 := make([]int, 1000)
	for i := range p1 {
		p1[i] = 100 + (i*53)%15000
	}
	reply := gen(p1, 16)
	slot, rows := m.KVBoundClassesForTest()
	expect(rows, 0, 1000, "fast")
	expect(rows, 1000, len(rows), "exact")
	prod1 := kv2Slot(t, r, slot, 1000)
	if err := r.useKVSlot(other(slot)); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), embs(p1), 0); err != nil {
		t.Fatal(err)
	}
	if b, p := kv2RowsEqual(kv2Slot(t, r, other(slot), 1000), prod1, rb, 0, 1000); b >= 0 {
		t.Errorf("cold prompt: rows recorded fast differ from the batched pass at buffer %d row %d", b, p)
	}
	// Put the binding back where the decoder believes it is: binding a slot directly, behind the decoder, would send
	// the next generation to the wrong slot.
	if err := r.useKVSlot(slot); err != nil {
		t.Fatal(err)
	}

	// Its 5-row continuation, recorded fast: re-running the batched pass over its suffix, on the same rows, leaves them
	// unchanged, and re-running decode over it changes them (the control: the two lanes are distinguishable here).
	p2 := append(append(slices.Clone(p1), reply...), 16001, 16002, 16003, 16004, 16005)
	gen(p2, 4)
	slot2, rows := m.KVBoundClassesForTest()
	if slot2 != slot || r.kvSlot != slot {
		t.Fatalf("the continuation ran on slot %d (resident bound %d), the turn on %d", slot2, r.kvSlot, slot)
	}
	from := len(p1) + len(reply)
	expect(rows, from, len(p2), "fast")
	prod := kv2Slot(t, r, slot, len(p2))
	if _, err := a.PrefillLast(context.Background(), embs(p2[from:]), from); err != nil {
		t.Fatal(err)
	}
	if b, p := kv2RowsEqual(kv2Slot(t, r, slot, len(p2)), prod, rb, from, len(p2)); b >= 0 {
		t.Errorf("continuation: the batched pass over the same rows rewrites buffer %d row %d: the rows recorded fast are not its", b, p)
	}
	for p := from; p < len(p2); p++ {
		if _, err := a.Forward(embs(p2[p : p+1])[0], p); err != nil {
			t.Fatal(err)
		}
	}
	if b, _ := kv2RowsEqual(kv2Slot(t, r, slot, len(p2)), prod, rb, from, len(p2)); b < 0 {
		t.Error("control: decode over the suffix left the rows unchanged, so this check cannot tell the lanes apart")
	}

	// A 30-token prompt, under the floor: recorded exact, and equal to decode.
	p3 := make([]int, 30)
	for i := range p3 {
		p3[i] = 17000 + i
	}
	gen(p3, 2)
	slot, rows = m.KVBoundClassesForTest()
	expect(rows, 0, 30, "exact")
	prod = kv2Slot(t, r, slot, 30)
	if err := r.useKVSlot(other(slot)); err != nil {
		t.Fatal(err)
	}
	for p, e := range embs(p3) {
		if _, err := a.Forward(e, p); err != nil {
			t.Fatal(err)
		}
	}
	if b, p := kv2RowsEqual(kv2Slot(t, r, other(slot), 30), prod, rb, 0, 30); b >= 0 {
		t.Errorf("short prompt: rows recorded exact differ from decode at buffer %d row %d", b, p)
	}
}
