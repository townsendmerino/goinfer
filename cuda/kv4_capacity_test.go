//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// KV4a-3's capacity gates at tier 0 (docs/measurements/kv4-cuda-2026-10-21.md §2, G3a and G3d), on a copy of llama-attnfa-tiny whose
// window (131,072) is wider than the pool, so the pool is what bounds one conversation.

const kv4CapWindow = 131072

func kv4CapLoad(t *testing.T, slots int) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	dir := kv4LongFixture(t, kv4CapWindow)
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentKVSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	if r.KVSlots() != slots {
		t.Fatalf("built %d slots, want %d", r.KVSlots(), slots)
	}
	return m, r
}

// kv4Ids is n token ids, fixed by salt, none of them the fixture's EOS (2).
func kv4Ids(n, salt, vocab int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = 3 + (i*31+salt*17)%(vocab-3)
	}
	return ids
}

// TestKV4_oneConversationUsesThePool is G3a: with four slots and the other three idle and empty, one conversation prefills and decodes
// until it stops, through the decoder's own generation. Pass: it reaches at least 90% of the pool's positions, and it stops as at the
// context cap (finish_reason "length": Budget clamped, no error).
func TestKV4_oneConversationUsesThePool(t *testing.T) {
	m, r := kv4CapLoad(t, 4)
	pool := r.KVSlots() * r.kvShareBlocks * kvBlock
	t.Logf("share %d positions, pool %d, one conversation's cap %d", r.kvShareCtx, pool, r.ctxCap)
	if r.ctxCap < pool {
		t.Fatalf("one conversation's cap %d is under the pool's %d positions", r.ctxCap, pool)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := kv4Ids(pool-2048, 1, vocab)
	out, gen := m.Generate(context.Background(), prompt, pool, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
	n := 0
	for range out {
		n++
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("the conversation ended with an error, not as at the cap: %v", err)
	}
	reached := len(prompt) + n
	t.Logf("reached %d positions (%.1f%% of the pool), budget %d clamped %v", reached, 100*float64(reached)/float64(pool), gen.Budget, gen.BudgetClamped)
	if reached < pool*9/10 {
		t.Fatalf("reached %d positions, under 90%% of the pool's %d", reached, pool)
	}
	if !gen.BudgetClamped || n != gen.Budget {
		t.Fatalf("emitted %d with budget %d (clamped %v): the turn did not end as at the context cap", n, gen.Budget, gen.BudgetClamped)
	}
}

// kv4Prefill prefills ids into slot s from position from and returns the last logits, or the error.
func kv4Prefill(m *decoder.Model, r *cudaResident, s int, ids []int, from int) ([]float32, error) {
	if err := r.UseKVSlot(s); err != nil {
		return nil, err
	}
	e := make([][]float32, len(ids))
	for i, id := range ids {
		e[i] = append([]float32(nil), m.EmbedResidentForTest(id)...)
	}
	lg, err := r.PrefillLast(context.Background(), e, from)
	return append([]float32(nil), lg...), err
}

// TestKV4_pressureFollowsD5 is G3d: four slots; three idle ones hold conversations of about their share; the fourth, running, grows past
// its share. Pass: (1) the idle slots are released least recently used first, each then answers 0 to ReusableKVSlot for its old prompt,
// and that prompt prefilled again gives the logits it gave before; (2) the borrower, with no idle slot left and another slot running
// within its share, gets ErrKVPoolFull; (3) a
// running conversation within its share that arrives while the borrower holds the pool takes its blocks back and gets the logits it gets
// alone, and the borrower's next write gets ErrKVPoolFull.
func TestKV4_pressureFollowsD5(t *testing.T) {
	m, r := kv4CapLoad(t, 4)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	conv := share - 1024 // an idle conversation: a little under its share
	busy := func(s int, b bool) { r.SetKVSlotBusy(s, b) }

	first := map[int][]float32{}
	for _, s := range []int{1, 2, 3} { // least recently used first: 1, then 2, then 3
		lg, err := kv4Prefill(m, r, s, kv4Ids(conv, s, vocab), 0)
		if err != nil {
			t.Fatalf("slot %d: %v", s, err)
		}
		first[s] = lg
	}
	busy(0, true)
	grow := kv4Ids(4*share, 9, vocab)
	if _, err := kv4Prefill(m, r, 0, grow[:conv], 0); err != nil {
		t.Fatalf("slot 0 within its share: %v", err)
	}
	// Past its share by enough to need slot 1's blocks and not slot 2's.
	need := conv + (share - conv) + 3*(share-conv) + conv/2
	if _, err := kv4Prefill(m, r, 0, grow[conv:need], conv); err != nil {
		t.Fatalf("slot 0 borrowing to %d: %v", need, err)
	}
	want := map[int]int{1: 0, 2: conv, 3: conv}
	for s, w := range want {
		if got := r.ReusableKVSlot(s, conv); got != w {
			t.Errorf("slot %d answers %d for its %d-token prompt, want %d (released least recently used first)", s, got, conv, w)
		}
	}
	if lg, err := kv4Prefill(m, r, 1, kv4Ids(conv, 1, vocab), 0); err != nil {
		t.Errorf("slot 1's prompt again: %v", err)
	} else if j, ok := sameBits(lg, first[1]); !ok {
		t.Errorf("slot 1's prompt prefilled again differs at %d from its first run", j)
	}
	// (2) slot 3 starts running (within its share, so nothing can be taken from it); the borrower grows to the whole pool, which
	// it can only get by releasing every idle slot, and then not.
	busy(3, true)
	_, err := kv4Prefill(m, r, 0, grow[need:4*share], need)
	if !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Fatalf("the borrower past every idle slot: got %v, want ErrKVPoolFull", err)
	}
	// (3) a conversation within its share arrives while the borrower holds the pool.
	held := r.kvHeld(0)
	busy(2, true)
	alone := kv4Ids(conv, 7, vocab)
	got, err := kv4Prefill(m, r, 2, alone, 0)
	if err != nil {
		t.Fatalf("a conversation within its share failed while the borrower held %d blocks: %v", held, err)
	}
	_, ra := kv4CapLoad(t, 4)
	ref, err := kv4Prefill(m, ra, 0, alone, 0)
	if err != nil {
		t.Fatal(err)
	}
	if j, ok := sameBits(got, ref); !ok {
		t.Errorf("the conversation within its share differs at %d from the same prompt alone", j)
	}
	if _, err := kv4Prefill(m, r, 0, grow[need:need+64], need); !errors.Is(err, decoder.ErrKVPoolFull) {
		t.Errorf("the stopped borrower's next write: got %v, want ErrKVPoolFull", err)
	}
	t.Logf("share %d positions; the borrower held %d blocks when the conversation within its share arrived", share, held)
}

// kv7Load is kv4CapLoad's model with a host budget in bytes.
func kv7Load(t *testing.T, slots int, hostBytes int64) (*decoder.Model, *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	m, err := decoder.Load(kv4LongFixture(t, kv4CapWindow), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentKVSlots: slots, KVHostBytes: hostBytes, KVDemote: hostBytes > 0})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, mustResident(t, m)
}

// kv7Release leaves slot 1 released by pressure from slot 0, and slots 2 and 3, holding the same conv rows, not. Slot 2 was written last, so
// it is the last an idle release takes: slot 3 goes when slot 1's promotion needs blocks. It returns conv and the ids.
func kv7Release(t *testing.T, m *decoder.Model, r *cudaResident) (conv int, ids []int) {
	t.Helper()
	return kv7ReleaseShort(t, m, r, 0)
}

// kv7ReleaseShort is kv7Release with slot 1 written again from position 0 with its first short rows before the pressure, when short is
// above 0: its table still names conv's blocks, and only short rows are valid.
func kv7ReleaseShort(t *testing.T, m *decoder.Model, r *cudaResident, short int) (conv int, ids []int) {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	conv = share - 1024
	ids = kv4Ids(conv+128, 1, vocab)
	for _, s := range []int{1, 3, 2} { // least recently used first
		if _, err := kv4Prefill(m, r, s, ids[:conv], 0); err != nil {
			t.Fatalf("slot %d: %v", s, err)
		}
		if s == 1 && short > 0 {
			if _, err := kv4Prefill(m, r, s, ids[:short], 0); err != nil {
				t.Fatalf("slot %d's short sequence: %v", s, err)
			}
		}
	}
	r.SetKVSlotBusy(0, true)
	t.Cleanup(func() { r.SetKVSlotBusy(0, false) })
	grow := kv4Ids(4*share, 9, vocab)
	need := conv + (share - conv) + 3*(share-conv) + conv/2 // past its share by enough to need slot 1's blocks and not slot 2's
	if _, err := kv4Prefill(m, r, 0, grow[:need], 0); err != nil {
		t.Fatalf("slot 0 growing to %d: %v", need, err)
	}
	if !r.kvReleased(1) || r.kvReleased(2) {
		t.Fatalf("released: slot 1 %v, slot 2 %v; want only slot 1", r.kvReleased(1), r.kvReleased(2))
	}
	return conv, ids
}

// TestKV7_demoteAndPromote is gates 1 and 3 of docs/measurements/kv7-demote-2026-11/README.md: a released idle slot keeps its rows in host
// memory, still answers ReusableKVSlot, and promoted back continues with logits bit-identical to a slot holding the same rows that was
// never released; the bytes return to the budget. A plant that loses the rows (a zeroed host copy) must differ.
func TestKV7_demoteAndPromote(t *testing.T) {
	for _, plant := range []bool{false, true} {
		m, r := kv7Load(t, 4, 1<<30)
		conv, ids := kv7Release(t, m, r)
		if got := r.ReusableKVSlot(1, conv); got != conv {
			t.Fatalf("released slot 1 answers %d for its %d rows, want them", got, conv)
		}
		st, _ := m.KVSnapshotStats()
		held := r.kvDemoted[1].Bytes()
		if st.Demoted != 1 || st.Used != held || held <= 0 {
			t.Fatalf("stats after the release: demoted %d, used %d; the copy is %d bytes", st.Demoted, st.Used, held)
		}
		if plant {
			clear(r.kvDemoted[1].host.Slice())
		}
		blocks, err := r.PromoteKV(1, 0)
		if err != nil {
			t.Fatalf("PromoteKV: %v", err)
		}
		if want := (conv + kvBlock - 1) / kvBlock; blocks != want {
			t.Errorf("promoted %d blocks, want %d", blocks, want)
		}
		if r.kvReleased(1) || r.kvDemoted[1] != nil {
			t.Errorf("after the promotion: released %v, copy %v", r.kvReleased(1), r.kvDemoted[1] != nil)
		}
		// Slot 1's bytes went back; what is charged is the copy of slot 3, which the promotion's blocks released.
		var kept int64
		for _, d := range r.kvDemoted {
			if d != nil {
				kept += d.Bytes()
			}
		}
		if st, _ := m.KVSnapshotStats(); st.Used != kept || r.kvReleased(2) {
			t.Errorf("after the promotion %d host bytes are charged, %d are kept; the control slot 2 released: %v", st.Used, kept, r.kvReleased(2))
		}
		got, err := kv4Prefill(m, r, 1, ids[conv:], conv)
		if err != nil {
			t.Fatalf("slot 1 continuing: %v", err)
		}
		want, err := kv4Prefill(m, r, 2, ids[conv:], conv)
		if err != nil {
			t.Fatalf("slot 2 continuing: %v", err)
		}
		j, same := sameBits(got, want)
		switch {
		case plant && same:
			t.Error("plant: a zeroed host copy continues bit-identically: the gate reads nothing")
		case !plant && !same:
			t.Errorf("a promoted slot's continuation differs at %d from a slot never released", j)
		}
	}
}

// TestKV7_budgetAndNewSequence is gate 3's other half: a budget too small for the rows releases as before and counts the refusal; and a new
// sequence on a demoted slot drops the copy and returns its bytes.
func TestKV7_budgetAndNewSequence(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<10)
	conv, _ := kv7Release(t, m, r)
	if got := r.ReusableKVSlot(1, conv); got != 0 || r.kvDemoted[1] != nil {
		t.Errorf("a budget too small: slot 1 answers %d, copy kept %v; want 0 and none", got, r.kvDemoted[1] != nil)
	}
	st, _ := m.KVSnapshotStats()
	if st.Declined["demote budget"] != 1 || st.Demoted != 0 || st.Used != 0 {
		t.Errorf("a budget too small: declined %v, demoted %d, used %d", st.Declined, st.Demoted, st.Used)
	}
	m, r = kv7Load(t, 4, 1<<30)
	_, ids := kv7Release(t, m, r)
	if r.kvDemoted[1] == nil {
		t.Fatal("no copy kept")
	}
	r.SetKVSlotBusy(0, false)
	if _, err := kv4Prefill(m, r, 1, ids[:256], 0); err != nil {
		t.Fatalf("a new sequence on slot 1: %v", err)
	}
	if st, _ := m.KVSnapshotStats(); r.kvDemoted[1] != nil || st.Used != 0 {
		t.Errorf("after a new sequence: copy kept %v, %d host bytes charged", r.kvDemoted[1] != nil, st.Used)
	}
}

// TestKV7_generationReportsThePromotion is gate 2, through the decoder's own generation: a conversation whose slot the pool released
// continues with reuse above zero and its blocks promoted, reported on the Generation; without a host budget the same sequence reuses
// nothing, as before this change.
func TestKV7_generationReportsThePromotion(t *testing.T) {
	gen := func(m *decoder.Model, ids []int) *decoder.Generation {
		out, g := m.Generate(context.Background(), ids, 1, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
		for range out {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
		return g
	}
	for _, c := range []struct {
		name   string
		budget int64
	}{{"a host budget", 1 << 30}, {"no host budget", 0}} {
		m, r := kv7Load(t, 2, c.budget)
		_, _, _, _, _, _, vocab := m.Dims()
		share := r.kvShareBlocks * kvBlock
		a := kv4Ids(share-1024, 1, vocab)
		gen(m, a)
		// A long prompt of another conversation, past its share by more than the free blocks: it must take the idle slot's.
		gen(m, kv4Ids(2*share-512, 2, vocab))
		next := append(append([]int(nil), a...), kv4Ids(256, 3, vocab)...)
		g := gen(m, next)
		t.Logf("%s: reused %d of %d, promoted %d blocks (%d rows)", c.name, g.PrefillReused, len(next), g.KVPromotedBlocks, g.KVPromotedRows)
		if c.budget > 0 && (g.PrefillReused <= 0 || g.KVPromotedBlocks != (len(a)+kvBlock-1)/kvBlock || g.KVPromotedRows != g.PrefillReused) {
			t.Errorf("%s: reused %d, promoted %d blocks and %d rows of a %d-token conversation", c.name, g.PrefillReused, g.KVPromotedBlocks, g.KVPromotedRows, len(a))
		}
		if c.budget == 0 && (g.PrefillReused != 0 || g.KVPromotedBlocks != 0) {
			t.Errorf("%s: reused %d, promoted %d blocks; want nothing", c.name, g.PrefillReused, g.KVPromotedBlocks)
		}
	}
}

// kv7PromoteAll promotes every row of slot's host copy.
func kv7PromoteAll(r *cudaResident, slot int) error {
	_, err := r.PromoteKV(slot, 0)
	return err
}

// kv7FillPool maps slot's blocks from position lo until the full group has no free block. Executor job.
func kv7FillPool(t *testing.T, r *cudaResident, slot, lo int) {
	t.Helper()
	if err := r.do(func() error {
		p := &r.kvPools[kvGroupFull]
		return r.kvMap(slot, kvGroupFull, lo, lo+(p.blocks-1-p.inUse())*kvBlock)
	}); err != nil {
		t.Fatalf("filling the pool from slot %d: %v", slot, err)
	}
	if p := &r.kvPools[kvGroupFull]; p.inUse() != p.blocks-1 {
		t.Fatalf("pool not full: %d of %d", p.inUse(), p.blocks-1)
	}
}

// kv7Hole unmaps block b of slot's table, so a copy from slot must allocate for that block.
func kv7Hole(t *testing.T, r *cudaResident, slot, b int) {
	t.Helper()
	if err := r.do(func() error {
		ids := r.kvTabs[slot].ids[kvGroupFull]
		if ids[b] == 0 {
			return fmt.Errorf("block %d of slot %d is not mapped", b, slot)
		}
		r.kvPools[kvGroupFull].unref(ids[b])
		ids[b] = 0
		r.kvTabs[slot].dirty[kvGroupFull] = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestKV7_copyFailureKeepsDemotedDst: a copy onto a demoted slot that fails part way leaves the slot released with its host copy, so the
// request can still promote it, and holds none of the blocks the copy had mapped. The conversation then continues bit-identically.
func TestKV7_copyFailureKeepsDemotedDst(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	conv, ids := kv7Release(t, m, r)
	st, _ := m.KVSnapshotStats()
	used := st.Used
	r.SetKVSlotBusy(2, true)
	r.SetKVSlotBusy(3, true)
	t.Cleanup(func() { r.SetKVSlotBusy(2, false); r.SetKVSlotBusy(3, false) })
	hole := r.kvShareBlocks + 1 // past dst's share, so the block cannot be taken back from a borrower
	if r.kvHeld(0) <= hole {
		t.Fatalf("slot 0 holds %d blocks, the plan needs more than %d", r.kvHeld(0), hole)
	}
	high := r.kvHeld(0) * kvBlock
	kv7Hole(t, r, 0, hole)
	kv7FillPool(t, r, 0, high)
	if err := r.UseKVSlot(1); err != nil {
		t.Fatal(err)
	}
	err := r.CopyKV(0, 1, 0, (hole+1)*kvBlock)
	if err == nil {
		t.Fatal("the copy did not fail: the test reads nothing")
	}
	t.Logf("the copy failed: %v", err)
	st, _ = m.KVSnapshotStats()
	if !r.kvReleased(1) || r.kvDemoted[1] == nil || r.kvHeld(1) != 0 || st.Used != used {
		t.Fatalf("after the failed copy: released %v, copy kept %v, %d blocks held, %d host bytes charged (was %d)",
			r.kvReleased(1), r.kvDemoted[1] != nil, r.kvHeld(1), st.Used, used)
	}
	if got := r.ReusableKVSlot(1, conv); got != conv {
		t.Errorf("slot 1 answers %d for its %d rows", got, conv)
	}
	if err := kv7PromoteAll(r, 1); err != nil {
		t.Fatal(err)
	}
	got, err := kv4Prefill(m, r, 1, ids[conv:], conv)
	if err != nil {
		t.Fatal(err)
	}
	want, err := kv4Prefill(m, r, 2, ids[conv:], conv)
	if err != nil {
		t.Fatal(err)
	}
	if j, same := sameBits(got, want); !same {
		t.Errorf("after the failed copy and the promotion, the continuation differs at %d", j)
	}
}

// TestKV7_copySourceHeldDuringCopy: a block the copy must allocate cannot release the idle slot it is copying from.
func TestKV7_copySourceHeldDuringCopy(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m, r)
	r.SetKVSlotBusy(3, true) // slot 2 is the only idle slot, and so the one a release would take
	t.Cleanup(func() { r.SetKVSlotBusy(3, false) })
	high := r.kvHeld(0) * kvBlock
	kv7Hole(t, r, 2, 5)
	kv7FillPool(t, r, 0, high)
	if err := r.UseKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if err := r.CopyKV(2, 1, 0, conv); err != nil {
		t.Fatalf("the copy: %v", err)
	}
	if r.kvReleased(2) {
		t.Error("the copy released its own source")
	}
	if r.kvSlotBusy[2] {
		t.Error("the copy left its source marked busy")
	}
}

// TestKV7_promoteCountsAsUse: a promoted slot is the most recently used, so the next pressure releases another idle slot, not it.
func TestKV7_promoteCountsAsUse(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	kv7Release(t, m, r)
	if err := kv7PromoteAll(r, 1); err != nil {
		t.Fatal(err)
	}
	if !r.kvReleased(3) || r.kvReleased(2) {
		t.Fatalf("after the promotion: slot 3 released %v, slot 2 released %v; want slot 3 only", r.kvReleased(3), r.kvReleased(2))
	}
	var ok bool
	if err := r.do(func() error { ok = r.kvEvictIdle(0); return nil }); err != nil || !ok {
		t.Fatalf("kvEvictIdle: %v, %v", ok, err)
	}
	if r.kvReleased(1) || !r.kvReleased(2) {
		t.Errorf("the next release took slot 1 %v, slot 2 %v; want slot 2", r.kvReleased(1), r.kvReleased(2))
	}
}

// TestKV7_demoteKeepsValidRows: a slot written again from position 0 with a shorter sequence still names its old blocks. Its host copy holds
// the valid rows, not those.
func TestKV7_demoteKeepsValidRows(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	const short = 1024
	if _, _ = kv7ReleaseShort(t, m, r, short); r.kvDemoted[1] == nil {
		t.Fatal("no copy kept")
	}
	if got := r.kvDemoted[1].rows; got != short {
		t.Errorf("the copy holds %d rows of a %d-row sequence", got, short)
	}
}

// TestKV7_demoteFailureCountsNothing: a copy that fails returns its bytes and counts no demotion.
func TestKV7_demoteFailureCountsNothing(t *testing.T) {
	kvDemoteFailForTest = errors.New("plant")
	t.Cleanup(func() { kvDemoteFailForTest = nil })
	m, r := kv7Load(t, 4, 1<<30)
	kv7Release(t, m, r)
	st, _ := m.KVSnapshotStats()
	if st.Demoted != 0 || st.Used != 0 || r.kvDemoted[1] != nil {
		t.Errorf("after a failed copy: demoted %d, %d host bytes charged, copy %v", st.Demoted, st.Used, r.kvDemoted[1] != nil)
	}
}

// TestKV7_promoteFailureLeavesTheCopy: a promotion the pool cannot serve leaves the slot released, unmapped, with its host copy charged.
func TestKV7_promoteFailureLeavesTheCopy(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	conv := share + share/4
	if _, err := kv4Prefill(m, r, 1, kv4Ids(conv, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	r.SetKVSlotBusy(0, true)
	r.SetKVSlotBusy(2, true)
	r.SetKVSlotBusy(3, true)
	t.Cleanup(func() { r.SetKVSlotBusy(0, false); r.SetKVSlotBusy(2, false); r.SetKVSlotBusy(3, false) })
	if err := r.UseKVSlot(0); err != nil {
		t.Fatal(err)
	}
	// Slots 2 and 3 hold their shares and slot 0 the rest, which is one share past its own once slot 1's blocks are released.
	for _, s := range []int{2, 3} {
		if err := r.do(func() error { return r.kvMap(s, kvGroupFull, 0, share) }); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.do(func() error {
		p := &r.kvPools[kvGroupFull]
		return r.kvMap(0, kvGroupFull, 0, (p.blocks-1-p.inUse())*kvBlock)
	}); err != nil {
		t.Fatal(err)
	}
	if err := r.do(func() error {
		_, e := r.kvTake(0, kvGroupFull, false)
		return e
	}); err != nil {
		t.Fatal(err)
	}
	if !r.kvReleased(1) || r.kvDemoted[1] == nil {
		t.Fatalf("slot 1: released %v, copy %v", r.kvReleased(1), r.kvDemoted[1] != nil)
	}
	kv7FillPool(t, r, 0, r.kvHeld(0)*kvBlock)
	st, _ := m.KVSnapshotStats()
	used := st.Used
	err := kv7PromoteAll(r, 1)
	if err == nil {
		t.Fatal("the promotion did not fail: the test reads nothing")
	}
	t.Logf("the promotion failed: %v", err)
	st, _ = m.KVSnapshotStats()
	if !r.kvReleased(1) || r.kvHeld(1) != 0 || r.kvDemoted[1] == nil || st.Used != used {
		t.Errorf("after the failed promotion: released %v, %d blocks held, copy kept %v, %d host bytes charged (was %d)",
			r.kvReleased(1), r.kvHeld(1), r.kvDemoted[1] != nil, st.Used, used)
	}
}

// TestKV7_copyFromDemoted: a request whose prefix another conversation's demoted slot holds promotes that slot, copies from it and reports
// both: the rows as shared, the blocks and the promotion in the counts.
func TestKV7_copyFromDemoted(t *testing.T) {
	gen := func(m *decoder.Model, ids []int) *decoder.Generation {
		out, g := m.Generate(context.Background(), ids, 1, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
		for range out {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
		return g
	}
	m, r := kv7Load(t, 3, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	a := kv4Ids(share-1024, 1, vocab)
	gen(m, a)
	gen(m, kv4Ids(2*share+2048, 2, vocab))
	if r.kvDemoted[0] == nil {
		t.Fatalf("slot 0 not demoted: released %v", r.kvReleased(0))
	}
	const lead = 1024 // below half of a, so the request does not take a's slot in place
	c := append(append([]int(nil), a[:lead]...), kv4Ids(4096, 3, vocab)...)
	g := gen(m, c)
	st, _ := m.KVSnapshotStats()
	t.Logf("reused %d, shared %d, promoted %d blocks (%d rows); declined %q; stats promoted %d, %d blocks", g.PrefillReused, g.PrefillShared, g.KVPromotedBlocks, g.KVPromotedRows, g.KVCopyDeclined, st.Promoted, st.PromotedBlocks)
	if g.PrefillShared != lead || g.PrefillReused != lead {
		t.Errorf("reused %d, shared %d; want %d of both", g.PrefillReused, g.PrefillShared, lead)
	}
	wantBlocks := (len(a) + 1 + kvBlock - 1) / kvBlock // the source comes back whole, and a's generated token is its last valid row
	if g.KVPromotedBlocks != wantBlocks || st.Promoted != 1 || st.PromotedBlocks != int64(wantBlocks) {
		t.Errorf("promoted: the generation %d blocks, the stats %d times, %d blocks; want %d blocks once", g.KVPromotedBlocks, st.Promoted, st.PromotedBlocks, wantBlocks)
	}
	if g.KVPromotedRows != 0 {
		t.Errorf("the generation reports %d rows from host memory; they reached it through the copy, as shared", g.KVPromotedRows)
	}
	if r.kvReleased(0) || r.kvDemoted[0] != nil {
		t.Errorf("the source: released %v, copy %v", r.kvReleased(0), r.kvDemoted[0] != nil)
	}
}

// TestKV7_copyFromDemotedIsBitExact: rows copied from a promoted source continue bit-identically with the same rows copied from a slot never
// released. A plant that loses the source's rows must differ.
func TestKV7_copyFromDemotedIsBitExact(t *testing.T) {
	for _, plant := range []bool{false, true} {
		m, r := kv7Load(t, 4, 1<<30)
		conv, ids := kv7Release(t, m, r)
		if plant {
			clear(r.kvDemoted[1].host.Slice())
		}
		if blocks, err := r.PromoteKV(1, 0); err != nil || blocks != conv/kvBlock {
			t.Fatalf("promoting the source: %d blocks, %v", blocks, err)
		}
		r.SetKVSlotBusy(1, true) // the continuations below take blocks from slot 0, the borrower, and not from these
		r.SetKVSlotBusy(2, true)
		t.Cleanup(func() { r.SetKVSlotBusy(1, false); r.SetKVSlotBusy(2, false) })
		const hit = 4096
		if err := r.UseKVSlot(3); err != nil {
			t.Fatal(err)
		}
		if err := r.CopyKV(1, 3, 0, hit); err != nil {
			t.Fatal(err)
		}
		got, err := kv4Prefill(m, r, 3, ids[hit:conv], hit)
		if err != nil {
			t.Fatal(err)
		}
		want, err := kv4Prefill(m, r, 2, ids[hit:conv], hit)
		if err != nil {
			t.Fatal(err)
		}
		j, same := sameBits(got, want)
		switch {
		case plant && same:
			t.Error("plant: a zeroed source continues bit-identically: the gate reads nothing")
		case !plant && !same:
			t.Errorf("copied from a promoted source, the continuation differs at %d", j)
		}
	}
}

// TestKV7_copyOntoDemoted: a request that binds a demoted slot and copies a longer run from another takes the whole run from the source, and
// reports it all as shared; the slot's own stale copy goes, with its bytes.
func TestKV7_copyOntoDemoted(t *testing.T) {
	gen := func(m *decoder.Model, ids []int) *decoder.Generation {
		out, g := m.Generate(context.Background(), ids, 1, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
		for range out {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
		return g
	}
	m, r := kv7Load(t, 3, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	p, q := kv4Ids(2048, 1, vocab), kv4Ids(4096, 2, vocab)
	lead := append(append([]int(nil), p...), q...)
	long := append(append([]int(nil), lead...), kv4Ids(8000, 4, vocab)...)
	gen(m, long)                                                                     // slot 0: holds the longer run, and gives up too much to be taken in place
	gen(m, append(append([]int(nil), p...), kv4Ids(share-1024-len(p), 3, vocab)...)) // slot 1: shares p
	gen(m, append(append([]int(nil), long...), kv4Ids(64, 7, vocab)...))             // slot 0 again, so slot 1 is the least recently used
	gen(m, kv4Ids(share+5616, 5, vocab))                                             // slot 2: releases slot 1 and no more
	if r.kvDemoted[1] == nil || r.kvReleased(0) {
		t.Fatalf("slot 1's copy %v, slot 0 released %v; want only slot 1 released", r.kvDemoted[1] != nil, r.kvReleased(0))
	}
	g := gen(m, append(append([]int(nil), lead...), kv4Ids(1024, 6, vocab)...))
	st, _ := m.KVSnapshotStats()
	t.Logf("slot %d: reused %d, shared %d, promoted %d blocks (%d rows); declined %q; %d host bytes charged", r.kvSlot, g.PrefillReused, g.PrefillShared, g.KVPromotedBlocks, g.KVPromotedRows, g.KVCopyDeclined, st.Used)
	if r.kvSlot != 1 {
		t.Fatalf("the request bound slot %d: the test reads nothing", r.kvSlot)
	}
	if g.PrefillReused != len(lead) || g.PrefillShared != len(lead) || g.KVPromotedBlocks != 0 || g.KVPromotedRows != 0 {
		t.Errorf("reused %d, shared %d, promoted %d blocks and %d rows; want %d reused and shared, nothing promoted", g.PrefillReused, g.PrefillShared, g.KVPromotedBlocks, g.KVPromotedRows, len(lead))
	}
	if r.kvReleased(1) || r.kvDemoted[1] != nil {
		t.Errorf("slot 1: released %v, copy %v", r.kvReleased(1), r.kvDemoted[1] != nil)
	}
}

// TestKV7_demoteIsOffWithoutItsOption: a host budget alone keeps prefix snapshots and no released conversation's rows.
func TestKV7_demoteIsOffWithoutItsOption(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(kv4LongFixture(t, kv4CapWindow), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentKVSlots: 4, KVHostBytes: 1 << 30})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	conv, _ := kv7Release(t, m, r)
	if got := r.ReusableKVSlot(1, conv); got != 0 || r.kvDemoted[1] != nil {
		t.Errorf("slot 1 answers %d, copy %v; want 0 and none", got, r.kvDemoted[1] != nil)
	}
	if st, ok := m.KVSnapshotStats(); !ok || st.Demoted != 0 || st.Budget != 1<<30 {
		t.Errorf("snapshot stats %v: %+v; want a budget and no demotion", ok, st)
	}
}

// TestKV7_olderCopyYieldsToANewerOne: a demotion with no room drops the oldest demoted copy, so an idle conversation that never returns does
// not hold the budget. The budget here holds one copy.
func TestKV7_olderCopyYieldsToANewerOne(t *testing.T) {
	m0, r0 := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m0, r0)
	one := r0.kvDemoted[1].Bytes()
	m, r := kv7Load(t, 4, one+one/2)
	kv7Release(t, m, r)
	if r.kvDemoted[1] == nil {
		t.Fatal("the first copy was not kept")
	}
	if err := r.do(func() error {
		if !r.kvEvictIdle(0) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	st, _ := m.KVSnapshotStats()
	if !r.kvReleased(3) || r.kvDemoted[3] == nil || r.kvDemoted[1] != nil || st.Used != one {
		t.Errorf("after the second release: slot 3 released %v, its copy %v; slot 1's copy %v; %d host bytes charged, one copy is %d",
			r.kvReleased(3), r.kvDemoted[3] != nil, r.kvDemoted[1] != nil, st.Used, one)
	}
	if got := r.ReusableKVSlot(1, conv); got != 0 {
		t.Errorf("slot 1, its copy gone, answers %d", got)
	}
}

// TestKV7_snapshotTakesRoomFromACopy: a prefix snapshot with no room drops a demoted copy before it gives up: the copy serves one conversation
// that may never return, and the snapshot any prompt with that prefix.
func TestKV7_snapshotTakesRoomFromACopy(t *testing.T) {
	gen := func(m *decoder.Model, ids []int) {
		out, g := m.Generate(context.Background(), ids, 1, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
		for range out {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
	}
	_, r0 := kv7Load(t, 3, 1<<30)
	share := r0.kvShareBlocks * kvBlock
	rows := share - 1024
	one := r0.KVSnapshotBytes(rows)
	m, r := kv7Load(t, 3, one+one/2)
	_, _, _, _, _, _, vocab := m.Dims()
	gen(m, kv4Ids(rows, 1, vocab))         // a snapshot of rows rows
	gen(m, kv4Ids(2*share+2048, 2, vocab)) // past the budget, so no snapshot; releases slot 0, whose copy displaces that snapshot
	if r.kvDemoted[0] == nil {
		t.Fatal("slot 0 not demoted")
	}
	st, _ := m.KVSnapshotStats()
	before := st.Declined["budget"]
	gen(m, kv4Ids(rows, 3, vocab)) // a new prompt, whose snapshot has no room beside the copy
	st, _ = m.KVSnapshotStats()
	t.Logf("declined %v, held %d, used %d of %d, copy %v", st.Declined, st.Held, st.Used, st.Budget, r.kvDemoted[0] != nil)
	if r.kvDemoted[0] != nil || st.Held != 1 || st.Declined["budget"] != before || st.Used > st.Budget {
		t.Errorf("after the third prompt: copy kept %v, %d snapshots held, budget declines %d (was %d), %d of %d bytes", r.kvDemoted[0] != nil, st.Held, st.Declined["budget"], before, st.Used, st.Budget)
	}
}
