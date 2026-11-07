//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/wrappedupload"
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
		blocks, err := r.PromoteKV(1, conv)
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

// kv7InFlight reserves n bytes of the host budget outside any copy, as a snapshot being taken does.
func kv7InFlight(r *cudaResident, n int64) bool {
	var ok bool
	_ = r.do(func() error { ok = r.kvHost.Reserve(n, false); return nil })
	return ok
}

// kv7PromoteAll promotes every row of slot's host copy.
func kv7PromoteAll(r *cudaResident, slot int) error {
	_, err := r.PromoteKV(slot, r.kvDemoted[slot].rows)
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
	var d *decoder.KVCopyDecline
	if !errors.As(err, &d) {
		t.Errorf("a copy that leaves dst as it was returned %v, not a decline", err)
	}
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

// TestKV7_copyFromDemoted: a request whose prefix another conversation's demoted slot holds takes those rows from the copy, and the slot goes on
// waiting in host memory for its own conversation. The rows report as shared, the blocks in the counts.
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
	wantBlocks := lead / kvBlock // the rows the copy needs, and no more
	if g.KVPromotedBlocks != wantBlocks || st.Promoted != 1 || st.PromotedBlocks != int64(wantBlocks) {
		t.Errorf("promoted: the generation %d blocks, the stats %d times, %d blocks; want %d blocks once", g.KVPromotedBlocks, st.Promoted, st.PromotedBlocks, wantBlocks)
	}
	if g.KVPromotedRows != 0 {
		t.Errorf("the generation reports %d rows from host memory; they reached it through the copy, as shared", g.KVPromotedRows)
	}
	if !r.kvReleased(0) || r.kvDemoted[0] == nil || r.kvHeld(0) != 0 {
		t.Errorf("the source: released %v, copy %v, %d blocks held; want it still waiting in host memory", r.kvReleased(0), r.kvDemoted[0] != nil, r.kvHeld(0))
	}
}

// TestKV7_copyFromDemotedIsBitExact: rows written from a demoted slot's copy continue bit-identically with the same rows held by a slot never
// released. A plant that loses the copy's rows must differ.
func TestKV7_copyFromDemotedIsBitExact(t *testing.T) {
	for _, c := range []struct {
		plant bool
		start int
	}{{false, 0}, {false, 1024}, {true, 0}, {true, 1024}} {
		plant := c.plant
		m, r := kv7Load(t, 4, 1<<30)
		conv, ids := kv7Release(t, m, r)
		if plant {
			clear(r.kvDemoted[1].host.Slice())
		}
		r.SetKVSlotBusy(2, true) // the continuations below take blocks from slot 0, the borrower, and not from this
		t.Cleanup(func() { r.SetKVSlotBusy(2, false) })
		const hit = 4096
		if err := r.UseKVSlot(3); err != nil {
			t.Fatal(err)
		}
		if blocks, held, err := r.CopyDemotedKV(1, 3, c.start, hit); err != nil || !held || blocks != (hit-c.start)/kvBlock {
			t.Fatalf("the copy: %d blocks, held %v, %v", blocks, held, err)
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

// TestKV7_copyOntoDemotedWithItsOwnPrefix: as TestKV7_copyOntoDemoted, with a demoted slot short enough that its own prefix passes the in-place
// cap, so the slot's own rows answer the request's score and the published hit on it is above zero. The copy still takes the whole run from the
// source, and every row reports as shared.
func TestKV7_copyOntoDemotedWithItsOwnPrefix(t *testing.T) {
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
	p, q := kv4Ids(2048, 1, vocab), kv4Ids(4096, 2, vocab)
	lead := append(append([]int(nil), p...), q...)
	long := append(append([]int(nil), lead...), kv4Ids(8000, 4, vocab)...)
	gen(m, long)                                                         // slot 0: holds the longer run, and gives up too much to be taken in place
	gen(m, append(append([]int(nil), p...), kv4Ids(1024, 3, vocab)...))  // slot 1: 3,072 tokens sharing 2,048
	gen(m, append(append([]int(nil), long...), kv4Ids(64, 7, vocab)...)) // slot 0 again, so slot 1 is the least recently used
	gen(m, kv4Ids(34500, 5, vocab))                                      // slot 2: releases slot 1 and no more
	if r.kvDemoted[1] == nil || r.kvReleased(0) {
		t.Fatalf("slot 1's copy %v, slot 0 released %v; want only slot 1 released", r.kvDemoted[1] != nil, r.kvReleased(0))
	}
	if got := r.ReusableKVSlot(1, len(p)); got != len(p) {
		t.Fatalf("slot 1 answers %d for its %d shared rows: the test reads nothing", got, len(p))
	}
	g := gen(m, append(append([]int(nil), lead...), kv4Ids(1024, 6, vocab)...))
	t.Logf("slot %d: reused %d, shared %d, promoted %d blocks (%d rows); declined %q", r.kvSlot, g.PrefillReused, g.PrefillShared, g.KVPromotedBlocks, g.KVPromotedRows, g.KVCopyDeclined)
	if r.kvSlot != 1 {
		t.Fatalf("the request bound slot %d: the test reads nothing", r.kvSlot)
	}
	if g.PrefillReused != len(lead) || g.PrefillShared != len(lead) || g.KVPromotedRows != 0 {
		t.Errorf("reused %d, shared %d, %d rows from host memory; want %d reused and shared, so none own", g.PrefillReused, g.PrefillShared, g.KVPromotedRows, len(lead))
	}
}

// TestKV7_noEvictionForWhatCannotFit: a copy that cannot fit beside the copy a promotion still holds drops nothing to try.
func TestKV7_noEvictionForWhatCannotFit(t *testing.T) {
	m0, r0 := kv7Load(t, 4, 1<<30)
	kv7Release(t, m0, r0)
	one := r0.kvDemoted[1].Bytes()
	m, r := kv7Load(t, 4, 2*one+one/5) // holds two copies, not three
	kv7Release(t, m, r)
	if err := r.do(func() error {
		if !r.kvEvictIdle(0) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if r.kvDemoted[1] == nil || r.kvDemoted[3] == nil {
		t.Fatalf("copies: slot 1 %v, slot 3 %v; want both", r.kvDemoted[1] != nil, r.kvDemoted[3] != nil)
	}
	var ok bool
	if err := r.do(func() error { ok = r.kvHostReserve(2*one, 2, 1); return nil }); err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("twice a copy fit beside the taker's copy under a budget of 2.2 copies")
	}
	if r.kvDemoted[3] == nil {
		t.Error("slot 3's copy was dropped for a reservation that could not fit beside the taker's")
	}
}

// TestKV7_partialPromotionAnswersWhatItHolds: a promotion of some of a copy's rows leaves the slot holding those, and answering 0 past them, as
// for a request that ended before it wrote.
func TestKV7_partialPromotionAnswersWhatItHolds(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m, r)
	const rows = 1024
	if blocks, err := r.PromoteKV(1, rows); err != nil || blocks != rows/kvBlock {
		t.Fatalf("promoting %d rows: %d blocks, %v", rows, blocks, err)
	}
	if got := r.ReusableKVSlot(1, rows); got != rows {
		t.Errorf("slot 1 answers %d for the %d rows it holds", got, rows)
	}
	if got := r.ReusableKVSlot(1, conv); got != 0 {
		t.Errorf("slot 1 answers %d for %d rows, holding %d", got, conv, rows)
	}
	if err := r.UseKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if got := r.ReusableKV(conv); got != 0 {
		t.Errorf("the bound slot answers %d for %d rows, holding %d", got, conv, rows)
	}
}

// TestKV7_promotingASourceKeepsTheBoundSlotsCopy: room for a copy made while a source is promoted does not come from the bound slot's own
// copy, which its request is about to use.
func TestKV7_promotingASourceKeepsTheBoundSlotsCopy(t *testing.T) {
	m0, r0 := kv7Load(t, 4, 1<<30)
	kv7Release(t, m0, r0)
	one := r0.kvDemoted[1].Bytes()
	m, r := kv7Load(t, 4, 2*one+one/5) // holds two copies
	kv7Release(t, m, r)
	if err := r.UseKVSlot(1); err != nil { // slot 1, demoted, is bound; slot 3 is the next release, and slot 2 the one after
		t.Fatal(err)
	}
	if err := r.do(func() error {
		if !r.kvEvictIdle(1) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if r.kvDemoted[1] == nil || r.kvDemoted[3] == nil {
		t.Fatalf("copies: slot 1 %v, slot 3 %v; want both", r.kvDemoted[1] != nil, r.kvDemoted[3] != nil)
	}
	var ok bool
	if err := r.do(func() error { ok = r.kvHostReserve(one, 2, 3); return nil }); err != nil { // slot 3 promoting as a source, slot 2's copy on its way
		t.Fatal(err)
	}
	if ok || r.kvDemoted[1] == nil {
		t.Errorf("reserved %v; the bound slot's copy kept: %v", ok, r.kvDemoted[1] != nil)
	}
}

// TestKV7_failedWriteFromZeroLeavesTheSlotReleased: a write from position 0 means the decoder has forgotten the old rows, so the copy goes first,
// and its bytes are there for the releases the write's blocks may cause. A write the pool refuses leaves the slot released and unmapped.
func TestKV7_failedWriteFromZeroLeavesTheSlotReleased(t *testing.T) {
	for _, c := range []struct {
		name  string
		write func(r *cudaResident) error
	}{
		{"prefill", func(r *cudaResident) error {
			return r.do(func() error { return r.kvEnsureSlot(1, 0, r.kvMaxBlocks()*kvBlock) }) // the whole table: more than a reclaim can free
		}},
		{"upload", func(r *cudaResident) error {
			n := r.kvMaxBlocks() * kvBlock * r.layers[0].kvDim
			if err := r.UseKVSlot(1); err != nil {
				t.Fatal(err)
			}
			return r.UploadKV(0, 0, make([]float32, n), make([]float32, n))
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, r := kv7Load(t, 4, 1<<30)
			kv7Release(t, m, r)
			r.SetKVSlotBusy(2, true)
			r.SetKVSlotBusy(3, true)
			t.Cleanup(func() { r.SetKVSlotBusy(2, false); r.SetKVSlotBusy(3, false) })
			kv7FillPool(t, r, 0, r.kvHeld(0)*kvBlock)
			held := r.kvDemoted[1].Bytes()
			st, _ := m.KVSnapshotStats()
			used := st.Used
			err := c.write(r)
			if err == nil {
				t.Fatal("the write did not fail: the test reads nothing")
			}
			t.Logf("the write failed: %v", err)
			st, _ = m.KVSnapshotStats()
			if !r.kvReleased(1) || r.kvHeld(1) != 0 || r.kvDemoted[1] != nil || st.Used != used-held {
				t.Errorf("after the failed write: released %v, %d blocks held, copy kept %v, %d host bytes charged (was %d, the copy %d)",
					r.kvReleased(1), r.kvHeld(1), r.kvDemoted[1] != nil, st.Used, used, held)
			}
			if got := r.ReusableKVSlot(1, kvBlock); got != 0 {
				t.Errorf("the slot answers %d after the failed write", got)
			}
		})
	}
}

// TestKV7_promotedSourceGoesBackToHostMemory: a slot promoted to be copied from, past its share and idle again, is released with a new copy
// when a slot within its share needs its blocks. A reclaim, which would lose its rows past the share, is only for a slot no release can take.
func TestKV7_promotedSourceGoesBackToHostMemory(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	conv := share + share/4
	if _, err := kv4Prefill(m, r, 1, kv4Ids(conv, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	for _, s := range []int{0, 2} {
		r.SetKVSlotBusy(s, true)
	}
	t.Cleanup(func() { r.SetKVSlotBusy(0, false); r.SetKVSlotBusy(2, false) })
	if err := r.UseKVSlot(3); err != nil {
		t.Fatal(err)
	}
	if err := r.do(func() error {
		if !r.kvEvictIdle(3) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if blocks, err := r.PromoteKV(1, conv); err != nil || blocks != conv/kvBlock {
		t.Fatalf("promoting the source: %d blocks, %v", blocks, err)
	}
	kv7FillPool(t, r, 0, 0)
	if err := r.do(func() error { return r.kvEnsureSlot(3, 0, kvBlock) }); err != nil {
		t.Fatal(err)
	}
	if r.kvStopped[1] || !r.kvReleased(1) || r.kvDemoted[1] == nil || r.kvDemoted[1].rows != conv {
		t.Errorf("the source: stopped %v, released %v, copy %v", r.kvStopped[1], r.kvReleased(1), r.kvDemoted[1] != nil)
	}
}

// TestKV7_copySourceNotReclaimedUnderAFullPool: a promoted source furthest past its share, with the pool full and a block the copy must allocate.
// No release can take the source, which is busy for the copy; nor can a reclaim, which would unmap rows the copy has yet to read and stop the
// source. The copy declines or takes its block from another borrower, and the source keeps its rows.
func TestKV7_copySourceNotReclaimedUnderAFullPool(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	conv := share + share/2
	if _, err := kv4Prefill(m, r, 1, kv4Ids(conv, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	for _, s := range []int{0, 2} {
		r.SetKVSlotBusy(s, true)
	}
	t.Cleanup(func() { r.SetKVSlotBusy(0, false); r.SetKVSlotBusy(2, false) })
	if err := r.UseKVSlot(3); err != nil {
		t.Fatal(err)
	}
	if err := r.do(func() error {
		if !r.kvEvictIdle(3) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := kv7PromoteAll(r, 1); err != nil {
		t.Fatal(err)
	}
	kv7Hole(t, r, 1, 5) // below the hit and the share: dst must allocate for it
	held := r.kvHeld(1)
	if err := r.do(func() error { return r.kvMap(0, kvGroupFull, 0, share+share/10) }); err != nil { // past its share by less than the source
		t.Fatal(err)
	}
	kv7FillPool(t, r, 2, 0) // the rest: past its share by less than the source too
	if ex2, ex1 := r.kvHeld(2)-r.kvShareBlocks, held-r.kvShareBlocks; ex2 >= ex1 {
		t.Fatalf("slot 2 is past its share by %d blocks, the source by %d: a reclaim would not reach the source", ex2, ex1)
	}
	end := (r.kvShareBlocks + 8) * kvBlock // past the source's share: a reclaim would unmap rows the copy has yet to read
	err := r.CopyKV(1, 3, 0, end)
	t.Logf("the copy: %v", err)
	if r.kvStopped[1] || r.kvHeld(1) != held {
		t.Errorf("the source: stopped %v, %d blocks held of %d", r.kvStopped[1], r.kvHeld(1), held)
	}
}

// TestKV7_answersOnlyForValidRows: a slot written again from position 0 with a shorter sequence, on rows that do not end a block. It answers for
// what it wrote and no more, whether its old blocks are still mapped or it now waits in host memory.
func TestKV7_answersOnlyForValidRows(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	const short = 1000
	conv, _ := kv7Release(t, m, r)
	_, _, _, _, _, _, vocab := m.Dims()
	if _, err := kv4Prefill(m, r, 2, kv4Ids(short, 1, vocab), 0); err != nil {
		t.Fatal(err)
	}
	if r.kvHeld(2)*kvBlock <= short || r.kvReleased(2) {
		t.Fatalf("slot 2 holds %d blocks, released %v: the test reads nothing", r.kvHeld(2), r.kvReleased(2))
	}
	if got := r.ReusableKVSlot(2, conv); got != 0 {
		t.Errorf("a live slot answers %d for %d rows, having written %d", got, conv, short)
	}
	if got := r.ReusableKVSlot(2, short); got != short {
		t.Errorf("a live slot answers %d for the %d rows it wrote", got, short)
	}
	if got := r.ReusableKVSlot(2, 1024); got != 0 {
		t.Errorf("a live slot answers %d for 1024 rows, having written %d in that block", got, short)
	}
	m2, r2 := kv7Load(t, 4, 1<<30)
	kv7ReleaseShort(t, m2, r2, short)
	if r2.kvDemoted[1] == nil {
		t.Fatal("no copy kept")
	}
	if got := r2.ReusableKVSlot(1, short); got != short {
		t.Errorf("a demoted slot answers %d for the %d rows it wrote", got, short)
	}
	if got := r2.ReusableKVSlot(1, 1024); got != 0 {
		t.Errorf("a demoted slot answers %d for 1024 rows, having written %d in that block", got, short)
	}
}

// TestKV7_noEvictionBesideAnInFlightReservation: bytes another caller has reserved, a snapshot being taken, are not droppable, so a copy that
// cannot fit beside them drops no other copy.
func TestKV7_noEvictionBesideAnInFlightReservation(t *testing.T) {
	m0, r0 := kv7Load(t, 4, 1<<30)
	kv7Release(t, m0, r0)
	one := r0.kvDemoted[1].Bytes()
	m, r := kv7Load(t, 4, 2*one+one/5) // holds two copies
	kv7Release(t, m, r)
	st, _ := m.KVSnapshotStats()
	inFlight := st.Budget - st.Used - one/10
	if !kv7InFlight(r, inFlight) {
		t.Fatal("the in-flight reservation")
	}
	var ok bool
	if err := r.do(func() error { ok = r.kvHostReserve(st.Budget-inFlight+one/20, 2, 3); return nil }); err != nil { // past the budget with slot 1's copy gone
		t.Fatal(err)
	}
	if ok {
		t.Fatal("the reservation fit: the test reads nothing")
	}
	if r.kvDemoted[1] == nil {
		t.Error("slot 1's copy was dropped for a reservation that could not fit beside the bytes in flight")
	}
}

// TestKV7_demoteIsOffUnderKVShareOff: with sharing off, as for an unauthenticated server, no request's timing may depend on what another left,
// so a named budget keeps no released conversation's rows, as it keeps no snapshots.
func TestKV7_demoteIsOffUnderKVShareOff(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(kv4LongFixture(t, kv4CapWindow), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentKVSlots: 4, KVHostBytes: 1 << 30, KVDemote: true, KVShareOff: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	conv, _ := kv7Release(t, m, r)
	if got := r.ReusableKVSlot(1, conv); got != 0 || r.kvDemoted[1] != nil {
		t.Errorf("slot 1 answers %d, copy %v; want 0 and none", got, r.kvDemoted[1] != nil)
	}
}

// TestKV7_validRowsKeepOrdinaryReuse: answering only for written rows costs a plain continuation nothing, on dense, recurrent and MoE families:
// a conversation's next turn still reuses its own rows. The prompt is past the shortest lead a non-exact prefill reuses.
func TestKV7_validRowsKeepOrdinaryReuse(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	for _, f := range []string{"llama-tiny", "qwen3moe-tiny", "qwen35-tiny", "nemotron-tiny", "olmo_hybrid-tiny", "gemma4-moe-kv-tiny32", "glm-tiny", "deepseek-tiny"} {
		t.Run(f, func(t *testing.T) {
			m, err := decoder.Load(windowVariantFixture(t, "../../testdata/"+f, 32768), decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 8192, ResidentKVSlots: 2})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			if _, ok := m.ResidentForwardForTest().(*cudaResident); !ok {
				t.Fatalf("not resident: %s", m.ResidentDecline())
			}
			_, _, _, _, _, _, vocab := m.Dims()
			prompt := kv4Ids(3072, 1, vocab)
			var turn []int
			out, g := m.Generate(context.Background(), prompt, 4, decoder.SamplingParams{})
			for tok := range out {
				turn = append(turn, tok)
			}
			if err := g.Err(); err != nil {
				t.Fatal(err)
			}
			next := append(append(append([]int(nil), prompt...), turn...), kv4Ids(512, 2, vocab)...)
			out, g = m.Generate(context.Background(), next, 1, decoder.SamplingParams{})
			for range out {
			}
			if err := g.Err(); err != nil {
				t.Fatal(err)
			}
			t.Logf("reused %d of %d, the record held %d, matched %d", g.PrefillReused, len(next), g.PrefillSlot, g.PrefillMatch)
			if want := len(prompt) + len(turn); g.PrefillReused != want {
				t.Errorf("a continuation reused %d rows of the %d it wrote", g.PrefillReused, want)
			}
		})
	}
}

// TestKV7_copyPoolFullFallsBackToTheSlotsOwnRows, through the decoder's own acquire: a request bound to a demoted slot wants a run longer than the
// slot's share from another demoted slot. The pool cannot hold it, so the copy declines, and the request falls back on the slot's own copy. Such a
// prompt is past the slot's share, so under this pool its prefill stops as at the context cap, whatever the copy did: what is read here is the acquire.
func TestKV7_copyPoolFullFallsBackToTheSlotsOwnRows(t *testing.T) {
	gen := func(m *decoder.Model, ids []int) *decoder.Generation {
		out, g := m.Generate(context.Background(), ids, 1, decoder.SamplingParams{LogitBias: map[int]float32{2: -1e9}})
		for range out {
		}
		if err := g.Err(); err != nil {
			t.Fatalf("generation: %v", err)
		}
		return g
	}
	m, r := kv7Load(t, 4, 1<<30)
	_, _, _, _, _, _, vocab := m.Dims()
	share := r.kvShareBlocks * kvBlock
	p := kv4Ids(2048, 1, vocab)
	lead := append(append([]int(nil), p...), kv4Ids(share+share/8-len(p), 2, vocab)...) // past dst's share
	long := append(append([]int(nil), lead...), kv4Ids(share+share/4, 4, vocab)...)     // gives up too much to be taken in place
	gen(m, long)                                                                        // slot 0, the source
	gen(m, append(append([]int(nil), p...), kv4Ids(1024, 3, vocab)...))                 // slot 1, 3,072 tokens sharing p
	gen(m, append(append([]int(nil), long...), kv4Ids(64, 7, vocab)...))                // slot 0 again, so slot 1 is the least recently used
	gen(m, kv4Ids(256, 8, vocab))                                                       // slot 2
	gen(m, kv4Ids(256, 9, vocab))                                                       // slot 3, and bound: the running borrower below
	r.SetKVSlotBusy(2, true)
	r.SetKVSlotBusy(3, true)
	t.Cleanup(func() { r.SetKVSlotBusy(2, false); r.SetKVSlotBusy(3, false) })
	for range 2 {
		if err := r.do(func() error {
			if !r.kvEvictIdle(3) {
				return fmt.Errorf("nothing to release")
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if r.kvDemoted[0] == nil || r.kvDemoted[1] == nil {
		t.Fatalf("copies: slot 0 %v, slot 1 %v; want both", r.kvDemoted[0] != nil, r.kvDemoted[1] != nil)
	}
	// Slots 2 and 3 hold the pool, each past its share, so a reclaim gives the slot one share and no more.
	if err := r.do(func() error { return r.kvMap(2, kvGroupFull, 0, 2*share) }); err != nil {
		t.Fatal(err)
	}
	kv7FillPool(t, r, 3, r.kvHeld(3)*kvBlock)
	next := append(append([]int(nil), lead...), kv4Ids(1024, 6, vocab)...)
	gen(m, next)
	st, _ := m.KVSnapshotStats()
	t.Logf("slot %d: promoted %d times, %d blocks; declined %v; the slot's copy kept: %v", r.kvSlot, st.Promoted, st.PromotedBlocks, st.Declined, r.kvDemoted[1] != nil)
	if r.kvSlot != 1 {
		t.Fatalf("the request bound slot %d: the test reads nothing", r.kvSlot)
	}
	if want := int64(len(p) / kvBlock); st.Promoted != 1 || st.PromotedBlocks != want {
		t.Errorf("promoted %d times, %d blocks; want the slot's own %d blocks, from its copy, once", st.Promoted, st.PromotedBlocks, want)
	}
}

// TestKV7_copyFromDemotedKeepsWhatDstHolds, through the decoder's own acquire: dst holds the first rows of the run itself. Only the rest comes from
// the source's copy, and only those blocks count.
func TestKV7_copyFromDemotedKeepsWhatDstHolds(t *testing.T) {
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
	p, q := kv4Ids(2048, 1, vocab), kv4Ids(4096, 2, vocab)
	lead := append(append([]int(nil), p...), q...)
	long := append(append([]int(nil), lead...), kv4Ids(8000, 4, vocab)...)
	gen(m, long)                                                         // slot 0, the source
	gen(m, append(append([]int(nil), p...), kv4Ids(1024, 3, vocab)...))  // slot 1, 3,072 tokens sharing p
	gen(m, append(append([]int(nil), long...), kv4Ids(64, 7, vocab)...)) // slot 0 again, so slot 1 is the least recently used
	gen(m, kv4Ids(256, 8, vocab))                                        // slot 2, and bound
	r.SetKVSlotBusy(1, true)
	if err := r.do(func() error {
		if !r.kvEvictIdle(2) {
			return fmt.Errorf("nothing to release")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	r.SetKVSlotBusy(1, false)
	if r.kvDemoted[0] == nil || r.kvReleased(1) {
		t.Fatalf("slot 0's copy %v, slot 1 released %v; want the source released and dst live", r.kvDemoted[0] != nil, r.kvReleased(1))
	}
	g := gen(m, append(append([]int(nil), lead...), kv4Ids(1024, 6, vocab)...))
	t.Logf("slot %d: reused %d, shared %d, promoted %d blocks; declined %q", r.kvSlot, g.PrefillReused, g.PrefillShared, g.KVPromotedBlocks, g.KVCopyDeclined)
	if r.kvSlot != 1 {
		t.Fatalf("the request bound slot %d: the test reads nothing", r.kvSlot)
	}
	if want := len(q) / kvBlock; g.KVPromotedBlocks != want || g.PrefillReused != len(lead) {
		t.Errorf("reused %d, %d blocks written from the copy; want %d reused and %d blocks", g.PrefillReused, g.KVPromotedBlocks, len(lead), want)
	}
}

// kv7StopSlot leaves slot 2 stopped: past its share, its blocks past the share taken back for slot 0 within its own. Slot 1 waits in host memory
// with conv rows. It returns conv.
func kv7StopSlot(t *testing.T, m *decoder.Model, r *cudaResident) int {
	t.Helper()
	conv, _ := kv7Release(t, m, r)
	if err := r.do(func() error { return r.kvMap(2, kvGroupFull, 0, r.kvShareBlocks*kvBlock+kvBlock) }); err != nil {
		t.Fatal(err)
	}
	r.kvValid[2] = r.kvShareBlocks*kvBlock + kvBlock
	r.SetKVSlotBusy(0, false)
	var ok bool
	if err := r.do(func() error {
		r.kvStopped[0] = true // slot 0, the larger borrower, is not the one under test
		ok = r.kvReclaim(3)
		r.kvStopped[0] = false
		return nil
	}); err != nil || !ok || !r.kvStopped[2] {
		t.Fatalf("the reclaim: %v, %v; slot 2 stopped: %v", ok, err, r.kvStopped[2])
	}
	return conv
}

// TestKV7_stoppedSlotAnswersNothing: a stopped slot's next write must start from position 0, so it answers 0 for the rows it still holds, and a
// request prefills it cold; it does not reuse them and then fail its first write.
func TestKV7_stoppedSlotAnswersNothing(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	kv7StopSlot(t, m, r)
	if got := r.ReusableKVSlot(2, kvBlock); got != 0 {
		t.Errorf("a stopped slot answers %d for a block it holds", got)
	}
	if err := r.UseKVSlot(2); err != nil {
		t.Fatal(err)
	}
	if got := r.ReusableKV(kvBlock); got != 0 {
		t.Errorf("a stopped bound slot answers %d for a block it holds", got)
	}
}

// TestKV7_writeOntoAStoppedSlot: rows copied or restored onto a stopped slot start it again from position 0, whatever it held, so the suffix's
// write at a later position succeeds.
func TestKV7_writeOntoAStoppedSlot(t *testing.T) {
	const end = 2048
	for _, c := range []struct {
		name  string
		write func(r *cudaResident) error
	}{
		{"from a live slot", func(r *cudaResident) error { return r.CopyKV(3, 2, kvBlock, end) }},
		{"from a demoted slot", func(r *cudaResident) error {
			_, held, err := r.CopyDemotedKV(1, 2, kvBlock, end)
			if !held {
				t.Fatal("the source was not released")
			}
			return err
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, r := kv7Load(t, 4, 1<<30)
			kv7StopSlot(t, m, r)
			if err := c.write(r); err != nil {
				t.Fatalf("the write: %v", err)
			}
			if r.kvStopped[2] {
				t.Error("the slot is still stopped")
			}
			if err := r.do(func() error { return r.kvEnsureSlot(2, end, kvBlock) }); err != nil {
				t.Errorf("the suffix's write: %v", err)
			}
		})
	}
}

// TestKV7_declineLeavesALiveDstAlone: a copy declined before it touches a live dst leaves dst answering for all its own rows.
func TestKV7_declineLeavesALiveDstAlone(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(windowVariantFixture(t, "../../testdata/mistral-tiny-window", 32768), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentWindowedKV: true, ResidentContext: 8192, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r := mustResident(t, m)
	if !r.kvWin {
		t.Fatal("not windowed: the test reads nothing")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	own := r.kvWindow + 64 // past the window, or no base can refuse a copy; within the slack, so dst answers for all of it
	for _, s := range []int{1, 0} {
		if _, err := kv4Prefill(m, r, s, kv4Ids(own, 1, vocab), 0); err != nil {
			t.Fatal(err)
		}
	}
	if got := r.ReusableKVSlot(1, own); got != own {
		t.Fatalf("dst answers %d for its %d rows before the copy", got, own)
	}
	r.kvBases[0] = own // the source's window has moved past anything the copy could continue from
	var d *decoder.KVCopyDecline
	if err := r.CopyKV(0, 1, 0, own/kvBlock*kvBlock); !errors.As(err, &d) {
		t.Fatalf("the copy: %v, want a decline", err)
	}
	if got := r.ReusableKVSlot(1, own); got != own {
		t.Errorf("after a decline that touched nothing, dst answers %d for its %d rows", got, own)
	}
}

// TestKV7_uploadPastZeroOnAReleasedSlot: as a write past position 0 on a released slot, an upload there is refused and leaves the slot as it was.
func TestKV7_uploadPastZeroOnAReleasedSlot(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m, r)
	if err := r.UseKVSlot(1); err != nil {
		t.Fatal(err)
	}
	n := kvBlock * r.layers[0].kvDim
	if err := r.UploadKV(0, kvBlock, make([]float32, n), make([]float32, n)); err == nil {
		t.Error("an upload at a later position on a released slot went through")
	}
	if !r.kvReleased(1) || r.kvHeld(1) != 0 || r.kvDemoted[1] == nil || r.ReusableKVSlot(1, conv) != conv {
		t.Errorf("after the upload: released %v, %d blocks held, copy %v, answers %d", r.kvReleased(1), r.kvHeld(1), r.kvDemoted[1] != nil, r.ReusableKVSlot(1, conv))
	}
}

// kvSlotState is what a table row reads of a slot, to compare before and after.
type kvSlotState struct {
	released, stopped, copied bool
	valid, held               int
}

func kvStateOf(r *cudaResident, s int) kvSlotState {
	return kvSlotState{r.kvReleased(s), r.kvStopped[s], r.kvDemoted[s] != nil, r.kvValid[s], r.kvHeld(s)}
}

// TestKV7_slotWriteTable holds every path that writes a slot to one rule, in every state the slot can be in and every way the write can end.
//
//   - It succeeds: the slot is live, holds no host copy, answers for the rows written, and takes the next write after them.
//   - It is declined or refused: the slot is as it was.
//   - It fails part way on a live slot: the slot answers only for rows it still holds, and holds no block past them.
//   - It fails part way on a released or stopped slot: the slot stays so and holds no block. A copy, restore or promote leaves any host copy, since
//     the request can still use it; a write or upload from position 0 has dropped it, since the decoder had already forgotten those rows.
func TestKV7_slotWriteTable(t *testing.T) {
	const own, end = 1024, 2048
	const target, demotedSrc, liveSrc = 2, 1, 3
	type op struct {
		name   string
		keeps  bool // a failure part way leaves any host copy
		from   int  // the first row a live slot gives up to the write
		write  func(r *cudaResident) error
		refuse func(r *cudaResident) error // a form of the write that declines or is refused, in the state named by refused; nil when none
	}
	snapshot := func(t *testing.T, r *cudaResident) *cudaKVSnapshot {
		h, err := r.SnapshotKV(liveSrc, end)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(h.Release)
		return h.(*cudaKVSnapshot)
	}
	upload := func(r *cudaResident, base int) error {
		n := end * r.layers[0].kvDim
		if err := r.UseKVSlot(target); err != nil {
			return err
		}
		return r.UploadKV(0, base, make([]float32, n), make([]float32, n))
	}
	ops := func(t *testing.T, r *cudaResident) []op {
		snap := snapshot(t, r)
		return []op{
			{"copy", true, kvBlock,
				func(r *cudaResident) error { return r.CopyKV(liveSrc, target, kvBlock, end) },
				func(r *cudaResident) error { return r.CopyKV(demotedSrc, target, kvBlock, end) }},
			{"copy from a demoted slot", true, own,
				func(r *cudaResident) error { _, _, err := r.CopyDemotedKV(demotedSrc, target, own, end); return err },
				func(r *cudaResident) error { _, _, err := r.CopyDemotedKV(demotedSrc, target, own, 2*end); return err }},
			{"restore", true, end,
				func(r *cudaResident) error { return r.RestoreKV(snap, target, end) },
				func(r *cudaResident) error { return r.RestoreKV(snap, target, 2*end) }},
			{"upload", false, 0,
				func(r *cudaResident) error { return upload(r, 0) },
				func(r *cudaResident) error { return upload(r, kvBlock) }},
			{"write", false, 0,
				func(r *cudaResident) error { return r.do(func() error { return r.kvEnsureSlot(target, 0, end) }) },
				func(r *cudaResident) error { return r.do(func() error { return r.kvEnsureSlot(target, kvBlock, end) }) }},
		}
	}
	for _, state := range []string{"live", "released", "demoted", "stopped"} {
		for i := range 5 {
			for _, ending := range []string{"success", "declined", "part way"} {
				t.Run(fmt.Sprintf("%s/%d/%s", state, i, ending), func(t *testing.T) {
					m, r := kv7Load(t, 4, 1<<30)
					_, _, _, _, _, _, vocab := m.Dims()
					ids := kv4Ids(end, 1, vocab)
					for _, c := range []struct{ s, n int }{{demotedSrc, end}, {target, own}, {liveSrc, end}} {
						if _, err := kv4Prefill(m, r, c.s, ids[:c.n], 0); err != nil {
							t.Fatal(err)
						}
					}
					if err := r.UseKVSlot(0); err != nil {
						t.Fatal(err)
					}
					evict := func() {
						if err := r.do(func() error {
							if !r.kvEvictIdle(0) {
								return fmt.Errorf("nothing to release")
							}
							return nil
						}); err != nil {
							t.Fatal(err)
						}
					}
					evict() // the demoted source
					switch state {
					case "demoted":
						evict()
					case "released":
						evict()
						if err := r.do(func() error { r.kvDropDemoted(target); r.kvValid[target] = 0; return nil }); err != nil {
							t.Fatal(err)
						}
					case "stopped":
						share := r.kvShareBlocks * kvBlock
						if err := r.do(func() error {
							if e := r.kvMap(target, kvGroupFull, 0, share+kvBlock); e != nil {
								return e
							}
							if !r.kvReclaim(0) {
								return fmt.Errorf("nothing reclaimed")
							}
							return nil
						}); err != nil {
							t.Fatal(err)
						}
					}
					o := ops(t, r)[i]
					name := o.name
					before := kvStateOf(r, target)
					fresh := before.released || before.stopped
					switch ending {
					case "success":
						if err := o.write(r); err != nil {
							t.Errorf("%s: %v", name, err)
							return
						}
						got := kvStateOf(r, target)
						if got.released || got.stopped || got.copied || got.valid != end {
							t.Errorf("%s: %+v; want live, no copy, %d rows", name, got, end)
						}
						if a := r.ReusableKVSlot(target, end); a != end {
							t.Errorf("%s: answers %d for the %d rows written", name, a, end)
						}
						if err := r.do(func() error { return r.kvEnsureSlot(target, end, kvBlock) }); err != nil {
							t.Errorf("%s: the next write: %v", name, err)
						}
					case "declined":
						// A live slot has no refused form of a write from position 0 at a later position: that is an ordinary continuation.
						if !fresh && (o.name == "upload" || o.name == "write") {
							return
						}
						if err := o.refuse(r); err == nil {
							t.Errorf("%s: not refused", name)
							return
						}
						if got := kvStateOf(r, target); got != before {
							t.Errorf("%s: %+v; want it as it was, %+v", name, got, before)
						}
					case "part way":
						// The copy allocates only for a block its source does not map; a stopped slot, only for a block it does not hold.
						if o.name == "copy" {
							kv7Hole(t, r, liveSrc, own/kvBlock+2)
						}
						if state == "stopped" {
							kv7Hole(t, r, target, own/kvBlock+2)
							before = kvStateOf(r, target)
						}
						kvTakeFailForTest = func(s int) bool { return s == target }
						err := o.write(r)
						kvTakeFailForTest = nil
						if err == nil {
							t.Errorf("%s: did not fail", name)
							return
						}
						got := kvStateOf(r, target)
						if fresh {
							want := before
							want.held = 0
							want.copied = before.copied && o.keeps
							if !want.copied {
								want.valid = 0
							}
							if got != want {
								t.Errorf("%s: %+v; want %+v (%v)", name, got, want, err)
							}
							return
						}
						if got.released || got.stopped || got.valid > own || got.valid < min(o.from, own) {
							t.Errorf("%s: %+v; want it live, answering for between %d and %d rows (%v)", name, got, min(o.from, own), own, err)
						}
						if got.held != kvBlocksFor(got.valid) {
							t.Errorf("%s: %d blocks held for %d rows (%v)", name, got.held, got.valid, err)
						}
					}
				})
			}
		}
	}
}

// TestKV7_promotionTable is TestKV7_slotWriteTable's rule for a promotion, which only a demoted slot has anything to do on.
func TestKV7_promotionTable(t *testing.T) {
	const rows = 1000 // not the end of a block: the slot answers for these and not the rest of their block
	m, r := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m, r)
	kvTakeFailForTest = func(s int) bool { return s == 1 }
	err := func() error { defer func() { kvTakeFailForTest = nil }(); return kv7PromoteAll(r, 1) }()
	if err == nil {
		t.Fatal("the promotion did not fail")
	}
	if got := kvStateOf(r, 1); !got.released || !got.copied || got.held != 0 || got.valid != conv {
		t.Errorf("part way: %+v; want it released, with its copy and its %d rows, no blocks (%v)", got, conv, err)
	}
	if a := r.ReusableKVSlot(1, conv); a != conv {
		t.Errorf("part way: answers %d", a)
	}
	blocks, err := r.PromoteKV(1, rows)
	if want := (rows + kvBlock - 1) / kvBlock; err != nil || blocks != want {
		t.Fatalf("promoting %d rows: %d blocks, %v; want %d", rows, blocks, err, want)
	}
	got := kvStateOf(r, 1)
	if got.released || got.copied || got.stopped || got.valid != rows {
		t.Errorf("after the promotion: %+v; want it live, no copy, %d rows", got, rows)
	}
	if a := r.ReusableKVSlot(1, rows+1); a != 0 {
		t.Errorf("answers %d for a row past the %d promoted", a, rows)
	}
	for _, s := range []int{2} { // a live slot has nothing to promote
		before := kvStateOf(r, s)
		if blocks, err := r.PromoteKV(s, rows); err != nil || blocks != 0 || kvStateOf(r, s) != before {
			t.Errorf("a live slot: %d blocks, %v, %+v", blocks, err, kvStateOf(r, s))
		}
	}
}

// TestKV7_rangeChecks: a copy from a demoted slot refuses a range that is not whole blocks, and neither copy reads rows its source no longer answers for.
func TestKV7_rangeChecks(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	conv, _ := kv7Release(t, m, r)
	for _, c := range []struct{ start, end int }{{-kvBlock, kvBlock}, {kvBlock, 0}, {0, kvBlock + 1}, {1, kvBlock}, {0, 0}} {
		if _, _, err := r.CopyDemotedKV(1, 3, c.start, c.end); err == nil {
			t.Errorf("rows [%d, %d): not refused", c.start, c.end)
		}
	}
	// dst holds fewer rows than start says: a hole would be left below it, so the copy begins at what dst holds.
	if err := r.do(func() error { r.kvValid[3] = kvBlock; return nil }); err != nil {
		t.Fatal(err)
	}
	if blocks, _, err := r.CopyDemotedKV(1, 3, 4*kvBlock, conv); err != nil || blocks != (conv-kvBlock)/kvBlock {
		t.Errorf("a start past what dst holds: %d blocks, %v; want %d", blocks, err, (conv-kvBlock)/kvBlock)
	}
	var d *decoder.KVCopyDecline
	if err := r.do(func() error { r.kvValid[2] = kvBlock; return nil }); err != nil {
		t.Fatal(err)
	}
	if err := r.CopyKV(2, 3, 0, 2*kvBlock); !errors.As(err, &d) {
		t.Errorf("a copy past what the source wrote: %v, want a decline", err)
	}
}

// TestKV7_uploadPass: the layers of one upload are one write. A failure at any layer, on the device copy or the mapping, leaves a released or stopped slot
// as the write-from-zero rule says, and a live slot answering for nothing; the slot is live only once the last layer's rows have landed. A sliding
// layer's upload at a later position is part of the pass, not a refused write.
func TestKV7_uploadPass(t *testing.T) {
	const rows = 1024
	plant := errors.New("plant")
	layersOf := func(r *cudaResident) []int {
		var ls []int
		for l := range r.layers {
			if !r.layers[l].kvShared && !r.layers[l].noKV() && r.layers[l].kvDim > 0 {
				ls = append(ls, l)
			}
		}
		return ls
	}
	upload := func(r *cudaResident, l, base int) error {
		n := rows * r.layers[l].kvDim
		return r.UploadKV(l, base, make([]float32, n), make([]float32, n))
	}
	pass := func(r *cudaResident, base int) (first error) {
		if err := r.BeginKVUpload(); err != nil {
			return err
		}
		for i, l := range layersOf(r) {
			b := 0
			if i == 0 {
				b = base
			}
			if first = upload(r, l, b); first != nil {
				break
			}
		}
		if err := r.EndKVUpload(first != nil); first == nil {
			first = err
		}
		return first
	}
	setup := func(t *testing.T, released bool) (*cudaResident, int) {
		m, r := kv7Load(t, 4, 1<<30)
		_, _, _, _, _, _, vocab := m.Dims()
		for _, s := range []int{1, 2} {
			if _, err := kv4Prefill(m, r, s, kv4Ids(rows, 1, vocab), 0); err != nil {
				t.Fatal(err)
			}
		}
		if released {
			if err := r.UseKVSlot(0); err != nil {
				t.Fatal(err)
			}
			if err := r.do(func() error {
				if !r.kvEvictIdle(0) {
					return fmt.Errorf("nothing to release")
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		return r, 1
	}
	t.Run("a released slot, the second layer failing", func(t *testing.T) {
		r, s := setup(t, true)
		ls := layersOf(r)
		if len(ls) < 2 {
			t.Fatalf("%d layers with KV: the test reads nothing", len(ls))
		}
		kvUploadFailForTest = func(l int) error {
			if l == ls[1] {
				return plant
			}
			return nil
		}
		defer func() { kvUploadFailForTest = nil }()
		if err := pass(r, 0); !errors.Is(err, plant) {
			t.Fatalf("the pass: %v", err)
		}
		if got := kvStateOf(r, s); !got.released || got.stopped || got.copied || got.valid != 0 || got.held != 0 {
			t.Errorf("%+v; want it released, no copy, no rows, no blocks", got)
		}
		if a := r.ReusableKVSlot(s, kvBlock); a != 0 {
			t.Errorf("answers %d", a)
		}
	})
	t.Run("a live slot, the device copy failing", func(t *testing.T) {
		r, s := setup(t, false)
		kvUploadFailForTest = func(int) error { return plant }
		defer func() { kvUploadFailForTest = nil }()
		if err := upload(r, layersOf(r)[0], 0); !errors.Is(err, plant) {
			t.Fatalf("the upload: %v", err)
		}
		if got := kvStateOf(r, s); got.released || got.valid != 0 || got.held != 0 {
			t.Errorf("%+v; want it live, answering for nothing, no blocks", got)
		}
	})
	t.Run("a released slot, one upload, the device copy failing", func(t *testing.T) {
		r, s := setup(t, true)
		kvUploadFailForTest = func(int) error { return plant }
		defer func() { kvUploadFailForTest = nil }()
		if err := upload(r, layersOf(r)[0], 0); !errors.Is(err, plant) {
			t.Fatalf("the upload: %v", err)
		}
		if got := kvStateOf(r, s); !got.released || got.valid != 0 || got.held != 0 {
			t.Errorf("%+v; want it released, no rows, no blocks", got)
		}
	})
	t.Run("a released slot, every layer landing", func(t *testing.T) {
		r, s := setup(t, true)
		if err := pass(r, 0); err != nil {
			t.Fatal(err)
		}
		if got := kvStateOf(r, s); got.released || got.copied || got.valid != rows {
			t.Errorf("%+v; want it live, no copy, %d rows", got, rows)
		}
		if a := r.ReusableKVSlot(s, rows); a != rows {
			t.Errorf("answers %d", a)
		}
	})
}

// TestKV7_uploadPassLayers, on a Gemma 2 shape, sliding and full-attention layers in turn: a pass holds rows only if every layer reached the same end and every
// full-attention layer ran from position 0, whatever order they came in. A sliding layer may start later: the CPU's ring has wrapped, whether or not the
// resident windows it. Anything else fails the pass, and the slot is left as a failed write from position 0 leaves it.
func TestKV7_uploadPassLayers(t *testing.T) {
	const later, end = kvBlock, kvBlock + 256
	setup := func(t *testing.T, windowed, release bool) (*cudaResident, []int, []int) {
		requireCUDADevice(t)
		m, err := decoder.Load(windowVariantFixture(t, "../../testdata/gemma2-tiny", 32768), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentWindowedKV: windowed, ResidentContext: 8192, ResidentKVSlots: 2})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.Close() })
		r := mustResident(t, m)
		var sliding, full []int
		for l := range r.layers {
			switch Ly := &r.layers[l]; {
			case Ly.kvShared || Ly.noKV() || Ly.kvDim == 0:
			case Ly.window > 0:
				sliding = append(sliding, l)
			default:
				full = append(full, l)
			}
		}
		if len(sliding) == 0 || len(full) == 0 || sliding[0] > full[0] {
			t.Fatalf("sliding layers %v, full layers %v: not the Gemma 2 order this reads", sliding, full)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		if _, err := kv4Prefill(m, r, 1, kv4Ids(256, 1, vocab), 0); err != nil {
			t.Fatal(err)
		}
		if err := r.UseKVSlot(0); err != nil {
			t.Fatal(err)
		}
		if release {
			if err := r.do(func() error {
				if !r.kvEvictIdle(0) {
					return fmt.Errorf("nothing to release")
				}
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		return r, sliding, full
	}
	up := func(r *cudaResident, l, base, hi int) error {
		n := (hi - base) * r.layers[l].kvDim
		return r.UploadKV(l, base, make([]float32, n), make([]float32, n))
	}
	type span struct{ l, base, hi int }
	pass := func(r *cudaResident, spans []span) error {
		if err := r.BeginKVUpload(); err != nil {
			return err
		}
		var err error
		for _, c := range spans {
			if err = up(r, c.l, c.base, c.hi); err != nil {
				break
			}
		}
		if e := r.EndKVUpload(err != nil); err == nil {
			err = e
		}
		return err
	}
	bad := []struct {
		name  string
		spans func(sliding, full []int) []span
	}{
		{"a full-attention layer short of the end", func(sl, fu []int) []span {
			out := []span{}
			for _, l := range sl {
				out = append(out, span{l, later, end})
			}
			for _, l := range fu {
				out = append(out, span{l, 0, end - kvBlock})
			}
			return out
		}},
		{"a full-attention layer missing", func(sl, fu []int) []span {
			out := []span{}
			for _, l := range sl {
				out = append(out, span{l, later, end})
			}
			for _, l := range fu[1:] {
				out = append(out, span{l, 0, end})
			}
			return out
		}},
		{"no full-attention layer at all", func(sl, fu []int) []span {
			out := []span{}
			for _, l := range sl {
				out = append(out, span{l, later, end})
			}
			return out
		}},
		{"a sliding layer first, then a full-attention layer past position 0", func(sl, fu []int) []span {
			return []span{{sl[0], later, end}, {fu[0], later, end}}
		}},
		{"a sliding layer whose span does not cover its window", func(sl, fu []int) []span {
			out := []span{{sl[0], end - 1, end}}
			for _, l := range sl[1:] {
				out = append(out, span{l, later, end})
			}
			for _, l := range fu {
				out = append(out, span{l, 0, end})
			}
			return out
		}},
		{"a sliding layer uploaded twice", func(sl, fu []int) []span {
			out := []span{{sl[0], 0, kvBlock}}
			for _, l := range sl {
				out = append(out, span{l, later, end})
			}
			for _, l := range fu {
				out = append(out, span{l, 0, end})
			}
			return out
		}},
		{"nothing uploaded", func(sl, fu []int) []span { return nil }},
		{"a sliding layer short of the end", func(sl, fu []int) []span {
			out := []span{}
			for _, l := range sl {
				out = append(out, span{l, later, end - kvBlock})
			}
			for _, l := range fu {
				out = append(out, span{l, 0, end})
			}
			return out
		}},
	}
	for _, windowed := range []bool{true, false} {
		for _, c := range bad {
			t.Run(fmt.Sprintf("windowed=%v/%s", windowed, c.name), func(t *testing.T) {
				r, sl, fu := setup(t, windowed, true)
				if err := pass(r, c.spans(sl, fu)); err == nil {
					t.Errorf("the pass went through: %+v", kvStateOf(r, 1))
				}
				if got := kvStateOf(r, 1); !got.released || got.held != 0 || got.valid != 0 {
					t.Errorf("%+v; want it still released, no blocks, no rows", got)
				}
				if a := r.ReusableKVSlot(1, kvBlock); a != 0 {
					t.Errorf("answers %d", a)
				}
			})
		}
		good := func(sl, fu []int) []span {
			out := []span{}
			for i := range max(len(sl), len(fu)) { // in the fixture's order: a sliding layer, then a full-attention one
				if i < len(sl) {
					out = append(out, span{sl[i], later, end})
				}
				if i < len(fu) {
					out = append(out, span{fu[i], 0, end})
				}
			}
			return out
		}
		t.Run(fmt.Sprintf("windowed=%v/every layer, to the same end", windowed), func(t *testing.T) {
			r, sl, fu := setup(t, windowed, true)
			if err := pass(r, good(sl, fu)); err != nil {
				t.Fatalf("the pass: %v", err)
			}
			got := kvStateOf(r, 1)
			if got.released || got.stopped || got.valid != end || got.held < kvBlocksFor(end) {
				t.Errorf("%+v; want it live, %d rows, at least %d blocks", got, end, kvBlocksFor(end))
			}
			if a := r.ReusableKVSlot(1, end); a != end {
				t.Errorf("answers %d for its %d rows", a, end)
			}
		})
		t.Run(fmt.Sprintf("windowed=%v/a live slot, a pass that fails", windowed), func(t *testing.T) {
			r, sl, fu := setup(t, windowed, false)
			if err := pass(r, []span{{sl[0], later, end}, {fu[0], later, end}}); err == nil {
				t.Errorf("the pass went through: %+v", kvStateOf(r, 1))
			}
			if got := kvStateOf(r, 1); got.released || got.valid != 0 || got.held != 0 {
				t.Errorf("%+v; want it live, answering for nothing, no blocks", got)
			}
		})
	}
	// A sliding layer's rows below its base were never uploaded, so the slot answers only for prefixes whose window starts at or past that base: at a shorter
	// prefix the layer would read rows nothing wrote. The floor follows the rows through a copy, and such rows make no snapshot and no host copy.
	t.Run("a wrapped upload answers only for prefixes it covers", func(t *testing.T) {
		requireCUDADevice(t)
		m, err := decoder.Load(windowVariantFixture(t, "../../testdata/gemma2-tiny", 32768), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentContext: 8192, ResidentKVSlots: 3, KVHostBytes: 1 << 30, KVDemote: true})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.Close() })
		r := mustResident(t, m)
		var spans []span
		floor := 0
		for l := range r.layers {
			switch Ly := &r.layers[l]; {
			case Ly.kvShared || Ly.noKV() || Ly.kvDim == 0:
			case Ly.window > 0:
				spans = append(spans, span{l, later, end})
				floor = max(floor, later+int(Ly.window)-1)
			default:
				spans = append(spans, span{l, 0, end})
			}
		}
		if err := r.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		if err := pass(r, spans); err != nil {
			t.Fatalf("the pass: %v", err)
		}
		holds := func(what string, s int) {
			t.Helper()
			for _, c := range []struct{ prefix, want int }{{floor - 1, 0}, {floor, floor}, {end, end}} {
				if a := r.ReusableKVSlot(s, c.prefix); a != c.want {
					t.Errorf("%s: slot %d answers %d for %d rows, want %d (the window's rows start at %d)", what, s, a, c.prefix, c.want, later)
				}
			}
		}
		holds("after the pass", 1)
		if _, err := r.SnapshotKV(1, end/kvBlock*kvBlock); err == nil {
			t.Errorf("a snapshot of rows that do not hold every prefix was taken")
		}
		if err := r.UseKVSlot(0); err != nil {
			t.Fatal(err)
		}
		const copied = end
		if err := r.CopyKV(1, 2, 0, copied); err != nil {
			t.Fatalf("the copy: %v", err)
		}
		holds("the copy", 2)
		if err := r.do(func() error {
			if !r.kvEvictIdle(0) {
				return fmt.Errorf("nothing to release")
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if r.kvDemoted[1] != nil { // TestKV7_demoteRefusesAWrappedUpload holds this on its own
			t.Fatalf("slot 1 was demoted: %+v", kvStateOf(r, 1))
		}
		if a := r.ReusableKVSlot(1, end); a != 0 {
			t.Errorf("released with no copy, slot 1 answers %d for %d rows", a, end)
		}
		if err := r.CopyKV(2, 0, 0, end); err != nil {
			t.Fatalf("the copy from the copy: %v", err)
		}
		holds("the copy from the copy", 0)
		if err := r.do(func() error { return r.kvEnsureSlot(0, 0, end) }); err != nil {
			t.Fatal(err)
		}
		if a := r.ReusableKVSlot(0, floor-1); a != floor-1 {
			t.Errorf("a new sequence from position 0 answers %d for %d rows", a, floor-1)
		}
	})
	// A windowed build's cap is its share, so nothing there is reclaimed: a stopped slot is read on a fixture with a pool to borrow from.
	t.Run("a full-attention layer on a stopped slot", func(t *testing.T) {
		m, r := kv7Load(t, 4, 1<<30)
		kv7StopSlot(t, m, r)
		if err := r.UseKVSlot(2); err != nil {
			t.Fatal(err)
		}
		if err := r.BeginKVUpload(); err != nil {
			t.Fatal(err)
		}
		err := up(r, 0, kvBlock, end)
		if e := r.EndKVUpload(err != nil); err == nil {
			err = e
		}
		if err == nil {
			t.Error("an upload that leaves the blocks below it unmapped went through")
		}
		if got := kvStateOf(r, 2); !got.stopped || got.held != 0 || got.valid != 0 {
			t.Errorf("%+v; want it still stopped, no blocks, no rows", got)
		}
	})
}

// TestKV7_partialCopyFromAWrappedUpload is item 4 of docs/measurements/kv7-demote-2026-11/README.md, "Before demotion can be turned on", with demotion off: a partial
// copy starting below the source's base carries rows nothing uploaded, so the destination takes the source's floor, answers only where a window has
// rows, and continues there bit-identically with the source.
func TestKV7_partialCopyFromAWrappedUpload(t *testing.T) {
	requireCUDADevice(t)
	const base, end = 2 * kvBlock, 6 * kvBlock
	m, err := decoder.Load(windowVariantFixture(t, "../../testdata/gemma2-tiny", 32768), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentContext: 8192, ResidentKVSlots: 2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	_, _, _, _, _, _, vocab := m.Dims()
	ids := kv4Ids(end+64, 1, vocab)
	if err := r.UseKVSlot(0); err != nil {
		t.Fatal(err)
	}
	// The same rows wherever a slot holds them, as two slots that computed the same tokens hold them.
	rng := rand.New(rand.NewSource(1))
	rows := map[int][2][]float32{}
	layers := []int{}
	floor := 0
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.kvShared || Ly.noKV() || Ly.kvDim == 0 {
			continue
		}
		k, v := make([]float32, end*Ly.kvDim), make([]float32, end*Ly.kvDim)
		for i := range k {
			k[i], v[i] = rng.Float32(), rng.Float32()
		}
		rows[l], layers = [2][]float32{k, v}, append(layers, l)
		if Ly.window > 0 {
			floor = max(floor, base+int(Ly.window)-1)
		}
	}
	upload := func(s, hi, late int) {
		t.Helper()
		if err := r.UseKVSlot(s); err != nil {
			t.Fatal(err)
		}
		if err := r.BeginKVUpload(); err != nil {
			t.Fatal(err)
		}
		for _, l := range layers {
			from, d := 0, r.layers[l].kvDim
			if r.layers[l].window > 0 {
				from = late
			}
			if err := r.UploadKV(l, from, rows[l][0][from*d:hi*d], rows[l][1][from*d:hi*d]); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.EndKVUpload(false); err != nil {
			t.Fatal(err)
		}
	}
	upload(1, kvBlock, 0)                                // dst holds the first block itself, so the copy below is partial
	upload(0, end, base)                                 // the source: a wrapped upload, its sliding rows from base
	if err := r.CopyKV(0, 1, kvBlock, end); err != nil { // start < base
		t.Fatalf("the partial copy: %v", err)
	}
	for _, c := range []struct{ prefix, want int }{{kvBlock, 0}, {floor - 1, 0}, {floor, floor}, {end, end}} {
		if a := r.ReusableKVSlot(1, c.prefix); a != c.want {
			t.Errorf("after a partial copy, the destination answers %d for %d rows, want %d (the source's rows start at %d, its floor is %d)", a, c.prefix, c.want, base, floor)
		}
	}
	got, err := kv4Prefill(m, r, 1, ids[end:], end)
	if err != nil {
		t.Fatal(err)
	}
	want, err := kv4Prefill(m, r, 0, ids[end:], end)
	if err != nil {
		t.Fatal(err)
	}
	if j, same := sameBits(got, want); !same {
		t.Errorf("continuing at %d, the destination differs from the source at %d", end, j)
	}
}

// TestKV7_failureOnAFreshSlotLeavesNoRows: a released or stopped slot with no host copy that fails part way answers for nothing and records nothing.
func TestKV7_failureOnAFreshSlotLeavesNoRows(t *testing.T) {
	m, r := kv7Load(t, 4, 1<<30)
	kv7StopSlot(t, m, r)
	kv7Hole(t, r, 2, 8)
	kvTakeFailForTest = func(s int) bool { return s == 2 }
	defer func() { kvTakeFailForTest = nil }()
	if _, _, err := r.CopyDemotedKV(1, 2, 0, 1024); err == nil {
		t.Fatal("the copy did not fail")
	}
	if got := kvStateOf(r, 2); !got.stopped || got.held != 0 || got.valid != 0 {
		t.Errorf("%+v; want it stopped, no blocks, no rows", got)
	}
}

// A wrapped upload on the shared bridge: the same tests every resident backend runs (internal/wrappedupload).
func TestWrappedUpload_rows(t *testing.T) {
	requireCUDADevice(t)
	wrappedupload.Rows(t, "cuda")
}

func TestWrappedUpload_reuse(t *testing.T) {
	requireCUDADevice(t)
	wrappedupload.Reuse(t, "cuda")
}

func TestWrappedUpload_turns(t *testing.T) {
	requireCUDADevice(t)
	wrappedupload.Turns(t, "cuda")
}

// TestKV7_demoteRefusesAWrappedUpload is the guard release v0.24.0 asks of demotion: a slot holding a wrapped upload holds no sliding-layer rows below its base, so a host
// copy from position 0 would keep zeros (docs/measurements/kv7-demote-2026-11/README.md, item 1). Its release keeps no copy and the conversation re-prefills. The same
// slot holding rows from position 0 keeps its copy, so the guard is not what stops every demotion here.
func TestKV7_demoteRefusesAWrappedUpload(t *testing.T) {
	requireCUDADevice(t)
	const base, end = 2 * kvBlock, 6 * kvBlock
	m, err := decoder.Load(windowVariantFixture(t, "../../testdata/gemma2-tiny", 32768), decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentKVSlots: 2, KVHostBytes: 1 << 30, KVDemote: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.Close() })
	r := mustResident(t, m)
	if r.kvHost == nil || r.kvWin {
		t.Fatalf("the rig reads nothing: a host budget %v, windowed KV %v", r.kvHost != nil, r.kvWin)
	}
	upload := func(from int) {
		t.Helper()
		if err := r.UseKVSlot(1); err != nil {
			t.Fatal(err)
		}
		if err := r.BeginKVUpload(); err != nil {
			t.Fatal(err)
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			if Ly.kvShared || Ly.noKV() || Ly.kvDim == 0 {
				continue
			}
			lo := 0
			if Ly.window > 0 {
				lo = from
			}
			k := make([]float32, (end-lo)*Ly.kvDim)
			for i := range k {
				k[i] = 1
			}
			if err := r.UploadKV(l, lo, k, k); err != nil {
				t.Fatal(err)
			}
		}
		if err := r.EndKVUpload(false); err != nil {
			t.Fatal(err)
		}
		if err := r.UseKVSlot(0); err != nil {
			t.Fatal(err)
		}
	}
	release := func() (bool, int) {
		t.Helper()
		var kept bool
		var floor int
		if err := r.do(func() error {
			floor = r.kvFloor[1] // what the upload set, before the release
			if !r.kvEvictIdle(0) {
				return fmt.Errorf("nothing to release")
			}
			kept = r.kvDemoted[1] != nil
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return kept, floor
	}
	upload(0)
	if kept, floor := release(); !kept || floor != 0 {
		t.Fatalf("a slot holding rows from position 0: copy kept %v, floor %d; want a copy and no floor", kept, floor)
	}
	upload(base)
	kept, floor := release()
	t.Logf("a wrapped upload: floor %d, copy kept %v", floor, kept)
	if floor <= 0 {
		t.Fatal("the upload set no floor: the test reads nothing")
	}
	if kept {
		t.Errorf("a slot with a floor of %d was demoted: its host copy holds zeros below the sliding layers' base", floor)
	}
	if got := r.ReusableKVSlot(1, end); got != 0 {
		t.Errorf("the released slot answers %d for %d rows; with no copy, want 0", got, end)
	}
}

// TestKV4_batchedStepEvictsTheLeastRecentlyWrittenIdleSlot: D5's step 1 through the batched step. Four slots; slot 0, the least recently
// written, is the bound slot but idle; slot 3 is idle and written later; slots 1 and 2 run, and fill the pool exactly. A batched step
// then needs a block for each running row (a step writes through each row's slot and leaves the binding on slot 0), so one idle slot's
// blocks are released: the least recently written, slot 0, and not slot 3.
func TestKV4_batchedStepEvictsTheLeastRecentlyWrittenIdleSlot(t *testing.T) {
	m, r := kv4CapLoad(t, 4)
	if lo, hi := r.BatchStepRange(); hi < 2 || lo > 2 {
		t.Skipf("this fixture takes no batched step (range %d..%d)", lo, hi)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	pool := r.kvPools[kvGroupFull].blocks - 1
	share := r.kvShareBlocks * kvBlock
	conv0 := share - 1024
	conv3 := conv0
	if (pool-conv0/kvBlock-conv3/kvBlock)%2 != 0 {
		conv3 += kvBlock
	}
	run := (pool - conv0/kvBlock - conv3/kvBlock) / 2 // blocks each running slot holds, so the pool is full
	if run*kvBlock <= share {
		t.Fatalf("a running slot holds %d positions, not past its share %d: the case reads nothing", run*kvBlock, share)
	}
	for _, c := range []struct{ slot, n int }{{0, conv0}, {3, conv3}, {1, run * kvBlock}, {2, run * kvBlock}} { // writes in this order
		if _, err := kv4Prefill(m, r, c.slot, kv4Ids(c.n, c.slot, vocab), 0); err != nil {
			t.Fatalf("slot %d: %v", c.slot, err)
		}
	}
	if free := len(r.kvPools[kvGroupFull].free); free != 0 {
		t.Fatalf("%d blocks are free before the step; the case needs the pool full", free)
	}
	held := func() (h [4]int, free int) {
		_ = r.do(func() error {
			for s := range h {
				h[s] = r.kvHeld(s)
			}
			free = len(r.kvPools[kvGroupFull].free)
			return nil
		})
		return h, free
	}
	before, _ := held()
	if before[0] != conv0/kvBlock || before[3] != conv3/kvBlock || before[1] != run || before[2] != run {
		t.Fatalf("blocks held before the step %v, want [%d %d %d %d]", before, conv0/kvBlock, run, run, conv3/kvBlock)
	}
	r.SetKVSlotBusy(1, true)
	r.SetKVSlotBusy(2, true)
	if err := r.UseKVSlot(0); err != nil { // bind the idle, least recently written slot without writing it
		t.Fatal(err)
	}
	emb := func(i int) []float32 { return append([]float32(nil), m.EmbedResidentForTest(3+i%(vocab-3))...) }
	if _, err := r.StepBatch([]decoder.ResidentBatchSeq{{Slot: 1, Pos: run * kvBlock, Emb: emb(1)}, {Slot: 2, Pos: run * kvBlock, Emb: emb(2)}}); err != nil {
		t.Fatalf("the batched step: %v", err)
	}
	// Slot 0 (least recently written, bound, idle) is released whole; slot 3 (written later) is untouched; each running row took the one
	// block its write crossed into; the rest of slot 0's blocks are free.
	after, free := held()
	if want := [4]int{0, run + 1, run + 1, before[3]}; after != want {
		t.Errorf("blocks held after the step %v, want %v", after, want)
	}
	if want := before[0] - 2; free != want {
		t.Errorf("%d blocks free after the step, want %d (slot 0's %d less the two the rows took)", free, want, before[0])
	}
}
