package decoder

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// donorFake is copyFake with a sliding window per slot (ResidentReusableKVSlot): held[s] is how many of a slot's first
// rows its window still holds (absent = all of them). It records each copy's source slot.
type donorFake struct {
	*copyFake
	held map[int]int
	srcs []int
}

func (f *donorFake) ReusableKVSlot(slot, prefix int) int {
	if h, ok := f.held[slot]; ok {
		return min(prefix, h)
	}
	return prefix
}

func (f *donorFake) CopyKV(src, dst, start, end int) error {
	f.srcs = append(f.srcs, src)
	return f.copyFake.CopyKV(src, dst, start, end)
}

type donorFakeBackend struct {
	Backend
	rf    *donorFake
	slots int
}

func (b *donorFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &donorFake{copyFake: &copyFake{mc3Fake: newMC3Fake(vocab, b.slots)}, held: map[int]int{}}
	return b.rf, true, nil
}
func (b *donorFakeBackend) Close() error { return nil }

func loadWithDonorFake(t *testing.T, slots int) (*Model, *donorFake) {
	t.Helper()
	be := &donorFakeBackend{slots: slots}
	name := fmt.Sprintf("fake-donor-%d", mc3FakeBackends.Add(1))
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with the donor fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

// TestKV2_donorWhoseWindowHoldsTheRows is KV2 follow-up 2 (docs/tasks/task-kv-architecture-2026-10.md): three slots,
// conversations 0 and 1 leave the shared preamble's blocks in slots 0 and 1, and slot 0's window has moved past them. A
// third conversation sharing the preamble lands in slot 2 and must copy the preamble from slot 1, the donor that still
// holds it, not decline on slot 0 (the first of the longest hits) and prefill cold; its reply is the reply it gives
// alone.
func TestKV2_donorWhoseWindowHoldsTheRows(t *testing.T) {
	const maxTok = 6
	alone := kv2Alone(t, 3, 1, maxTok, false)
	m, rf := loadWithDonorFake(t, 3)
	vocab := m.w.arch.VocabSize
	pre := kv2Preamble(vocab)
	convs := []*kv2Conv{newKV2Conv(pre, 0, vocab), newKV2Conv(pre, 1, vocab), newKV2Conv(pre, 2, vocab)}
	convs[0].turn(t, context.Background(), m, 0, maxTok)
	convs[1].turn(t, context.Background(), m, 1, maxTok)
	rf.held[0] = 0 // slot 0's window no longer holds the preamble's rows
	before := len(rf.srcs)
	convs[2].turn(t, context.Background(), m, 2, maxTok)
	if got := convs[2].shared[0]; got == 0 {
		t.Fatalf("conversation 2 copied nothing (copies from %v): a donor that holds the rows was passed over", rf.srcs[before:])
	}
	if srcs := rf.srcs[before:]; !slices.Equal(srcs, []int{1}) {
		t.Errorf("conversation 2 copied from slots %v, want [1]: slot 0's window no longer holds the rows", srcs)
	}
	if !slices.Equal(convs[2].turns[0].ids, alone[2][0].ids) {
		t.Errorf("conversation 2's reply %v differs from its reply alone %v", convs[2].turns[0].ids, alone[2][0].ids)
	}
}

// kvChooseDonor's rule on its own: longest capped hit first, the lower slot on a tie, skipping a slot whose window has
// moved past its hit; nothing when no candidate beats reuse.
func TestKVChooseDonor(t *testing.T) {
	res := &donorFake{held: map[int]int{0: 0}}
	for _, c := range []struct {
		hits         []int
		dst, reuse   int
		wantSrc, hit int
	}{
		{[]int{320, 320, 0}, 2, 0, 1, 320}, // slot 0 lost its rows: the tie goes to slot 1
		{[]int{0, 320, 384}, 0, 0, 2, 384}, // the longest hit first
		{[]int{0, 320, 320}, 0, 0, 1, 320}, // a tie goes to the lower slot
		{[]int{320, 0, 0}, 2, 0, -1, 0},    // the only candidate lost its rows
		{[]int{0, 256, 0}, 0, 256, -1, 0},  // nothing beats reuse
	} {
		src, hit, _ := kvChooseDonor(res, c.hits, kvmgr.ClassMap{}, 1000, c.dst, c.reuse)
		if src != c.wantSrc || hit != c.hit {
			t.Errorf("hits %v dst %d reuse %d: got slot %d hit %d, want slot %d hit %d", c.hits, c.dst, c.reuse, src, hit, c.wantSrc, c.hit)
		}
	}
}
