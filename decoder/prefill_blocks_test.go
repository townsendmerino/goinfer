package decoder

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"testing"
)

// blockFake is mc3Fake as a resident that splits its prefill (ResidentPrefillBlocks): it reports a block and a
// per-row scratch figure, records every PrefillLast call, and can decline or cancel at a chosen call. mc3Fake's pass
// writes each position as a Forward would, so a continuation equals the whole pass by construction and any difference
// a test sees comes from how the decoder cut the prompt.
type blockFake struct {
	*mc3Fake
	block     int
	perRow    int64
	tailExact bool
	floor     int
	calls     [][2]int // startPos and rows of every PrefillLast, in order
	declineAt int      // the 1-based PrefillLast call that declines; 0: none
	onCall    func(n int)
}

func (f *blockFake) PrefillLast(ctx context.Context, embs [][]float32, startPos int) ([]float32, error) {
	f.calls = append(f.calls, [2]int{startPos, len(embs)})
	if f.onCall != nil {
		f.onCall(len(f.calls))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.declineAt == len(f.calls) {
		return nil, errors.New("blockFake: this pass declines")
	}
	return f.mc3Fake.PrefillLast(ctx, embs, startPos)
}
func (f *blockFake) PrefillBlockRows() int              { return f.block }
func (f *blockFake) PrefillScratchBytes(rows int) int64 { return int64(rows) * f.perRow }
func (f *blockFake) PrefillTailExact() bool             { return f.tailExact }
func (f *blockFake) PrefillKernelFloor() int            { return f.floor }

type blockFakeBackend struct {
	Backend
	rf  *blockFake
	set func(*blockFake)
}

func (b *blockFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &blockFake{mc3Fake: newMC3Fake(vocab, 1), tailExact: true}
	if b.set != nil {
		b.set(b.rf)
	}
	return b.rf, true, nil
}
func (b *blockFakeBackend) Close() error { return nil }

var blockFakeBackends atomic.Int32

func loadWithBlockFake(t *testing.T, set func(*blockFake)) (*Model, *blockFake) {
	t.Helper()
	be := &blockFakeBackend{set: set}
	name := fmt.Sprintf("fake-blocks-%d", blockFakeBackends.Add(1))
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
		t.Fatalf("Load with the block fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

func blockPrompt(m *Model, n int) []int {
	vocab := m.w.arch.VocabSize
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 5) % vocab
	}
	return ids
}

// TestPrefillBlocks_cutsALongPromptIntoPasses is the shape of the split: passes of the resident's block while a block
// and prefillTailMin rows remain, then one last pass, each starting where the one before ended.
func TestPrefillBlocks_cutsALongPromptIntoPasses(t *testing.T) {
	for _, c := range []struct {
		name          string
		block, prompt int
		want          [][2]int
	}{
		{"three passes", 10, 37, [][2]int{{0, 10}, {10, 10}, {20, 17}}},
		{"the last pass keeps the block's remainder", 10, 27, [][2]int{{0, 10}, {10, 17}}},
		{"a tail of exactly the minimum", 10, 28, [][2]int{{0, 10}, {10, 10}, {20, 8}}},
		{"one row short of a second pass", 10, 17, [][2]int{{0, 17}}},
		{"shorter than a block", 64, 20, [][2]int{{0, 20}}},
		{"no block reported", 0, 37, [][2]int{{0, 37}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, rf := loadWithBlockFake(t, func(f *blockFake) { f.block = c.block })
			if _, err := m.residentPrefillSeed(context.Background(), blockPrompt(m, c.prompt), 0, false); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(rf.calls, c.want) {
				t.Errorf("PrefillLast calls (startPos, rows) = %v, want %v", rf.calls, c.want)
			}
			for i, call := range rf.calls {
				if last := i == len(rf.calls)-1; len(rf.calls) > 1 && last && (call[1] < prefillTailMin || call[1] > c.block+prefillTailMin-1) {
					t.Errorf("the last pass has %d rows, outside [%d, %d]", call[1], prefillTailMin, c.block+prefillTailMin-1)
				}
			}
		})
	}
}

// TestPrefillBlocks_generationMatchesOnePass (G-I8-1's decoder half) runs the same prompt through Model.Generate on a
// resident that splits and one that does not: the same tokens and the same history in the resident.
func TestPrefillBlocks_generationMatchesOnePass(t *testing.T) {
	gen := func(block int) ([]int, []uint64, int) {
		m, rf := loadWithBlockFake(t, func(f *blockFake) { f.block = block })
		ch, g := m.Generate(context.Background(), blockPrompt(m, 61), 12, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return ids, slices.Clone(rf.hist[0]), len(rf.calls)
	}
	wantIDs, wantHist, onePass := gen(0)
	if onePass != 1 || len(wantIDs) == 0 {
		t.Fatalf("the reference made %d passes and %d tokens, want one pass", onePass, len(wantIDs))
	}
	for _, block := range []int{8, 10, 16} {
		ids, hist, passes := gen(block)
		if passes < 3 {
			t.Fatalf("block %d: %d passes over a 61-token prompt, so nothing was split", block, passes)
		}
		if !slices.Equal(ids, wantIDs) || !slices.Equal(hist, wantHist) {
			t.Errorf("block %d: the generation differs from one pass (tokens equal %v, history equal %v)",
				block, slices.Equal(ids, wantIDs), slices.Equal(hist, wantHist))
		}
	}
}

// TestPrefillBlocks_aDeclinedBlockKeepsTheOnesBefore: when a pass declines partway, the rows already written stay and
// the rest, from the declined block's first row, takes the token-by-token path.
func TestPrefillBlocks_aDeclinedBlockKeepsTheOnesBefore(t *testing.T) {
	want, _ := loadWithBlockFake(t, nil)
	prompt := blockPrompt(want, 37)
	wantLg, err := want.residentPrefillSeed(context.Background(), prompt, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	m, rf := loadWithBlockFake(t, func(f *blockFake) { f.block, f.declineAt = 10, 2 })
	lg, err := m.residentPrefillSeed(context.Background(), prompt, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(rf.calls, [][2]int{{0, 10}, {10, 10}}) {
		t.Errorf("PrefillLast calls = %v, want the first block and the declined second", rf.calls)
	}
	if rf.soloFwds != 27 {
		t.Errorf("%d sequential forwards after the decline, want the 27 rows from position 10", rf.soloFwds)
	}
	if !slices.Equal(lg, wantLg) {
		t.Error("the seed logits differ from an undeclined prefill's")
	}
}

// TestPrefillBlocks_cancelledBetweenBlocks: a context cancelled during one pass stops the prefill before the next and
// is returned, not treated as a decline.
func TestPrefillBlocks_cancelledBetweenBlocks(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m, rf := loadWithBlockFake(t, func(f *blockFake) {
		f.block = 10
		f.onCall = func(n int) {
			if n == 2 {
				cancel()
			}
		}
	})
	_, err := m.residentPrefillSeed(ctx, blockPrompt(m, 57), 0, false)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(rf.calls) != 2 || rf.soloFwds != 0 {
		t.Errorf("%d passes and %d sequential forwards after the cancel, want 2 and 0", len(rf.calls), rf.soloFwds)
	}
}

// TestPrefillBlockRows_needsAnExactContinuation: a block is used only when the resident's continuation equals its whole
// pass, and a block under the resident's kernel-class floor is raised to the floor for a prompt that reaches it.
func TestPrefillBlockRows_needsAnExactContinuation(t *testing.T) {
	m, rf := loadWithBlockFake(t, func(f *blockFake) { f.block = 10 })
	if got := m.prefillBlockRows(100); got != 10 {
		t.Errorf("prefillBlockRows = %d, want the resident's 10", got)
	}
	rf.tailExact = false
	if got := m.prefillBlockRows(100); got != 0 {
		t.Errorf("prefillBlockRows = %d for a resident whose continuation is not exact, want 0", got)
	}
	if _, ok := m.residentPrefillScratch(100); ok {
		t.Error("residentPrefillScratch priced a block for a resident that does not split")
	}
	rf.tailExact, rf.floor = true, 16
	if got := m.prefillBlockRows(100); got != 16 {
		t.Errorf("prefillBlockRows = %d with a 16-token kernel floor, want 16", got)
	}
	if got := m.prefillBlockRows(12); got != 10 {
		t.Errorf("prefillBlockRows = %d for a prompt under the floor, want the resident's 10", got)
	}
}

// TestAdmitPrefillMemory_pricesABlockedResidentByTheBlock (G-I8-4): with room for one block and not for the batched
// CPU sweep over the whole prompt, a resident request is admitted when the resident splits its prefill, and refused
// when it reports no block, when its block does not fit, and on the CPU path.
func TestAdmitPrefillMemory_pricesABlockedResidentByTheBlock(t *testing.T) {
	injectHostRAM(t, 64<<30)
	// The room is under prefillAttnScratchBudget alone, so the whole-prompt price cannot fit at any prompt length.
	injectHostRAMAvailable(t, 300<<20)
	const promptTokens, maxTokens = 20000, 4096

	m, rf := loadWithBlockFake(t, func(f *blockFake) { f.block, f.perRow = 1000, 1000 })
	if whole := prefillScratchBytes(m.Config(), promptTokens); whole <= int64(float64(300<<20)*prefillAvailFraction) {
		t.Fatalf("the whole-prompt price %d fits the room; the test proves nothing", whole)
	}
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err != nil {
		t.Errorf("a blocked resident's request was refused though one block fits: %v", err)
	}
	wantRows := int64(rf.block + prefillTailMin - 1)
	if got, ok := m.residentPrefillScratch(promptTokens); !ok || got != wantRows*rf.perRow+wantRows*4*int64(m.ResidentEmbedLen()) {
		t.Errorf("residentPrefillScratch = %d, %v; want the longest pass's %d rows of scratch and embeddings", got, ok, wantRows)
	}
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, false, 1); err == nil {
		t.Error("the CPU path was admitted at a blocked resident's price")
	}
	rf.perRow = 1 << 20 // one block no longer fits the room
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err == nil {
		t.Error("a request whose block does not fit was admitted: the resident's figure is not what is priced")
	}
	rf.block, rf.perRow = 0, 1000
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err == nil {
		t.Error("a resident that reports no block was admitted under the room the whole-prompt price needs")
	}
}
