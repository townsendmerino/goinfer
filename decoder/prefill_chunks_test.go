package decoder

import (
	"fmt"
	"sync/atomic"
	"testing"
)

// chunkFake is mc3Fake as a resident that cuts its own long prefill (ResidentPrefillChunks, CUDA's shape): it reports a chunk, a
// per-row scratch figure and a tail minimum, and is NOT a ResidentPrefillBlocks, so the decoder does not split its prompts.
type chunkFake struct {
	*mc3Fake
	chunk   int
	perRow  int64
	tailMin int
}

func (f *chunkFake) PrefillChunkRows() int              { return f.chunk }
func (f *chunkFake) PrefillScratchBytes(rows int) int64 { return int64(rows) * f.perRow }
func (f *chunkFake) PrefillTailMinRows() int            { return f.tailMin }

type chunkFakeBackend struct {
	Backend
	rf  *chunkFake
	set func(*chunkFake)
}

func (b *chunkFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &chunkFake{mc3Fake: newMC3Fake(vocab, 1)}
	if b.set != nil {
		b.set(b.rf)
	}
	return b.rf, true, nil
}
func (b *chunkFakeBackend) Close() error { return nil }

var chunkFakeBackends atomic.Int32

func loadWithChunkFake(t *testing.T, set func(*chunkFake)) (*Model, *chunkFake) {
	t.Helper()
	be := &chunkFakeBackend{set: set}
	name := fmt.Sprintf("fake-chunks-%d", chunkFakeBackends.Add(1))
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
		t.Fatalf("Load with the chunk fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

// TestAdmitPrefillMemory_pricesAChunkingResidentByItsChunk (KV3, CUDA's half): with room for one chunk's scratch and the suffix's
// embeddings and not for the batched CPU sweep over the whole prompt, a request on a resident that cuts its own prefill is admitted;
// it is refused when its chunk does not fit, when it reports no chunk, and on the CPU path. The price is the resident's figure for
// min(prompt, chunk) rows plus the embeddings of the whole prompt.
func TestAdmitPrefillMemory_pricesAChunkingResidentByItsChunk(t *testing.T) {
	injectHostRAM(t, 64<<30)
	injectHostRAMAvailable(t, 300<<20)
	const promptTokens, maxTokens = 20000, 4096

	m, rf := loadWithChunkFake(t, func(f *chunkFake) { f.chunk, f.perRow, f.tailMin = 512, 1000, 16 })
	if whole := prefillScratchBytes(m.Config(), promptTokens); whole <= int64(float64(300<<20)*prefillAvailFraction) {
		t.Fatalf("the whole-prompt price %d fits the room; the test proves nothing", whole)
	}
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err != nil {
		t.Errorf("a chunking resident's request was refused though one chunk fits: %v", err)
	}
	want := int64(rf.chunk)*rf.perRow + int64(promptTokens)*4*int64(m.ResidentEmbedLen())
	if got, ok := m.residentPrefillScratch(promptTokens); !ok || got != want {
		t.Errorf("residentPrefillScratch = %d, %v; want one chunk's scratch plus the whole suffix's embeddings, %d", got, ok, want)
	}
	if got, _ := m.residentPrefillScratch(100); got != 100*rf.perRow+int64(100)*4*int64(m.ResidentEmbedLen()) {
		t.Errorf("a prompt shorter than the chunk is priced at its own rows, got %d", got)
	}
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, false, 1); err == nil {
		t.Error("the CPU path was admitted at a chunking resident's price")
	}
	rf.perRow = 1 << 20
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err == nil {
		t.Error("a request whose chunk does not fit was admitted: the resident's figure is not what is priced")
	}
	rf.chunk, rf.perRow = 0, 1000
	if err := m.AdmitPrefillMemoryShare(promptTokens, maxTokens, true, 1); err == nil {
		t.Error("a resident that reports no chunk was admitted under the room the whole-prompt price needs")
	}
}

// TestPrefillTailMinRows: the shortest last pass is the default 8 unless the resident asks for more, and never less.
func TestPrefillTailMinRows(t *testing.T) {
	m, rf := loadWithChunkFake(t, func(f *chunkFake) { f.chunk, f.perRow, f.tailMin = 512, 1, 16 })
	if got := m.prefillTailMinRows(); got != 16 {
		t.Errorf("prefillTailMinRows = %d for a resident that asks for 16", got)
	}
	rf.tailMin = 3
	if got := m.prefillTailMinRows(); got != prefillTailMin {
		t.Errorf("prefillTailMinRows = %d: a resident cannot lower the default %d", got, prefillTailMin)
	}
	m2, _ := loadWithBlockFake(t, func(f *blockFake) { f.block = 10 })
	if got := m2.prefillTailMinRows(); got != prefillTailMin {
		t.Errorf("a resident with no tail minimum got %d, want the default %d", got, prefillTailMin)
	}
}
