package decoder

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// rideFake is mc3Fake plus a ragged step (ResidentRaggedStepper): a prefill chunk's rows write their positions as PrefillLast
// would, and a decode row is its StepBatch row. rides records each step that carried a chunk; rideErr makes the step refuse a
// chunk before it writes anything, as a backend that declines one does.
type rideFake struct {
	*mc3Fake
	rideErr  error
	attempts int // steps that were handed a chunk, refused ones included
	rides    []rideCall
}

type rideCall struct{ chunkRows, decodeRows, promptLen int }

func (f *rideFake) StepRagged(seqs []ResidentRaggedSeq) ([]ResidentRaggedOut, error) {
	f.enter()
	defer f.leave()
	for _, q := range seqs {
		if q.Prefill != nil {
			f.attempts++
			if f.rideErr != nil {
				return nil, f.rideErr
			}
		}
	}
	f.events = append(f.events, 'R')
	f.wait()
	var call rideCall
	out := make([]ResidentRaggedOut, len(seqs))
	for i, q := range seqs {
		if q.Prefill != nil {
			call.chunkRows += len(q.Embs)
			call.promptLen = q.Prefill.PromptLen
			for j, e := range q.Embs {
				f.fwd(q.Slot, e, q.Pos+j)
			}
			continue
		}
		if len(q.Embs) != 1 {
			return nil, fmt.Errorf("rideFake: a decode sequence with %d rows", len(q.Embs))
		}
		call.decodeRows++
		lg := f.fwd(q.Slot, q.Embs[0], q.Pos)
		f.last = append(f.last[:0], lg...)
		f.writes++
		switch {
		case q.Draw != nil && q.Draw.Greedy:
			out[i] = ResidentRaggedOut{ID: argmax(lg)}
		case q.Draw != nil:
			out[i] = ResidentRaggedOut{ID: fakeDraw(lg, q.Draw.Seed, q.Draw.Draw)}
		default:
			out[i] = ResidentRaggedOut{Logits: [][]float32{lg}, ID: -1}
		}
	}
	f.rides = append(f.rides, call)
	return out, nil
}

type rideFakeBackend struct {
	Backend
	rf    *rideFake
	slots int
}

func (b *rideFakeBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &rideFake{mc3Fake: newMC3Fake(vocab, b.slots)}
	return b.rf, true, nil
}
func (b *rideFakeBackend) Close() error { return nil }

var rideFakeBackends atomic.Int32

func loadWithRideFake(t *testing.T, slots int) (*Model, *rideFake) {
	t.Helper()
	be := &rideFakeBackend{slots: slots}
	name := fmt.Sprintf("fake-ride-%d", rideFakeBackends.Add(1))
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
		t.Fatalf("Load with the ride fake: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	return m, be.rf
}

// rideCell runs TestMC3_longPrefillChunksWhileOthersDecode's cell on a ride fake: two generations decoding, then a newcomer
// with a 1200-token prompt and a 256-token chunk. Every reply must equal the same conversation served alone.
func rideCell(t *testing.T, rideErr error) (*Model, *rideFake) {
	t.Helper()
	long := func(m *Model) []int {
		v := m.w.arch.VocabSize
		p := make([]int, 1200)
		for i := range p {
			p[i] = (i*7 + 3) % v
		}
		return p
	}
	gen := func(m *Model, prompt []int, maxTok int) []int {
		ch, g := m.Generate(context.Background(), prompt, maxTok, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Errorf("generate: %v", err)
		}
		return ids
	}
	mAlone, _ := loadWithRideFake(t, 4)
	d0, d1 := []int{1, 2, 3, 40}, []int{1, 2, 3, 50}
	aloneD0, aloneD1, aloneNew := gen(mAlone, d0, 60), gen(mAlone, d1, 60), gen(mAlone, long(mAlone), 4)

	m, rf := loadWithRideFake(t, 4)
	rf.delay = 3 * time.Millisecond
	rf.rideErr = rideErr
	m.prefillChunk = 256 // Options.ResidentPrefillChunk
	m.EnableResidentConcurrency(4)
	var got0, got1, gotNew []int
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); got0 = gen(m, d0, 60) }()
	go func() { defer wg.Done(); got1 = gen(m, d1, 60) }()
	go func() {
		defer wg.Done()
		time.Sleep(40 * time.Millisecond) // the two are decoding by now
		gotNew = gen(m, long(m), 4)
	}()
	wg.Wait()
	for _, c := range []struct {
		name      string
		got, want []int
	}{{"decoder 0", got0, aloneD0}, {"decoder 1", got1, aloneD1}, {"newcomer", gotNew, aloneNew}} {
		if !slices.Equal(c.got, c.want) {
			t.Errorf("%s: concurrent %v, alone %v", c.name, c.got, c.want)
		}
	}
	return m, rf
}

// TestRF3_newcomerChunksRideTheDecodeRuns: on a resident that takes ragged steps, a newcomer's prefill chunks are carried by
// the decode runs (mc3Prefill through rideChunk and the coalescer): every chunk is one step with decode rows beside it, the
// chunk's PromptLen is the whole prompt, and no chunk takes the resident for a pass of its own. Only the tail is a prefill pass.
// Docs: docs/tasks/task-forward-and-memory-2026-11.md, "RF3: plan and registration" (RF3-1).
func TestRF3_newcomerChunksRideTheDecodeRuns(t *testing.T) {
	m, rf := rideCell(t, nil)
	ev := string(rf.events)
	t.Logf("resident events: %d ragged steps, %d prefill passes, %d batched steps; rides %v", strings.Count(ev, "R"), strings.Count(ev, "P"), strings.Count(ev, "S"), rf.rides)
	if len(rf.rides) != 4 { // 1200 = 256 x 4 + 176: a chunk is cut while a chunk plus the tail minimum remain
		t.Fatalf("%d chunks rode a decode run, want 4", len(rf.rides))
	}
	rows := 0
	for i, c := range rf.rides {
		if c.chunkRows != 256 || c.promptLen != 1200 {
			t.Errorf("ride %d: a %d-row chunk of a %d-token prompt, want 256 of 1200", i, c.chunkRows, c.promptLen)
		}
		if c.decodeRows < 1 {
			t.Errorf("ride %d carried no decode row: the chunk ran as a pass of its own", i)
		}
		rows += c.decodeRows
	}
	if n := strings.Count(ev, "P"); n != 1 {
		t.Errorf("%d prefill passes, want 1 (the newcomer's tail): a chunk took the resident for itself", n)
	}
	st := m.ResidentBatchStats()
	if st.RidePasses != len(rf.rides) || st.RideRows != rows || st.RideNs <= 0 {
		t.Errorf("stats: %d ride passes, %d ride rows, %d ns; want %d, %d and a timed ride", st.RidePasses, st.RideRows, st.RideNs, len(rf.rides), rows)
	}
	if st.PrefillPasses != 3 { // the newcomer's tail and the two decoders' seeds, as PrefillPasses counts them
		t.Errorf("PrefillPasses = %d, want 3", st.PrefillPasses)
	}
}

// TestRF3_declinedChunkAlternates: when the ragged step refuses a chunk (a family it declines), the decode run it rode is
// served without it, and that chunk and the rest of the prompt go back to the alternation, a pass of its own and then a decode run.
// Every reply is unchanged, and the step is offered a chunk once.
func TestRF3_declinedChunkAlternates(t *testing.T) {
	m, rf := rideCell(t, errors.New("rideFake: declined"))
	ev := string(rf.events)
	if rf.attempts != 1 || len(rf.rides) != 0 {
		t.Errorf("the step was offered %d chunks and took %d, want 1 and 0", rf.attempts, len(rf.rides))
	}
	if n := strings.Count(ev, "P"); n != 5 { // 4 alternated chunks and the tail
		t.Errorf("%d prefill passes after the decline, want 5", n)
	}
	if st := m.ResidentBatchStats(); st.RidePasses != 0 {
		t.Errorf("RidePasses = %d after a decline, want 0", st.RidePasses)
	}
}
