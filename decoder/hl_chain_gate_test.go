package decoder

import (
	"context"
	"hash/fnv"
	"math"
	"slices"
	"testing"
)

// HL-R1 (docs/measurements/hl-2026-11/registration.md): the greedy chain under a gated logit processor runs while the gate is closed
// and stops at the step the gate opens. Through Generate, with a fake chain resident whose logits at a position depend on the
// embedding written at every earlier position, so a forward that ran the wrong token, or a position the chain skipped, changes every
// later step. No GPU; the committed tiny fixture.

// chainGateFake is a resident whose logits at position p are a one-hot row chosen by a hash of the embeddings written at positions
// 0..p. It offers the greedy fast path and the greedy chain; the chain queues the forward past the token it returns, as Metal's does.
type chainGateFake struct {
	*fakeResident
	m       *Model
	chainOn bool // GreedyChainAvailable
	hist    map[int]uint64

	chainOpen bool
	chainPos  int

	starts, nexts, stops, fulls, argmaxes int
	events                                []string
}

func (f *chainGateFake) write(emb []float32, pos int) {
	h := fnv.New64a()
	for _, x := range emb {
		b := math.Float32bits(x)
		h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
	}
	f.hist[pos] = h.Sum64()
}

func (f *chainGateFake) row(pos int) []float32 {
	st := uint64(1469598103934665603)
	for p := 0; p <= pos; p++ {
		st = (st ^ f.hist[p]) * 1099511628211
	}
	out := make([]float32, f.vocab)
	out[st%uint64(f.vocab)] = 1
	return out
}

func rowArgmax(row []float32) int {
	best := 0
	for i, v := range row {
		if v > row[best] {
			best = i
		}
	}
	return best
}

func (f *chainGateFake) Forward(emb []float32, pos int) ([]float32, error) {
	f.write(emb, pos)
	f.fulls++
	f.events = append(f.events, "forward")
	return f.row(pos), nil
}

func (f *chainGateFake) ForwardN(embs [][]float32, startPos int) ([][]float32, error) {
	rows := make([][]float32, 0, len(embs))
	for i, e := range embs {
		f.write(e, startPos+i)
		rows = append(rows, f.row(startPos+i))
	}
	return rows, nil
}

func (f *chainGateFake) ForwardArgmax(emb []float32, pos int) (int, error) {
	f.write(emb, pos)
	f.argmaxes++
	return rowArgmax(f.row(pos)), nil
}

func (f *chainGateFake) GreedyChainAvailable() bool { return f.chainOn }
func (f *chainGateFake) GreedyChainStart(id, pos int) error {
	f.starts++
	f.events = append(f.events, "start")
	f.chainOpen, f.chainPos = true, pos
	f.write(f.m.embedResidentInto(id, nil), pos) // the forward queued for the token that opens the chain
	return nil
}
func (f *chainGateFake) GreedyChainNext() (int, error) {
	f.nexts++
	f.events = append(f.events, "next")
	tok := rowArgmax(f.row(f.chainPos))
	f.chainPos++
	f.write(f.m.embedResidentInto(tok, nil), f.chainPos) // the forward queued past the token returned
	return tok, nil
}
func (f *chainGateFake) GreedyChainStop() {
	if f.chainOpen {
		f.stops++
		f.events = append(f.events, "stop")
	}
	f.chainOpen = false
}

type chainGateBackend struct {
	Backend
	chainOn bool
	rf      *chainGateFake
}

func (b *chainGateBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &chainGateFake{fakeResident: &fakeResident{vocab: vocab}, m: m, chainOn: b.chainOn, hist: map[int]uint64{}}
	return b.rf, true, nil
}
func (b *chainGateBackend) Close() error { return nil }

func loadChainGate(t *testing.T, name string, chainOn bool) (*Model, *chainGateBackend) {
	t.Helper()
	be := &chainGateBackend{chainOn: chainOn}
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
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture not resident-eligible")
	}
	return m, be
}

func TestHLR_chainRunsWhileTheGateIsClosedAndStopsAtEachOpening(t *testing.T) {
	const n = 14
	// The gate opens for the steps whose generated-token count is 3 or 4, and 8: two openings, three closed runs.
	open := func(gen []int) bool { return len(gen) == 3 || len(gen) == 4 || len(gen) == 8 }

	run := func(chainOn bool, name string) ([]int, *chainGateFake, int) {
		m, be := loadChainGate(t, name, chainOn)
		processed := 0
		var rf *chainGateFake
		sp := SamplingParams{
			LogitProcessorGate: open,
			LogitProcessor: func(_ []int, row []float32) {
				processed++
				// ban the row's argmax, so the host's pick differs from the device's at every open step
				row[rowArgmax(row)] = float32(math.Inf(-1))
				if rf != nil && rf.chainOpen {
					t.Errorf("%s: the processor ran with the chain still open", name)
				}
			},
		}
		// the resident exists after Load; bind it for the closure above
		rf = be.rf
		stream, g := m.Generate(context.Background(), []int{1, 2, 3}, n, sp)
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("%s: generate: %v", name, err)
		}
		if len(out) != n {
			t.Fatalf("%s: %d tokens, want %d (an early EOS would skip the later gate windows)", name, len(out), n)
		}
		return out, be.rf, processed
	}

	want, off, openSteps := run(false, "fake-hlr-off")
	if openSteps == 0 || off.starts != 0 {
		t.Fatalf("test setup: chain off should never start (starts %d) and the gate should open (%d open steps)", off.starts, openSteps)
	}
	got, on, openSteps2 := run(true, "fake-hlr-on")
	if !slices.Equal(got, want) {
		t.Fatalf("tokens differ with the chain on:\n  chain %v\n  today %v", got, want)
	}
	if openSteps2 != openSteps {
		t.Errorf("the processor ran %d times with the chain, %d without", openSteps2, openSteps)
	}
	// 3 closed runs, 2 openings: a start for each run, a stop at each opening and one at the end of the turn
	if on.starts != 3 || on.stops != 3 {
		t.Errorf("chain starts %d, stops %d; want 3 and 3 (events %v)", on.starts, on.stops, on.events)
	}
	// every open step ran Forward for its full row and no closed step did: the same full-row forwards as today's path (the prompt's,
	// plus one per open step)
	if on.fulls != off.fulls {
		t.Errorf("full-row forwards %d with the chain, %d without", on.fulls, off.fulls)
	}
	if off.fulls < openSteps {
		t.Errorf("test setup: %d full-row forwards without the chain for %d open steps", off.fulls, openSteps)
	}
	// the chain served every closed step: no ForwardArgmax after the chain opened its first run
	if on.argmaxes != 0 {
		t.Errorf("%d closed steps took ForwardArgmax; the chain should have served them", on.argmaxes)
	}
}
