package decoder

import (
	"context"
	"hash/fnv"
	"math"
	"slices"
	"testing"
)

// HL-G2 (docs/tasks/task-forward-architecture-2026-10.md §6.1): the greedy chain under a gated processor, through the
// decode loop. fakeChainRowResident runs ahead as a real chain does: GreedyChainNext and GreedyChainNextRow queue the
// next forward with the device's pick before returning. Its logits hash the token at every position up to the one
// asked for, so a forward left running the device's token where the host picked another changes every later row.

type fakeChainRowResident struct {
	*fakeResident
	m     *Model
	avail bool
	kv    []int // the token each position holds
	byEmb map[string]int

	open      bool
	qPos      int       // the forward queued last: its position
	qRow      []float32 // and its row
	argmax    int
	nexts     int
	rows      int
	redoAt    []int
	breach    string // the first call that broke the chain's contract
	chainUsed bool
}

func embKey(emb []float32) string {
	b := make([]byte, 0, 4*len(emb))
	for _, v := range emb {
		u := math.Float32bits(v)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	return string(b)
}

func (f *fakeChainRowResident) tokenOf(emb []float32) int {
	if f.byEmb == nil {
		f.byEmb = map[string]int{}
		for id := range f.vocab {
			f.byEmb[embKey(f.m.embedResidentInto(id, nil))] = id
		}
	}
	id, ok := f.byEmb[embKey(emb)]
	if !ok && f.breach == "" {
		f.breach = "an embedding that is no token's row"
	}
	return id
}

// write puts tok at pos and returns the row a forward there produces.
func (f *fakeChainRowResident) write(pos, tok int) []float32 {
	for len(f.kv) <= pos {
		f.kv = append(f.kv, -1)
	}
	f.kv[pos] = tok
	h := fnv.New64a()
	for _, t := range f.kv[:pos+1] {
		h.Write([]byte{byte(t), byte(t >> 8), byte(t >> 16)})
	}
	seed := h.Sum64()
	row := make([]float32, f.vocab)
	for i := range row {
		x := seed ^ (uint64(i)+1)*0x9E3779B97F4A7C15
		x ^= x >> 31
		x *= 0xBF58476D1CE4E5B9
		row[i] = float32(x>>40) / float32(1<<24)
	}
	return row
}

func (f *fakeChainRowResident) Forward(emb []float32, pos int) ([]float32, error) {
	f.forwards++
	return f.write(pos, f.tokenOf(emb)), nil
}

func (f *fakeChainRowResident) ForwardN(embs [][]float32, start int) ([][]float32, error) {
	var out [][]float32
	for i, e := range embs {
		r, _ := f.Forward(e, start+i)
		out = append(out, r)
	}
	return out, nil
}

func (f *fakeChainRowResident) ForwardArgmax(emb []float32, pos int) (int, error) {
	f.argmax++
	return argmaxRow(f.write(pos, f.tokenOf(emb))), nil
}

func argmaxRow(r []float32) int {
	best := 0
	for i, v := range r {
		if v > r[best] {
			best = i
		}
	}
	return best
}

func (f *fakeChainRowResident) GreedyChainAvailable() bool { return f.avail }

func (f *fakeChainRowResident) GreedyChainStart(id, pos int) error {
	f.open, f.chainUsed = true, true
	f.qPos, f.qRow = pos, f.write(pos, id)
	return nil
}

func (f *fakeChainRowResident) take() ([]float32, int) {
	if !f.open && f.breach == "" {
		f.breach = "a token asked of a closed chain"
	}
	row := f.qRow
	p := argmaxRow(row)
	f.qPos, f.qRow = f.qPos+1, f.write(f.qPos+1, p) // run ahead with the device's pick
	return row, p
}

func (f *fakeChainRowResident) GreedyChainNext() (int, error) {
	f.nexts++
	_, p := f.take()
	return p, nil
}

func (f *fakeChainRowResident) GreedyChainNextRow() (int, []float32, error) {
	f.rows++
	row, p := f.take()
	return p, row, nil
}

func (f *fakeChainRowResident) GreedyChainRedo(id, pos int) error {
	if pos != f.qPos && f.breach == "" {
		f.breach = "a redo at a position the chain has not queued"
	}
	f.redoAt = append(f.redoAt, pos)
	f.qRow = f.write(pos, id)
	return nil
}

func (f *fakeChainRowResident) GreedyChainStop() { f.open = false }

type fakeChainRowBackend struct {
	Backend
	rf    *fakeChainRowResident
	avail bool
	calls int // the processor's calls
}

func (b *fakeChainRowBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeChainRowResident{fakeResident: &fakeResident{vocab: vocab}, m: m, avail: b.avail}
	return b.rf, true, nil
}
func (b *fakeChainRowBackend) Close() error { return nil }

func loadWithFakeChainRow(t *testing.T, name string, avail bool) (*Model, *fakeChainRowBackend) {
	t.Helper()
	be := &fakeChainRowBackend{avail: avail}
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

// TestHLChainRow_gatedProcessor is HL-G2: a gate that opens twice, a processor that bans the argmax on every other
// open step and leaves the rest alone. With the chain the tokens must equal the same turn without it, every open step
// must take its row from the chain, and a redo must run exactly at the position after each banned pick, and nowhere
// else.
func TestHLChainRow_gatedProcessor(t *testing.T) {
	prompt := []int{1, 2, 3}
	const n = 24
	open := func(gen []int) bool { l := len(gen); return (l >= 3 && l < 7) || (l >= 12 && l < 16) }
	run := func(name string, avail bool) (out []int, banned []int, be *fakeChainRowBackend) {
		m, be := loadWithFakeChainRow(t, name, avail)
		sp := SamplingParams{
			LogitProcessor: func(gen []int, logits []float32) {
				be.calls++
				if len(gen)%2 == 0 {
					logits[argmaxRow(logits)] = float32(math.Inf(-1))
					banned = append(banned, len(gen))
				}
			},
			LogitProcessorGate: open,
		}
		stream, g := m.Generate(context.Background(), prompt, n, sp)
		for id := range stream {
			out = append(out, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("%s: generate: %v", name, err)
		}
		return out, banned, be
	}
	ref, refBanned, refBe := run("fake-hl-chain-off", false)
	got, banned, be := run("fake-hl-chain-on", true)
	rf := be.rf
	t.Logf("%d tokens; chain: %d next, %d rows, redos at %v; banned at generated lengths %v; reference: %d argmax, %d full forwards",
		len(got), rf.nexts, rf.rows, rf.redoAt, banned, refBe.rf.argmax, refBe.rf.forwards)
	if refBe.rf.chainUsed || refBe.rf.argmax == 0 {
		t.Fatalf("the reference arm used the chain (%v) or never the fast path (%d argmax calls): it is not today's path", refBe.rf.chainUsed, refBe.rf.argmax)
	}
	if rf.breach != "" {
		t.Fatalf("the decode loop broke the chain's contract: %s", rf.breach)
	}
	if !slices.Equal(got, ref) || !slices.Equal(banned, refBanned) {
		t.Fatalf("with the chain the tokens are %v (banned at %v), without it %v (banned at %v)", got, banned, ref, refBanned)
	}
	if rf.nexts == 0 || rf.rows == 0 {
		t.Fatalf("the chain served %d closed and %d open steps: the test needs both", rf.nexts, rf.rows)
	}
	if rf.rows != be.calls {
		t.Errorf("the processor ran %d times and the chain returned %d rows: an open step left the chain", be.calls, rf.rows)
	}
	if rf.argmax != 0 {
		t.Errorf("with the chain available, %d steps still took ForwardArgmax", rf.argmax)
	}
	// A pick banned at generated length L is output token L, forwarded at position len(prompt)+L when it was emitted.
	var want []int
	for _, l := range banned {
		if l < len(got) {
			want = append(want, len(prompt)+l)
		}
	}
	if len(want) == 0 {
		t.Fatal("no banned pick was emitted: the test exercised no redo")
	}
	if !slices.Equal(rf.redoAt, want) {
		t.Fatalf("redos at %v, want exactly %v (the position after each banned pick)", rf.redoAt, want)
	}
}
