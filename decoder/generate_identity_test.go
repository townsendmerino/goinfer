package decoder

import (
	"context"
	"encoding/json"
	"flag"
	"hash/fnv"
	"math"
	"os"
	"runtime"
	"sort"
	"testing"
)

var generateIdentityUpdate = flag.Bool("generate.identity.update", false, "rewrite testdata/generate_identity.json")

const generateIdentityGolden = "../testdata/generate_identity.json"

// identResident is a resident whose logits are a deterministic function of the input embedding and the position, so the plain
// forward, the on-device argmax and the greedy chain all agree on every token, and a decode that reads a stale position or the
// wrong token's embedding changes the stream. The counters record which path ran.
type identResident struct {
	vocab                                       int
	m                                           *Model
	forwards, argmaxes, chainStarts, chainNexts int
	chainTemp                                   float64
	chainID, chainPos                           int
}

func (f *identResident) logits(emb []float32, pos int) []float32 {
	h := fnv.New64a()
	var b [8]byte
	for _, x := range emb {
		u := math.Float32bits(x)
		b[0], b[1], b[2], b[3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
		h.Write(b[:4])
	}
	b[0], b[1], b[2], b[3] = byte(pos), byte(pos>>8), byte(pos>>16), byte(pos>>24)
	h.Write(b[:4])
	s := h.Sum64()
	out := make([]float32, f.vocab)
	for v := range out {
		out[v] = float32((s+uint64(v)*2654435761)>>7%4096) / 512
	}
	return out
}

func (f *identResident) Forward(emb []float32, pos int) ([]float32, error) {
	f.forwards++
	return f.logits(emb, pos), nil
}
func (f *identResident) ForwardN(embs [][]float32, start int) ([][]float32, error) {
	rows := make([][]float32, len(embs))
	for i := range embs {
		r, _ := f.Forward(embs[i], start+i)
		rows[i] = r
	}
	return rows, nil
}
func (f *identResident) UploadKV(layer, base int, keys, vals []float32) error { return nil }
func (f *identResident) TruncateTo(pos int)                                   {}
func (f *identResident) Reset()                                               {}
func (f *identResident) Close() error                                         { return nil }
func (f *identResident) ResidentCaps() ResidentCaps                           { return ResidentCaps{} }

// identArgmax adds the on-device argmax; identChain adds the greedy chain on top of it.
type identArgmax struct{ *identResident }

func (f identArgmax) ForwardArgmax(emb []float32, pos int) (int, error) {
	f.argmaxes++
	return argmaxF32(f.logits(emb, pos)), nil
}

type identChain struct{ identArgmax }

func (f identChain) GreedyChainAvailable() bool { return true }
func (f identChain) GreedyChainStart(id, pos int) error {
	f.chainStarts++
	f.chainID, f.chainPos = id, pos
	return nil
}
func (f identChain) GreedyChainNext() (int, error) {
	f.chainNexts++
	emb := make([]float32, f.m.w.Cfg.HiddenDim)
	f.m.embedToken(f.chainID, emb)
	tok := argmaxF32(f.logits(emb, f.chainPos))
	f.chainID, f.chainPos = tok, f.chainPos+1
	return tok, nil
}
func (f identChain) GreedyChainStop() {}

// identTopK adds the device top-K row, identSample the device Gumbel draw, identSampleChain the chained draw (with the greedy chain).
type identTopK struct{ *identResident }

func (f identTopK) TopKAvailable() bool { return true }
func (f identTopK) ForwardTopK(emb []float32, pos, k int, temperature float64, wantZ bool) (TopKRow, error) {
	l := f.logits(emb, pos)
	idx := make([]int, len(l))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if l[idx[a]] != l[idx[b]] {
			return l[idx[a]] > l[idx[b]]
		}
		return idx[a] < idx[b]
	})
	k = min(k, len(l))
	row := TopKRow{IDs: make([]int32, k), Logits: make([]float32, k), Full: func() ([]float32, error) { return l, nil }}
	for i := 0; i < k; i++ {
		row.IDs[i], row.Logits[i] = int32(idx[i]), l[idx[i]]
	}
	if wantZ {
		mx := float64(l[idx[0]])
		for _, x := range l {
			row.Z += math.Exp((float64(x) - mx) / temperature)
		}
	}
	return row, nil
}

func (f *identResident) gumbelPick(l []float32, temperature float64, seed, draw uint64) int {
	best, bestV := 0, math.Inf(-1)
	for v, x := range l {
		h := fnv.New64a()
		var b [24]byte
		for i, u := range [3]uint64{seed, draw, uint64(v)} {
			for j := 0; j < 8; j++ {
				b[i*8+j] = byte(u >> (8 * j))
			}
		}
		h.Write(b[:])
		u := (float64(h.Sum64()>>11) + 0.5) / (1 << 53)
		if g := float64(x)/temperature - math.Log(-math.Log(u)); g > bestV {
			best, bestV = v, g
		}
	}
	return best
}

type identSample struct{ *identResident }

func (f identSample) SampleAvailable() bool { return true }
func (f identSample) ForwardSample(emb []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	f.argmaxes++
	return f.gumbelPick(f.logits(emb, pos), temperature, seed, draw), nil
}

type identSampleChain struct{ identChain }

func (f identSampleChain) SampleAvailable() bool      { return true }
func (f identSampleChain) SampleChainAvailable() bool { return true }
func (f identSampleChain) ForwardSample(emb []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	f.argmaxes++
	return f.gumbelPick(f.logits(emb, pos), temperature, seed, draw), nil
}
func (f identSampleChain) SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error {
	f.chainStarts++
	f.chainID, f.chainPos = id, pos
	f.chainTemp = temperature
	return nil
}
func (f identSampleChain) SampleChainNext(seed, draw uint64) (int, error) {
	f.chainNexts++
	emb := make([]float32, f.m.w.Cfg.HiddenDim)
	f.m.embedToken(f.chainID, emb)
	tok := f.gumbelPick(f.logits(emb, f.chainPos), f.chainTemp, seed, draw)
	f.chainID, f.chainPos = tok, f.chainPos+1
	return tok, nil
}

type identBackend struct {
	Backend
	kind string
	rf   *identResident
	m    *Model
}

func (b *identBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.m = m
	b.rf = &identResident{vocab: vocab, m: m}
	switch b.kind {
	case "argmax":
		return identArgmax{b.rf}, true, nil
	case "chain":
		return identChain{identArgmax{b.rf}}, true, nil
	case "topk":
		return identTopK{b.rf}, true, nil
	case "sample":
		return identSample{b.rf}, true, nil
	case "samplechain":
		return identSampleChain{identChain{identArgmax{b.rf}}}, true, nil
	}
	return b.rf, true, nil
}
func (b *identBackend) Close() error { return nil }

type identCase struct {
	Name   string  `json:"name"`
	Tokens [][]int `json:"tokens"`
	Counts [5]int  `json:"counts"` // forwards, argmaxes, chain starts, chain nexts, tokens emitted
}

func collect(t *testing.T, ch <-chan int, g *Generation) []int {
	t.Helper()
	var out []int
	for id := range ch {
		out = append(out, id)
	}
	if g.Err() != nil {
		t.Fatalf("generation: %v", g.Err())
	}
	return out
}

func identityRun(t *testing.T) []identCase {
	t.Helper()
	prompt := []int{1, 2, 3, 4, 5}
	sampled := SamplingParams{Temperature: 0.8, TopK: 40, TopP: 0.95, Seed: 7, RepeatPenalty: 1.1}
	minP := SamplingParams{Temperature: 1.2, MinP: 0.05, Seed: 11, PresencePenalty: 0.3}
	var out []identCase
	run := func(name, kind string, sp SamplingParams, session bool) {
		var m *Model
		var be *identBackend
		if kind == "cpu" {
			var err error
			if m, err = Load(tinyFixture(t), Options{Backend: "cpu"}); err != nil {
				t.Fatal(err)
			}
		} else {
			be = &identBackend{kind: kind}
			bn := "ident-" + kind
			RegisterBackend(bn, func() (Backend, error) {
				cpu, err := NewBackend("cpu")
				if err != nil {
					return nil, err
				}
				be.Backend = cpu
				return be, nil
			})
			var err error
			if m, err = Load(tinyFixture(t), Options{Backend: bn}); err != nil {
				t.Fatal(err)
			}
			if !m.ResidentActive() {
				t.Skip("fixture is not resident-eligible")
			}
		}
		defer m.Close()
		c := identCase{Name: name}
		if session {
			s := m.NewSession(64)
			ch, g := s.Generate(context.Background(), prompt, 8, sp)
			first := collect(t, ch, g)
			ext := append(append(append([]int(nil), prompt...), first...), 9)
			ch, g = s.Generate(context.Background(), ext, 8, sp)
			c.Tokens = [][]int{first, collect(t, ch, g)}
		} else {
			ch, g := m.Generate(context.Background(), prompt, 24, sp)
			c.Tokens = [][]int{collect(t, ch, g)}
		}
		for _, ts := range c.Tokens {
			c.Counts[4] += len(ts)
		}
		if be != nil {
			c.Counts[0], c.Counts[1] = be.rf.forwards, be.rf.argmaxes
			c.Counts[2], c.Counts[3] = be.rf.chainStarts, be.rf.chainNexts
		}
		out = append(out, c)
	}
	run("cpu/greedy", "cpu", SamplingParams{}, false)
	run("cpu/sampled", "cpu", sampled, false)
	run("cpu/minp", "cpu", minP, false)
	run("cpu/session-prefix-reuse", "cpu", sampled, true)
	run("resident/greedy-logits", "plain", SamplingParams{}, false)
	run("resident/sampled", "plain", sampled, false)
	run("resident/greedy-argmax", "argmax", SamplingParams{}, false)
	run("resident/greedy-chain", "chain", SamplingParams{}, false)
	run("resident/topk-sampled", "topk", SamplingParams{Temperature: 0.8, TopK: 40, TopP: 0.95, Seed: 7}, false)
	run("resident/temperature-draw", "sample", SamplingParams{Temperature: 0.9, Seed: 5}, false)
	run("resident/sampled-chain", "samplechain", SamplingParams{Temperature: 0.9, Seed: 5}, false)
	run("resident/top1-sampled-chain", "chain", SamplingParams{Temperature: 0.7, TopK: 1, Seed: 3}, false)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// TestGenerateInto_tokenStreamIdentity holds generateInto's emitted tokens, and which decode path produced them, to a recorded
// stream on the tiny fixture: CPU greedy and seeded sampling (top-k, top-p, min-p, penalties), a session extending its own prefix,
// and a fake resident taking the logits path, the device argmax and the greedy chain. A refactor of generateInto that is meant to
// change nothing must leave every stream and counter as recorded (-generate.identity.update records them).
func TestGenerateInto_tokenStreamIdentity(t *testing.T) {
	got := identityRun(t)
	if *generateIdentityUpdate {
		b, err := json.MarshalIndent(got, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(generateIdentityGolden, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", generateIdentityGolden)
		return
	}
	raw, err := os.ReadFile(generateIdentityGolden)
	if err != nil {
		t.Fatal(err)
	}
	var want []identCase
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("%d cases, recorded %d", len(got), len(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		gj, _ := json.Marshal(g)
		wj, _ := json.Marshal(w)
		if string(gj) != string(wj) {
			t.Errorf("%s:\n got  %s\n want %s", g.Name, gj, wj)
		}
		if len(g.Tokens) == 0 || g.Counts[4] == 0 {
			t.Errorf("%s: VOID: no token was generated", g.Name)
		}
	}
	// The resident cases must have taken the path their name says.
	for _, c := range got {
		switch c.Name {
		case "resident/greedy-argmax":
			if c.Counts[1] == 0 {
				t.Errorf("%s: the device argmax never ran", c.Name)
			}
		case "resident/greedy-chain":
			if c.Counts[3] == 0 {
				t.Errorf("%s: the greedy chain never ran", c.Name)
			}
		}
	}
}

// TestGenerateInto_allocsPerGenerate logs the allocations of one 24-token Generate on the tiny fixture (CPU greedy, CPU sampled,
// and the fake resident's chain), the figure a refactor of generateInto's locals must leave unchanged; run it with -v before and
// after such a change and compare the three numbers.
func TestGenerateInto_allocsPerGenerate(t *testing.T) {
	prompt := []int{1, 2, 3, 4, 5}
	for _, c := range []struct {
		name, kind string
		sp         SamplingParams
	}{
		{"cpu/greedy", "cpu", SamplingParams{}},
		{"cpu/sampled", "cpu", SamplingParams{Temperature: 0.8, TopK: 40, TopP: 0.95, Seed: 7, RepeatPenalty: 1.1}},
		{"resident/greedy-chain", "chain", SamplingParams{}},
	} {
		var m *Model
		var err error
		if c.kind == "cpu" {
			m, err = Load(tinyFixture(t), Options{Backend: "cpu"})
		} else {
			be := &identBackend{kind: c.kind}
			bn := "ident-allocs-" + c.kind
			RegisterBackend(bn, func() (Backend, error) {
				cpu, e := NewBackend("cpu")
				if e != nil {
					return nil, e
				}
				be.Backend = cpu
				return be, nil
			})
			m, err = Load(tinyFixture(t), Options{Backend: bn})
		}
		if err != nil {
			t.Fatal(err)
		}
		run := func() {
			ch, g := m.Generate(context.Background(), prompt, 24, c.sp)
			for range ch { //nolint:revive // draining the stream is the point
			}
			_ = g
		}
		run() // warm the caches the first generation fills
		allocs := testing.AllocsPerRun(5, run)
		var a, b runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&a)
		for range 5 {
			run()
		}
		runtime.ReadMemStats(&b)
		t.Logf("%-24s %.0f allocs, %d bytes per Generate", c.name, allocs, (b.TotalAlloc-a.TotalAlloc)/5)
		m.Close()
	}
}
