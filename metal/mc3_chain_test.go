//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// The greedy and sampled chains (C-B01 / C-P02) under MC3: a generation alone on the batcher runs the chain with the
// resident held across its tokens (decoder's holdSolo), and hands the resident back the first token anyone else wants
// it. Before this, every generation on a model with more than one KV slot took an MC3 place, the chains required
// mc3 == nil, and serve (2 slots by default on Metal) never ran them (the post-merge peer read, 2026-10-03). These
// tests drive production's Generate with the batcher on, which is where that gap hid: the chains' own A/Bs ran with
// no batcher at all.

const mc3ChainFixture = "../testdata/llama-attnfa-tiny"

// mc3ChainModel loads the committed tiny fixture resident, with slots KV slots and MC3 on when slots > 1.
func mc3ChainModel(t *testing.T, slots int) *decoder.Model {
	t.Helper()
	m, err := decoder.Load(mc3ChainFixture, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: slots})
	if err != nil {
		t.Skipf("load %s: %v", mc3ChainFixture, err)
	}
	if !m.ResidentActive() {
		m.Close()
		t.Fatalf("%s did not go resident: %s", mc3ChainFixture, m.ResidentDecline())
	}
	if slots > 1 {
		if got := m.EnableResidentConcurrency(slots); got != slots {
			m.Close()
			t.Fatalf("EnableResidentConcurrency(%d) = %d", slots, got)
		}
	}
	return m
}

func mc3ChainPrompt(seed, n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (i*31 + seed*7 + 3) % 200
	}
	return p
}

// mc3ChainRun generates maxTok tokens and returns them.
func mc3ChainRun(t *testing.T, m *decoder.Model, prompt []int, maxTok int, sp decoder.SamplingParams) []int {
	t.Helper()
	ch, g := m.Generate(context.Background(), prompt, maxTok, sp)
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	return ids
}

// TestMC3Chain_aloneMatchesUnbatchedAndRunsTheChain: a lone generation on a 2-slot MC3 model emits exactly the tokens
// the unbatched model emits (whose decode runs the chain), greedy and at temperature 0.8, AND it ran them held: the
// batcher's HeldTokens counts them. The second check is the point: the tokens alone would also match on the old code,
// which never chained under MC3.
func TestMC3Chain_aloneMatchesUnbatchedAndRunsTheChain(t *testing.T) {
	plain := mc3ChainModel(t, 1)
	rf := plain.ResidentForwardForTest().(*metalResident)
	if why := rf.r.greedyChainWhyNot(); why != "" {
		plain.Close()
		t.Fatalf("the chain is not available on %s (%s): this gate would test nothing", mc3ChainFixture, why)
	}
	cases := []struct {
		name string
		sp   decoder.SamplingParams
	}{
		{"greedy", decoder.SamplingParams{}},
		{"T=0.8", decoder.SamplingParams{Temperature: 0.8, Seed: 11}},
	}
	// 12 tokens: below the 16-token fast-prefill floor (A-P02), so the unbatched model (no step) runs the sequential loop
	// and the MC3 model the step, both exact. At 16 and over the unbatched model takes the f16 pass and the two differ
	// by design; this test is about the chain, not the prefill route.
	prompt := mc3ChainPrompt(1, 12)
	want := map[string][]int{}
	for _, c := range cases {
		want[c.name] = mc3ChainRun(t, plain, prompt, 96, c.sp)
	}
	plain.Close()

	m := mc3ChainModel(t, 2)
	defer m.Close()
	for _, c := range cases {
		before := m.ResidentBatchStats()
		got := mc3ChainRun(t, m, prompt, 96, c.sp)
		st := m.ResidentBatchStats()
		if !slices.Equal(got, want[c.name]) {
			t.Fatalf("%s: alone under MC3 emitted %v, unbatched %v", c.name, got, want[c.name])
		}
		held := st.HeldTokens - before.HeldTokens
		if held < len(got)-2 {
			t.Fatalf("%s: %d of %d tokens ran held: the chain did not run under MC3 (stats %+v)", c.name, held, len(got), st)
		}
		t.Logf("%s: %d tokens identical to unbatched; %d ran held, in %d hold(s)", c.name, len(got), held, st.Holds-before.Holds)
	}
}

// TestMC3Chain_newcomerJoinsAndBothMatchAlone: generation A decodes alone (held, chained); B arrives mid-way. A must
// let the resident go, the two must batch, and both must emit exactly what each emits alone. A hold that never let go
// would deadlock B's prefill; one that let go too late would show no batched steps.
func TestMC3Chain_newcomerJoinsAndBothMatchAlone(t *testing.T) {
	pa, pb := mc3ChainPrompt(2, 48), mc3ChainPrompt(3, 33)
	const na, nb = 160, 64
	m := mc3ChainModel(t, 2)
	defer m.Close()
	wantA := mc3ChainRun(t, m, pa, na, decoder.SamplingParams{})
	wantB := mc3ChainRun(t, m, pb, nb, decoder.SamplingParams{})

	before := m.ResidentBatchStats()
	mr, _ := m.ResidentForwardForTest().(*metalResident)
	greedyBefore := 0
	if mr != nil && mr.r.batch != nil {
		greedyBefore = mr.r.batch.greedyDevRows
	}
	chA, gA := m.Generate(context.Background(), pa, na, decoder.SamplingParams{})
	var gotA []int
	for id := range chA {
		gotA = append(gotA, id)
		if len(gotA) == 24 {
			break
		}
	}
	doneB := make(chan []int, 1)
	go func() { doneB <- mc3ChainRun(t, m, pb, nb, decoder.SamplingParams{}) }()
	for id := range chA {
		gotA = append(gotA, id)
	}
	if err := gA.Err(); err != nil {
		t.Fatalf("A: %v", err)
	}
	var gotB []int
	select {
	case gotB = <-doneB:
	case <-time.After(60 * time.Second):
		t.Fatal("B did not finish within 60 s: the hold kept the resident from the newcomer")
	}
	st := m.ResidentBatchStats()
	if !slices.Equal(gotA, wantA) {
		t.Errorf("A with a newcomer: %v, alone %v", gotA, wantA)
	}
	if !slices.Equal(gotB, wantB) {
		t.Errorf("B joining a held chain: %v, alone %v", gotB, wantB)
	}
	held, steps := st.HeldTokens-before.HeldTokens, st.Steps-before.Steps
	if held == 0 || steps == 0 {
		t.Errorf("held tokens %d, batched steps %d: want both (A chained alone, then batched with B)", held, steps)
	}
	// E-P08: the batched steps' greedy rows came back as device-argmax ids, not logits rows (decoder's Greedy draws)
	if mr == nil || mr.r.batch == nil || mr.r.batch.greedyDevRows-greedyBefore == 0 {
		t.Errorf("no batched greedy row was served by the device argmax (E-P08's route did not run)")
	}
	t.Logf("A %d tokens, B %d, both identical to alone; %d held tokens, %d batched steps, %d holds", len(gotA), len(gotB), held, steps, st.Holds-before.Holds)
}

// TestMC3Chain_stalledConsumerDoesNotStarveANewcomer: A's consumer stops reading while A holds the resident. A newcomer
// must still be served (the send grace lets the hold go), and A, read again afterwards, must finish exactly as alone.
func TestMC3Chain_stalledConsumerDoesNotStarveANewcomer(t *testing.T) {
	pa, pb := mc3ChainPrompt(4, 40), mc3ChainPrompt(5, 30)
	m := mc3ChainModel(t, 2)
	defer m.Close()
	wantA := mc3ChainRun(t, m, pa, 80, decoder.SamplingParams{})
	wantB := mc3ChainRun(t, m, pb, 40, decoder.SamplingParams{})

	chA, gA := m.Generate(context.Background(), pa, 80, decoder.SamplingParams{})
	var gotA []int
	for id := range chA {
		gotA = append(gotA, id)
		if len(gotA) == 10 {
			break // A's consumer stalls here, with A holding
		}
	}
	doneB := make(chan []int, 1)
	go func() { doneB <- mc3ChainRun(t, m, pb, 40, decoder.SamplingParams{}) }()
	var gotB []int
	select {
	case gotB = <-doneB:
	case <-time.After(30 * time.Second):
		t.Fatal("B did not finish while A's consumer was stalled: the hold was not let go")
	}
	for id := range chA {
		gotA = append(gotA, id)
	}
	if err := gA.Err(); err != nil {
		t.Fatalf("A: %v", err)
	}
	if !slices.Equal(gotB, wantB) {
		t.Errorf("B beside a stalled A: %v, alone %v", gotB, wantB)
	}
	if !slices.Equal(gotA, wantA) {
		t.Errorf("A after its stall: %v, alone %v", gotA, wantA)
	}
}
