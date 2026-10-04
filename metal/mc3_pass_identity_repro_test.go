//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC3PassNewcomer_identityRepro and TestMC3PassStall_identityRepro: an OPEN defect, kept as its repro
// (docs/tasks/task-metal-audit-2026-10.md, A-P02 result). TestMC3Chain_newcomerJoinsAndBothMatchAlone and
// _stalledConsumerDoesNotStarveANewcomer with prompts of 100/80 and 90/70 tokens, which take the f16 batched pass instead of
// the step: on the 2-slot fixture both generations diverge from their alone runs some tens of tokens after the newcomer
// joins (A first differs at token 42 of 160, B at 47 of 64). RED at 0eb53e90, before the serve chain, and with the chain
// off; the pass is slot-independent (the same prompt and 60 decode steps on slot 0 and slot 1 match bit for bit). Opt-in
// (GOINFER_MC3_PASS_REPRO=1) because it is red until fixed; a fix turns it green.
func TestMC3PassNewcomer_identityRepro(t *testing.T) {
	pa, pb := mc3ChainPrompt(2, 100), mc3ChainPrompt(3, 80)
	const na, nb = 160, 64
	if os.Getenv("GOINFER_MC3_PASS_REPRO") != "1" {
		t.Skip("the open MC3 identity defect's repro (task doc, A-P02 result); set GOINFER_MC3_PASS_REPRO=1: it is RED until fixed")
	}
	m := mc3ChainModel(t, 2)
	defer m.Close()
	wantA := mc3ChainRun(t, m, pa, na, decoder.SamplingParams{})
	wantB := mc3ChainRun(t, m, pb, nb, decoder.SamplingParams{})

	before := m.ResidentBatchStats()
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
	fd := func(a, b []int) int {
		for i := range min(len(a), len(b)) {
			if a[i] != b[i] {
				return i
			}
		}
		return -1
	}
	t.Logf("first diff: A %d (of %d/%d), B %d (of %d/%d)", fd(gotA, wantA), len(gotA), len(wantA), fd(gotB, wantB), len(gotB), len(wantB))
	if !slices.Equal(gotA, wantA) {
		t.Errorf("A with a newcomer differs")
	}
	if !slices.Equal(gotB, wantB) {
		t.Errorf("B joining a held chain: %v, alone %v", gotB, wantB)
	}
	held, steps := st.HeldTokens-before.HeldTokens, st.Steps-before.Steps
	if held == 0 || steps == 0 {
		t.Errorf("held tokens %d, batched steps %d: want both (A chained alone, then batched with B)", held, steps)
	}
	t.Logf("A %d tokens, B %d, both identical to alone; %d held tokens, %d batched steps, %d holds", len(gotA), len(gotB), held, steps, st.Holds-before.Holds)
}

func TestMC3PassStall_identityRepro(t *testing.T) {
	pa, pb := mc3ChainPrompt(4, 90), mc3ChainPrompt(5, 70)
	if os.Getenv("GOINFER_MC3_PASS_REPRO") != "1" {
		t.Skip("the open MC3 identity defect's repro (task doc, A-P02 result); set GOINFER_MC3_PASS_REPRO=1: it is RED until fixed")
	}
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
