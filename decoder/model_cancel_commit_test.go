package decoder

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// TestGenerateResident_cancelCommitsExactlyWhatWasEmitted is audit R-02's gate: a cancelled
// resident generation must record resIDs as prompt+generated (what the cache actually holds at
// the cancellation point — the last-sampled token was never forwarded, so its K/V was never
// written), instead of forgetting and forcing the next turn to cold-prefill an interrupt that
// left nothing inconsistent behind.
//
// Uses the same fake resident backend as resident_seam_test.go — no GPU, no downloaded model —
// so this is gated on every push rather than only where CUDA/Metal hardware happens to be
// available.
func TestGenerateResident_cancelCommitsExactlyWhatWasEmitted(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	prompt := []int{1, 2, 3}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, gen := m.Generate(ctx, prompt, 4096, SamplingParams{})

	var got []int
	for tok := range stream {
		got = append(got, tok)
		if len(got) == 3 {
			cancel()
		}
		// Same bound as TestDecodeCancellationStillHonored: not a timing assertion, just a
		// backstop against the cancel signal being silently ignored.
		if len(got) > 512 {
			t.Fatalf("cancellation ignored: still emitting after %d tokens", len(got))
		}
	}

	if !errors.Is(gen.Err(), context.Canceled) {
		t.Fatalf("gen.Err() = %v, want context.Canceled", gen.Err())
	}
	if len(got) == 0 {
		t.Fatal("test setup: cancellation raced ahead of every token — got none, cannot check the commit")
	}

	want := append(append([]int{}, prompt...), got...)
	if !equalIntSlices(m.resIDs, want) {
		t.Errorf("after cancel, resIDs = %v, want prompt+emitted %v — the cache is consistent with "+
			"exactly the tokens the caller actually received and must be recorded, not forgotten",
			m.resIDs, want)
	}
}

// erroringResident is a ResidentForward that succeeds okCount times then fails — used to drive
// generateInto's OTHER early exit (a forward error, not cancellation) without going through
// fakeResident's ContextCap, which generateInto clamps maxTokens against UP FRONT (so it never
// actually calls Forward at the cap; it just decodes fewer tokens cleanly, no error).
type erroringResident struct {
	vocab   int
	okCount int
	calls   int
}

func (r *erroringResident) Forward(embedding []float32, pos int) ([]float32, error) {
	r.calls++
	if r.calls > r.okCount {
		return nil, fmt.Errorf("erroringResident: forced failure at call %d", r.calls)
	}
	out := make([]float32, r.vocab)
	out[pos%r.vocab] = 1
	return out, nil
}

func (r *erroringResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	rows := make([][]float32, 0, len(embeddings))
	for i := range embeddings {
		row, err := r.Forward(embeddings[i], startPos+i)
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func (r *erroringResident) UploadKV(int, []float32, []float32) error { return nil }
func (r *erroringResident) TruncateTo(int)                           {}
func (r *erroringResident) Reset()                                   {}
func (r *erroringResident) Close() error                             { return nil }

type erroringResidencyBackend struct {
	Backend
	er *erroringResident
}

func (b *erroringResidencyBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.er = &erroringResident{vocab: vocab, okCount: 1} // prompt's one prefill token succeeds
	return b.er, true, nil
}

func (b *erroringResidencyBackend) Close() error { return nil }

// TestGenerateResident_forwardErrorStillForgets is the negative case named in R-02's own fix
// note: the OTHER early exit in this same loop (a forward error, not cancellation) must keep
// forgetting, because a forward error can leave a PARTIAL write behind — unlike the cancellation
// exit, where the failing operation (the send) never touches the cache at all.
func TestGenerateResident_forwardErrorStillForgets(t *testing.T) {
	be := &erroringResidencyBackend{}
	name := "erroring-resident-r02"
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
		t.Fatalf("Load with erroring resident backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}

	// One token prompt: prefill's single Forward call (okCount=1) succeeds, so useGPU's
	// resIDs-forgetting-before-prefill already ran; the FIRST decode-loop Forward then fails.
	stream, gen := m.Generate(context.Background(), []int{1}, 4096, SamplingParams{})
	for range stream { //nolint:revive // draining the stream is the point
	}
	if gen.Err() == nil {
		t.Fatal("test setup: expected the erroring resident's forced failure, got nil")
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after a forward error, want nil — a partial write is possible on this "+
			"exit and must not be trusted", m.resIDs)
	}
}

// TestGenerateNgramSpeculative_residentCommitsAcceptedSequence is audit R-03's gate for
// spec_ngram.go's resident branch (the OTHER writer named there, alongside spec_eagle.go — see
// that file's own note on why its forget calls are out of scope): on completion, resIDs must be
// (a prefix of) prompt+everything actually emitted, not stay forgotten from the pre-prefill
// residentForgetIDs call. Recurrent families never reach this branch at all
// (validateNgramSpec rejects them before the goroutine starts), so there is nothing to gate for
// the forget-otherwise half of R-01/R-03's general shape here.
//
// Exactly prompt+emitted is NOT always achievable: a round's own trailing token is streamed
// before it is forwarded (forwarding happens at the START of the next round, as that round's
// targetVerify seq[0] — see the "one asymmetry" note in spec_ngram.go), so a return right after
// that specific emit is one token behind the stream. Safe (the cache is never claimed to hold
// more than it truly does) but the test has to allow for it rather than require exact equality.
func TestGenerateNgramSpeculative_residentCommitsAcceptedSequence(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() || !m.DecodeRunnerEligible() {
		t.Skip("fixture is not resident+DecodeRunner-eligible")
	}

	prompt := []int{1, 2, 3}
	// An empty-table NgramDrafter always misses (kEff=0 every round), so this degenerates to
	// the target's own greedy draw each round — deterministic against fakeResident's
	// pos%vocab argmax, and it still drives the exact resident ForwardN/TruncateTo/commit path
	// genNgramInto's resident branch uses.
	stream, gen, err := m.GenerateNgramSpeculative(context.Background(), prompt, 5, &NgramDrafter{}, 4, SamplingParams{})
	if err != nil {
		t.Fatalf("GenerateNgramSpeculative: %v", err)
	}
	var got []int
	for tok := range stream {
		got = append(got, tok)
	}
	if gen.Err() != nil {
		t.Fatalf("gen.Err() = %v, want nil", gen.Err())
	}
	if len(got) == 0 {
		t.Fatal("test setup: no tokens emitted, cannot check the commit")
	}

	want := append(append([]int{}, prompt...), got...)
	if len(m.resIDs) < len(want)-1 || len(m.resIDs) > len(want) {
		t.Fatalf("resIDs has %d token(s), want %d (prompt+emitted) or %d (one short, the "+
			"structural round-tail lag) — got %v, prompt+emitted was %v",
			len(m.resIDs), len(want), len(want)-1, m.resIDs, want)
	}
	if !equalIntSlices(m.resIDs, want[:len(m.resIDs)]) {
		t.Errorf("resIDs = %v is not a prefix of prompt+emitted %v — the resident branch's cache "+
			"must never claim tokens it did not actually forward", m.resIDs, want)
	}
}
