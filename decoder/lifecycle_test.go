package decoder

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Close's contract (docs/use-from-go.md, "Closing a model"): it may be called while a generation runs, ends it with ErrClosed, and returns
// only when nothing is running on the weights; a generation started afterwards is refused by name. These tests call Close the way a caller's
// main does, through Generate and a ranging consumer, with the consumer first reading and then not reading at all.

func closeWithin(t *testing.T, m *Model, d time.Duration) {
	t.Helper()
	done := make(chan error, 1)
	go func() { done <- m.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(d):
		t.Fatalf("Close did not return within %v: it is waiting on a generation that cannot end", d)
	}
}

func TestClose_duringGenerationEndsItWithErrClosed(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	out, g := m.Generate(context.Background(), []int{1, 2, 3}, 1_000_000, SamplingParams{})
	for range 3 { // a few tokens have flowed, so the generation is mid-decode
		if _, ok := <-out; !ok {
			t.Fatal("the generation ended before Close ran")
		}
	}
	drained := make(chan int)
	go func() {
		n := 0
		for range out {
			n++
		}
		drained <- n
	}()
	closeWithin(t, m, 30*time.Second)
	select {
	case <-drained:
	case <-time.After(10 * time.Second):
		t.Fatal("the channel did not close after Close")
	}
	if err := g.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generation.Err = %v, want ErrClosed", err)
	}
}

// The consumer has stopped reading, so the generation is blocked sending its next token: Close must still end it.
func TestClose_whenTheConsumerHasStoppedReading(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	out, g := m.Generate(context.Background(), []int{1, 2, 3}, 1_000_000, SamplingParams{})
	<-out
	time.Sleep(50 * time.Millisecond) // let it reach the blocked send
	closeWithin(t, m, 30*time.Second)
	for range out { // closed: this ends at once
	}
	if err := g.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generation.Err = %v, want ErrClosed", err)
	}
}

// After Close every Generate-family call is refused with ErrClosed and a closed channel, and a second Close is a no-op.
func TestClose_refusesLaterGenerationsAndIsIdempotent(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	out, g := m.Generate(context.Background(), []int{1, 2, 3}, 8, SamplingParams{})
	if _, ok := <-out; ok {
		t.Fatal("a generation on a closed model produced a token")
	}
	if err := g.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generation.Err = %v, want ErrClosed", err)
	}
}

// A caller's own cancellation is still its own: a generation cancelled by its ctx before Close reports the ctx error, not ErrClosed.
func TestClose_doesNotRelabelACallersCancellation(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx, cancel := context.WithCancel(context.Background())
	out, g := m.Generate(ctx, []int{1, 2, 3}, 1_000_000, SamplingParams{})
	<-out
	cancel()
	for range out {
	}
	if err := g.Err(); !errors.Is(err, context.Canceled) || errors.Is(err, ErrClosed) {
		t.Fatalf("Generation.Err = %v, want context.Canceled", err)
	}
}

// S5: on a .giw model the mapping is gone after Close, and a generation started then used to be a SIGSEGV.
func TestClose_giwModelRefusesGenerateAfterClose(t *testing.T) {
	giwPath := buildTinyGIWInt4(t, "../testdata/llama-tiny")
	m, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatalf("Load(.giw): %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	out, g := m.Generate(context.Background(), []int{1, 2, 3}, 8, SamplingParams{})
	for range out {
		t.Fatal("a generation on a closed .giw model produced a token")
	}
	if err := g.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generation.Err = %v, want ErrClosed", err)
	}
}

// A session's generation is a generation: Close ends it, and a session turn after Close is refused.
func TestClose_endsASessionGeneration(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	s := m.NewSession(64)
	out, g := s.Generate(context.Background(), []int{1, 2, 3}, 1_000_000, SamplingParams{})
	<-out
	closeWithin(t, m, 30*time.Second)
	for range out {
	}
	if err := g.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("session Generation.Err = %v, want ErrClosed", err)
	}
	out2, g2 := s.Generate(context.Background(), []int{1, 2, 3, 4}, 8, SamplingParams{})
	for range out2 {
		t.Fatal("a session turn on a closed model produced a token")
	}
	if err := g2.Err(); !errors.Is(err, ErrClosed) {
		t.Fatalf("session turn after Close: Err = %v, want ErrClosed", err)
	}
}

// Every method that reads the weights for a caller, not only the Generate family, is refused after Close and is waited for by it.
func TestClose_refusesTheWeightReadingCallsAfterClose(t *testing.T) {
	giwPath := buildTinyGIWInt4(t, "../testdata/llama-tiny")
	m, err := Load(giwPath, Options{})
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache(8)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	dst := make([]float32, m.Config().HiddenDim)
	calls := map[string]func() error{
		"HiddenLast":         func() error { _, e := m.HiddenLast([]int{1, 2, 3}); return e },
		"PromptHidden":       func() error { _, e := m.PromptHidden(ctx, []int{1, 2, 3}); return e },
		"PromptHiddenAll":    func() error { _, e := m.PromptHiddenAll(ctx, []int{1, 2, 3}); return e },
		"PromptHiddenMany":   func() error { _, e := m.PromptHiddenMany(ctx, [][]int{{1, 2}, {3, 4}}); return e },
		"PromptLogitsMany":   func() error { _, e := m.PromptLogitsMany(ctx, [][]int{{1, 2}}); return e },
		"ForwardCapture":     func() error { _, _, e := m.ForwardCapture(1, cache, []int{0}); return e },
		"OutputEmbeddingRow": func() error { return m.OutputEmbeddingRow(1, dst) },
		"LoadAdapter":        func() error { return m.LoadAdapter("a", t.TempDir()) },
		"NewBlockSpec":       func() error { _, e := m.NewBlockSpec(nil, nil); return e },
	}
	for name, call := range calls {
		if err := call(); !errors.Is(err, ErrClosed) {
			t.Errorf("%s after Close: %v, want ErrClosed", name, err)
		}
	}
}

// Prompt-state calls running back to back when Close arrives: each is cancelled or finishes, Close returns, and the loop ends with
// ErrClosed, never a crash or another error.
func TestClose_duringPromptStateCalls(t *testing.T) {
	prompt := make([]int, 100)
	for i := range prompt {
		prompt[i] = 1 + i%50
	}
	for round := 0; round < 20; round++ {
		m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
		if err != nil {
			t.Fatal(err)
		}
		res := make(chan error, 1)
		go func() {
			for {
				if _, e := m.PromptHiddenAll(context.Background(), prompt); e != nil {
					res <- e
					return
				}
				if _, e := m.PromptHiddenMany(context.Background(), [][]int{prompt, prompt[:50]}); e != nil {
					res <- e
					return
				}
			}
		}()
		time.Sleep(time.Duration(round%5) * 200 * time.Microsecond)
		closeWithin(t, m, 60*time.Second)
		select {
		case e := <-res:
			if !errors.Is(e, ErrClosed) {
				t.Fatalf("round %d: the loop ended with %v, want ErrClosed", round, e)
			}
		case <-time.After(30 * time.Second):
			t.Fatalf("round %d: the loop did not end after Close", round)
		}
	}
}
