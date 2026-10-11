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
