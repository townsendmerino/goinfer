package decoder

import (
	"context"
	"errors"
	"testing"
	"time"
)

// TestGeneration_errReportsCancellation: a generation whose context is cancelled or past its deadline, before it starts or mid-decode, ends its
// stream and reports the context's error through Err (errors.Is against context.Canceled), never a clean nil end.
func TestGeneration_errReportsCancellation(t *testing.T) {
	m, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	prompt := []int{1, 2, 3, 4, 5}
	t.Run("before it starts", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		ch, g := m.Generate(ctx, prompt, 64, SamplingParams{})
		for range ch {
		}
		if !errors.Is(g.Err(), context.Canceled) {
			t.Errorf("Err() = %v, want context.Canceled", g.Err())
		}
	})
	t.Run("an expired deadline", func(t *testing.T) {
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
		defer cancel()
		ch, g := m.Generate(ctx, prompt, 64, SamplingParams{})
		for range ch {
		}
		if !errors.Is(g.Err(), context.DeadlineExceeded) {
			t.Errorf("Err() = %v, want context.DeadlineExceeded", g.Err())
		}
	})
	t.Run("mid-decode", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		ch, g := m.Generate(ctx, prompt, 4096, SamplingParams{})
		n := 0
		for range ch {
			if n++; n == 2 {
				cancel()
			}
		}
		if n < 2 {
			t.Fatalf("VOID: the stream ended after %d tokens, before the cancel (err %v)", n, g.Err())
		}
		if !errors.Is(g.Err(), context.Canceled) {
			t.Errorf("after %d tokens, Err() = %v, want context.Canceled", n, g.Err())
		}
	})
}
