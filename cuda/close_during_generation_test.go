//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// M6 (docs/measurements/library-surface-2026-10-09.md): Model.Close during a generation released the device under an open decode chain. This is
// the property on the CUDA resident path, through the public Generate and Close: a generation that has produced tokens on the device is ended by
// Close, which returns only after the device work stopped, and the model refuses generations afterward. Repeated, because the old failure was a
// race on where in a step Close landed. Skips without a device or when the model did not go resident (the property is about the device path).
func TestClose_duringResidentGeneration(t *testing.T) {
	for round := range 12 {
		m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Skipf("no CUDA load: %v", err)
		}
		if m.ResidentForwardForTest() == nil {
			m.Close()
			t.Skip("the model did not go resident on this device")
		}
		out, g := m.Generate(context.Background(), []int{1, 2, 3}, 1_000_000, decoder.SamplingParams{})
		for range 1 + round%4 { // land Close at different points of the first steps
			if _, ok := <-out; !ok {
				t.Fatalf("round %d: the generation ended before Close", round)
			}
		}
		if round%2 == 0 {
			time.Sleep(time.Duration(round) * time.Millisecond) // sometimes with the next send already blocked
		}
		done := make(chan error, 1)
		go func() { done <- m.Close() }()
		go func() {
			for range out {
			}
		}()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("round %d: Close: %v", round, err)
			}
		case <-time.After(60 * time.Second):
			t.Fatalf("round %d: Close did not return", round)
		}
		for range out {
		}
		if err := g.Err(); !errors.Is(err, decoder.ErrClosed) {
			t.Fatalf("round %d: Generation.Err = %v, want ErrClosed", round, err)
		}
		out2, g2 := m.Generate(context.Background(), []int{1, 2, 3}, 4, decoder.SamplingParams{})
		for range out2 {
			t.Fatalf("round %d: a generation after Close produced a token", round)
		}
		if err := g2.Err(); !errors.Is(err, decoder.ErrClosed) {
			t.Fatalf("round %d: after Close, Err = %v, want ErrClosed", round, err)
		}
	}
}
