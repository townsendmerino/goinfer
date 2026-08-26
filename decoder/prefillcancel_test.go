package decoder

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Gates for G18 — prefill honoring cancellation.
//
// The before-state these pin: the serve layer passed r.Context() into drive
// correctly, but the context stopped at generateInto — prefillLogits,
// forwardLayersN and runLayersFromEmbedN took no context at all. An abandoned
// client therefore left a core prefilling to completion, measured at 47:38 of
// CPU with nothing attached, and a retrying harness stacked one such generation
// per retry.
//
// These use an ALREADY-cancelled context and a mid-flight cancel rather than a
// long prompt, so they are fast and deterministic: what is being gated is that
// the loop looks at all, and that it stops promptly once it does.

func benchModelOrSkip(t *testing.T) *Model {
	t.Helper()
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	if !m.canBatchN(64) {
		t.Skip("model has no batched prefill path")
	}
	return m
}

func promptIDs(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = 785 // any valid id; content is irrelevant to cancellation
	}
	return ids
}

// An already-cancelled context must stop the batched prefill at the FIRST layer,
// before any real work.
func TestPrefillCancelledBeforeStart(t *testing.T) {
	m := benchModelOrSkip(t)
	ids := promptIDs(512)
	cache := m.NewCache(len(ids) + 8)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	start := time.Now()
	_, err := m.forwardLayersN(ctx, ids, cache)
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("forwardLayersN with a cancelled context: err = %v, want context.Canceled", err)
	}
	// A 512-token prefill is seconds when it runs; returning in well under that
	// is the observable difference between checking and not checking.
	if elapsed > 2*time.Second {
		t.Errorf("cancelled prefill took %v — it should return before doing a layer's work", elapsed)
	}
	t.Logf("cancelled-before-start prefill returned in %v", elapsed)
}

// The retry-storm shape: a client gives up mid-prefill. The abandoned call must
// stop on its own, promptly, rather than run to completion — otherwise each
// retry stacks another generation on a server that is still burning the last.
func TestPrefillCancelMidFlight(t *testing.T) {
	m := benchModelOrSkip(t)
	// Long enough that an unchecked prefill would run far past the cancel.
	ids := promptIDs(3072)
	cache := m.NewCache(len(ids) + 8)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := m.forwardLayersN(ctx, ids, cache)
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-flight cancel: err = %v, want context.Canceled", err)
	}
	// Bound, not a threshold assertion on the machine's speed: the granularity is
	// one layer, so this asserts the loop noticed within a small multiple of that
	// rather than running the whole 3072-token prefill (minutes on CPU).
	if elapsed > 60*time.Second {
		t.Errorf("abandoned prefill ran %v after cancel — the per-layer check is not bounding it", elapsed)
	}
	t.Logf("mid-flight cancel observed after %v (cancel fired at 300ms)", elapsed)
}

// Scope check, not an assumption: the fix's claim is that DECODE already
// honored cancellation and only prefill did not. Verify that rather than
// asserting it in a commit message.
func TestDecodeCancellationStillHonored(t *testing.T) {
	m := benchModelOrSkip(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stream, gen := m.Generate(ctx, promptIDs(8), 4096, SamplingParams{Temperature: 0})
	got := 0
	for range stream {
		got++
		if got == 3 {
			cancel()
		}
		if got > 512 {
			t.Fatalf("decode ignored cancellation: still emitting after %d tokens", got)
		}
	}
	t.Logf("decode stopped %d tokens after cancel (gen.Err() = %v)", got-3, gen.Err())
	if got >= 4096 {
		t.Error("decode ran to maxTokens despite cancellation")
	}
}
