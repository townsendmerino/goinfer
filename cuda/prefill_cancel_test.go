//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillCancel pins the guarantee chunked prefill weakened and this restores: a cancelled
// request stops the prompt, rather than running it to completion on a GPU nobody is waiting for.
//
// The sequential fallback checks ctx.Err() PER TOKEN, so an abandoned client does not leave the whole
// prompt streaming through the device. A batched pass has no such loop, so without Prefiller carrying
// a context the granularity is the whole pass (one 512-row MoE chunk takes seconds, against
// milliseconds for the per-token path to notice). That is the regression under test.
//
// Two arms, because they fail differently:
//   - already-cancelled: PrefillLast must return ctx.Err() and NOT the logits. Catches a missing
//     check outright.
//   - cancelled mid-flight: the call must return promptly rather than after the whole prompt.
//     "Promptly" is asserted against the SAME model's uncancelled cost measured in the same
//     process, not a wall-clock constant — a fixed millisecond budget would be a machine-speed
//     assertion wearing a correctness costume.
func TestPrefillCancel(t *testing.T) {
	const dir = "../testdata/gemma4-moe-scaled" // MoE: exercises the per-ROW check, the coarse case
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T)", mc.ResidentForwardForTest())
	}
	if batched, why := rf.PrefillPath(); !batched {
		t.Skipf("batched prefill declined: %s", why)
	}

	const M = 512
	embs := make([][]float32, M)
	var s uint32 = 20260905
	for i := range embs {
		row := make([]float32, rf.hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}

	// Baseline: what this prompt costs when nobody cancels. The comparison, not a constant.
	rf.Reset()
	t0 := time.Now()
	if _, e := rf.PrefillLast(context.Background(), embs, 0); e != nil {
		t.Fatalf("uncancelled PrefillLast: %v", e)
	}
	full := time.Since(t0)
	t.Logf("uncancelled prefill of %d rows: %s", M, full.Round(time.Millisecond))

	t.Run("already-cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		rf.Reset()
		lg, e := rf.PrefillLast(ctx, embs, 0)
		if e == nil {
			t.Fatalf("a cancelled prefill returned %d logits instead of an error — the whole prompt "+
				"ran on a request nobody was waiting for", len(lg))
		}
		if !errors.Is(e, context.Canceled) {
			t.Errorf("want context.Canceled, got %v", e)
		}
	})

	t.Run("cancelled-mid-flight", func(t *testing.T) {
		// Cancel a fifth of the way in. With the check in place the call unwinds at the next row or
		// chunk boundary; without it, it runs to `full` regardless.
		ctx, cancel := context.WithCancel(context.Background())
		time.AfterFunc(full/5, cancel)
		rf.Reset()
		t1 := time.Now()
		_, e := rf.PrefillLast(ctx, embs, 0)
		took := time.Since(t1)
		t.Logf("cancelled at ~%s, returned after %s (uncancelled: %s)",
			(full / 5).Round(time.Millisecond), took.Round(time.Millisecond), full.Round(time.Millisecond))
		if e == nil {
			t.Fatal("mid-flight cancel was ignored — PrefillLast ran the prompt to completion")
		}
		if !errors.Is(e, context.Canceled) {
			t.Errorf("want context.Canceled, got %v", e)
		}
		// Generous: half the uncancelled cost. The point is that it does not run to completion, not
		// that it unwinds within some particular number of milliseconds.
		if took > full/2 {
			t.Errorf("cancelled prefill took %s against an uncancelled %s — it is not stopping at a "+
				"row or chunk boundary, so the cancellation check is too coarse to matter", took, full)
		}
	})
}

// TestPrefillCancel_dense covers what TestPrefillCancel structurally cannot. That test uses a MoE
// fixture, where the per-ROW check inside the FFN loop catches a cancel before the chunk boundary
// ever matters — so deleting the chunk-boundary check left it GREEN. Measured, not assumed: the
// mutation was run and the test passed with the check removed.
//
// A dense model has no per-row loop, so the chunk boundary is the only place a cancel can be seen.
// GOINFER_PREFILL_CHUNK forces several chunks over this prompt; without it M would fit in one pass
// and the loop would be skipped entirely.
func TestPrefillCancel_dense(t *testing.T) {
	const dir = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	t.Setenv("GOINFER_PREFILL_CHUNK", "32")
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Skipf("not CUDA-resident (%T)", mc.ResidentForwardForTest())
	}
	for l := range rf.layers {
		if rf.layers[l].isMoE || rf.layers[l].g4moe {
			t.Fatal("fixture is MoE — then the per-row check covers this and the chunk boundary is " +
				"never the thing under test, which is the whole reason this test exists")
		}
	}

	const M = 512 // 16 chunks at the width set above
	embs := make([][]float32, M)
	var s uint32 = 987123
	for i := range embs {
		row := make([]float32, rf.hidden)
		for j := range row {
			s = s*1664525 + 1013904223
			row[j] = float32(int32(s>>16)%2001-1000) / 10000
		}
		embs[i] = row
	}

	// The baseline is a WARM run. The first prefill on a fresh model pays one-off costs (kernel JIT,
	// buffer first-touch), and a baseline taken from it overstates the steady-state cost several times
	// over, so a cancel scheduled at a fraction of it can fire after the run has already finished, which
	// reads as "cancel ignored" without any cancellation being wrong. So: one untimed warm-up, then the
	// fastest of three timed runs.
	prefill := func(ctx context.Context) (time.Duration, error) {
		rf.Reset()
		t0 := time.Now()
		_, e := rf.PrefillLast(ctx, embs, 0)
		return time.Since(t0), e
	}
	if _, e := prefill(context.Background()); e != nil {
		t.Fatalf("warm-up: %v", e)
	}
	full := time.Duration(1<<63 - 1)
	for range 3 {
		d, e := prefill(context.Background())
		if e != nil {
			t.Fatalf("uncancelled: %v", e)
		}
		full = min(full, d)
	}
	t.Logf("uncancelled dense prefill of %d rows at chunk 32: %s (fastest of 3 warm runs)", M, full.Round(time.Millisecond))

	// A run that completes before the cancel fires says nothing about cancellation (the machine was faster this time
	// than the baseline), so it is retried rather than judged; only a run the cancel actually landed inside is.
	var took time.Duration
	var e error
	landed := false
	for attempt := 1; attempt <= 5 && !landed; attempt++ {
		ctx, cancel := context.WithCancel(context.Background())
		timer := time.AfterFunc(full/5, cancel)
		took, e = prefill(ctx)
		timer.Stop()
		cancel()
		t.Logf("attempt %d: cancel at ~%s, returned after %s (err %v)", attempt, (full / 5).Round(time.Millisecond), took.Round(time.Millisecond), e)
		landed = e != nil || took >= full/5
	}
	if !landed {
		t.Fatalf("no attempt ran long enough for the cancel to land (baseline %s): the machine is too noisy to time this", full)
	}
	if e == nil {
		t.Fatal("dense mid-flight cancel ignored — the chunk-boundary check is not firing")
	}
	if !errors.Is(e, context.Canceled) {
		t.Errorf("want context.Canceled, got %v", e)
	}
	if took > full/2 {
		t.Errorf("cancelled dense prefill took %s against %s uncancelled — not stopping at a chunk boundary", took, full)
	}

	// The SINGLE-PASS branch. A prompt at or under the chunk width skips the loop entirely, so
	// neither the chunk-boundary check nor (on a dense model) any per-row check can see a cancel —
	// only the check at function entry can. Removing that entry check leaves every other case in this
	// file green.
	t.Run("single-pass", func(t *testing.T) {
		decoder.SetKnobEnvForTest(t, mc, "GOINFER_PREFILL_CHUNK", "4096") // wider than M ⇒ one pass, loop skipped
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		rf.Reset()
		lg, e := rf.PrefillLast(ctx, embs, 0)
		if e == nil {
			t.Fatalf("a cancelled single-pass prefill returned %d logits — a prompt that fits in one "+
				"chunk is uncancellable without the entry check", len(lg))
		}
		if !errors.Is(e, context.Canceled) {
			t.Errorf("want context.Canceled, got %v", e)
		}
	})
}
