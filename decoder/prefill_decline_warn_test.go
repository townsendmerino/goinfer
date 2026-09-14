package decoder

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// B20 (docs/queue-engineering.md) gate: residentPrefillSeed's decline must actually reach an
// operator-visible log naming the reason, not merely be read internally.
//
// WHY THIS EXISTS. V-05 (docs/review-2026-09-04.md) fixed a SILENT batched-prefill decline by
// making the launch refuse loudly (checkPrefillShmem et al.) instead of crashing, but B20 found
// the caller one level up, residentPrefillSeed, still threw that refusal away after checking it
// only for cancellation — the reason was reachable and correct but never reached an operator.
// 5cc4854 (2026-09-04, later the same day as the review) wired warnPrefillDeclined into the
// fallback, but nothing asserted on the log line's CONTENT — a doc comment claiming coverage is
// not coverage (CLAUDE.md's own rule) applies just as much to an unasserted log line as to an
// unasserted test body.

// decliningPrefiller is a fakeResident whose Prefiller always declines with a fixed reason,
// forcing residentPrefillSeed onto the sequential per-token fallback every time.
type decliningPrefiller struct {
	fakeResident
	reason string
	calls  int
}

func (d *decliningPrefiller) PrefillLast(_ context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	d.calls++
	return nil, fmt.Errorf("%s", d.reason)
}

type decliningPrefillerBackend struct {
	Backend
	rf *decliningPrefiller
}

func (b *decliningPrefillerBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &decliningPrefiller{
		fakeResident: fakeResident{vocab: vocab},
		reason:       "fake: layer 3 needs 52000 bytes of shared memory, device limit 49152",
	}
	return b.rf, true, nil
}

func (b *decliningPrefillerBackend) Close() error { return nil }

// TestResidentPrefillSeed_DeclineIsLoggedWithReason is the B20 gate: a batched-prefill decline
// must produce a stderr line naming the actual reason, and the request must still complete via
// the sequential fallback rather than failing outright.
func TestResidentPrefillSeed_DeclineIsLoggedWithReason(t *testing.T) {
	resetPrefillDeclineDedup() // start this test with no reason yet seen

	be := &decliningPrefillerBackend{}
	name := "fake-declining-prefiller-" + t.Name()
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
		t.Fatalf("Load with fake declining-prefiller backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} // >= 8 tokens: GOINFER_BATCHED_PREFILL's own floor

	// Generate MUST be started INSIDE the capture, not before it. captureStderr installs its pipe
	// by assigning the os.Stderr global, and warnPrefillDeclined reads that global from the
	// generation goroutine — so starting the goroutine first is a genuine data race on os.Stderr
	// (caught by -race in CI on both linux and darwin, 2026-09-12, run 34711378811; the detector
	// fired during this test and so blamed it, while the write was this line's own).
	//
	// It was also a correctness bug independent of the detector: the decline is logged during
	// PREFILL, which is the first thing the goroutine does, so any decline emitted between
	// Generate returning and the swap landing went to the REAL stderr and was missed. The test
	// would then assert on output it never captured — a flake that looks like a missing log line.
	// Starting inside the closure makes the capture cover the whole generation, which is what the
	// assertion below already assumed.
	var g *Generation
	got := captureStderr(t, func() {
		var stream <-chan int
		stream, g = m.Generate(context.Background(), prompt, 1, SamplingParams{})
		for range stream { //nolint:revive // draining the stream is the point
		}
	})
	if err := g.Err(); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if be.rf.calls == 0 {
		t.Fatal("PrefillLast was never called — the fixture did not reach the batched-prefill floor")
	}
	if !strings.Contains(got, be.rf.reason) {
		t.Fatalf("decline was not logged with its reason: got stderr %q, want it to contain %q", got, be.rf.reason)
	}
	if be.rf.forwards == 0 {
		t.Error("sequential fallback never ran after the decline — the request should still have completed")
	}
}

// TestWarnPrefillDeclined_FiresOncePerReason gates N-35 (audit-metal-2026-09-12.md): the dedup key
// is now the decline's normalized reason, not a single process-lifetime gate. B20's original fix
// sketch wanted per-reason dedupe; the FIRST shipped version was coarser (a bare sync.Once) —
// TestWarnPrefillDeclined_FiresOncePerProcess used to pin that coarser behaviour deliberately, on
// the grounds that a future change to it should be a decision, not a silent drift. This is that
// decision: on Metal, nearly every prompt under the fast-prefill floor declines with a message
// that differs only in its promptLen, so a bare sync.Once let the very first short prompt
// permanently silence a later, genuinely different decline (a resident-cap refusal, an OOM) an
// operator would want to see. Numbers are normalized out of the key so same-shape declines with
// different byte/token counts still collapse to one line, while a differently-worded reason gets
// its own.
func TestWarnPrefillDeclined_FiresOncePerReason(t *testing.T) {
	resetPrefillDeclineDedup()
	first := fmt.Errorf("metal: prompt too short (100 tokens) for fast prefill (floor=256)")
	firstRepeat := fmt.Errorf("metal: prompt too short (200 tokens) for fast prefill (floor=256)")
	second := fmt.Errorf("metal: prompt len 9000 at startPos 0 out of resident cap 4096")
	got := captureStderr(t, func() {
		warnPrefillDeclined(100, first)
		warnPrefillDeclined(200, firstRepeat)
		warnPrefillDeclined(9000, second)
	})
	if !strings.Contains(got, first.Error()) {
		t.Fatalf("first decline reason not logged: got %q", got)
	}
	if strings.Contains(got, firstRepeat.Error()) {
		t.Fatalf("same-shape repeat (different token count) was logged again despite per-reason dedup: got %q", got)
	}
	if !strings.Contains(got, second.Error()) {
		t.Fatalf("a differently-worded later decline was silenced by the earlier one's dedup: got %q", got)
	}
}
