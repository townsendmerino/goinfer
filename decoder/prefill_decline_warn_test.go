package decoder

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// B20 (docs/queue-engineering.md) gate: residentPrefillSeed's decline must actually reach an operator-visible log naming
// the reason, not merely be read internally. warnPrefillDeclined logs it, and these tests assert the log line's CONTENT:
// an unasserted log line is a coverage claim, not coverage (CLAUDE.md § Tests).

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
		vocab:  vocab,
		reason: "fake: layer 3 needs 52000 bytes of shared memory, device limit 49152",
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

	// Generate MUST be started INSIDE the capture, not before it. captureStderr installs its pipe by assigning the os.Stderr
	// global and warnPrefillDeclined reads that global from the generation goroutine, so starting the goroutine first is a data
	// race on os.Stderr. It is also a correctness bug: the decline is logged during PREFILL, the goroutine's first act, so one
	// emitted before the swap lands goes to the real stderr and the test asserts on output it never captured.
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

// TestWarnPrefillDeclined_FiresOncePerReason pins the dedup key: the decline's normalized reason, not a single
// process-lifetime gate. A bare sync.Once let the first short prompt permanently silence a later, genuinely different
// decline (a resident-cap refusal, an OOM) an operator would want to see; on Metal nearly every prompt under the
// fast-prefill floor declines with a message that differs only in its promptLen. Numbers are normalized out of the key so
// same-shape declines with different byte/token counts collapse to one line, while a differently-worded reason gets its
// own. (N-35, audit-metal-2026-09-12.md.)
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
