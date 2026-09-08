package decoder

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
)

// The G4 seam gate (docs/task-gpu-paths-2026-09.md).
//
// WHY THIS EXISTS. Same shape as resident_seam_test.go's TestSeam_GenerateRunsOnTheResident: a
// resident capability that nothing ever calls is indistinguishable from one that was never
// wired. These tests fake the resident backend and use the committed tiny fixture, so the G4
// wiring in decoder/embed.go (HiddenLast trying ResidentHiddenLast before falling to the CPU
// path) is gated on every push with no GPU and no downloaded model — real per-backend numerics
// (the cosine ≥ 0.9999 bar) are a separate, backend-specific gate this does not replace.

// fakeResidentHiddenLast extends fakeResident with ResidentHiddenLast, recording whether and how
// it was called, and optionally forcing a decline to exercise the CPU fallback.
type fakeResidentHiddenLast struct {
	*fakeResident
	hiddenDim int
	fail      bool

	calls        int32
	lastStartPos int
	lastLen      int
}

func (f *fakeResidentHiddenLast) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	atomic.AddInt32(&f.calls, 1)
	f.lastStartPos = startPos
	f.lastLen = len(embeddings)
	if f.fail {
		return nil, fmt.Errorf("fakeResidentHiddenLast: forced decline")
	}
	out := make([]float32, f.hiddenDim)
	out[0] = 1 // a sentinel distinguishing this from any real CPU-computed vector
	return out, nil
}

type fakeHiddenLastBackend struct {
	Backend
	rf *fakeResidentHiddenLast
}

func (b *fakeHiddenLastBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	hidden, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeResidentHiddenLast{fakeResident: &fakeResident{vocab: vocab}, hiddenDim: hidden}
	return b.rf, true, nil
}

func (b *fakeHiddenLastBackend) Close() error { return nil }

func loadWithFakeHiddenLastResident(t *testing.T) (*Model, *fakeHiddenLastBackend) {
	t.Helper()
	be := &fakeHiddenLastBackend{}
	name := fmt.Sprintf("fake-hiddenlast-resident-%s", t.Name())
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
		t.Fatalf("Load with fake resident HiddenLast backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, be
}

// TestSeam_HiddenLastUsesResidentWhenAvailable is the gate for G4 itself: a resident model whose
// backend implements ResidentHiddenLast must actually be asked for the vector, not silently run
// the CPU path with a resident backend sitting idle (the exact class of bug
// TestSeam_GenerateRunsOnTheResident exists to catch for Generate).
func TestSeam_HiddenLastUsesResidentWhenAvailable(t *testing.T) {
	m, be := loadWithFakeHiddenLastResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}
	if _, own := m.w.arch.ownForward(); own {
		t.Skip("fixture arch has its own runLayers; HiddenLast declines it before touching residency")
	}
	out, err := m.HiddenLast([]int{1, 2, 3})
	if err != nil {
		t.Fatalf("HiddenLast: %v", err)
	}
	if be.rf.calls != 1 {
		t.Fatalf("HiddenLast produced a result WITHOUT calling the resident runner (%d calls) — "+
			"the model is resident but embedding silently ran on the CPU, the G4 defect", be.rf.calls)
	}
	if be.rf.lastStartPos != 0 {
		t.Errorf("resident HiddenLast called with startPos=%d, want 0 (no prefix reuse for embeddings)", be.rf.lastStartPos)
	}
	if be.rf.lastLen != 3 {
		t.Errorf("resident HiddenLast called with %d embeddings, want 3 (len(ids))", be.rf.lastLen)
	}
	if len(out) != be.rf.hiddenDim || out[0] != 1 {
		t.Fatalf("HiddenLast returned %v, want the resident fake's sentinel output (len %d, [0]=1) — "+
			"the CPU path answered instead of the resident one", out, be.rf.hiddenDim)
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after HiddenLast, want nil — a HiddenLast prefill has nothing durable "+
			"worth remembering for the next Generate call and must leave the resident KV's contents unknown", m.resIDs)
	}
}

// TestSeam_HiddenLastFallsBackToCPUWhenResidentBusy pins the resBusy-loser behaviour: a
// generation already in flight on this Model must not be interrupted or corrupted by a
// concurrent HiddenLast call, and that call must still get a correct (CPU) answer rather than
// failing outright — the same graceful-decline contract Generate's resBusy loser already has.
func TestSeam_HiddenLastFallsBackToCPUWhenResidentBusy(t *testing.T) {
	m, be := loadWithFakeHiddenLastResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	if _, own := m.w.arch.ownForward(); own {
		t.Skip("fixture arch has its own runLayers; HiddenLast declines it before touching residency")
	}
	ids := []int{1, 2, 3}
	want, err := m.hiddenLastBatched(ids)
	if err != nil {
		if want, err = m.hiddenLastSequential(ids); err != nil {
			t.Fatalf("computing the CPU reference vector: %v", err)
		}
	}

	if !atomic.CompareAndSwapInt32(&m.resBusy, 0, 1) {
		t.Fatal("could not claim resBusy to simulate an in-flight generation")
	}
	defer atomic.StoreInt32(&m.resBusy, 0)

	out, err := m.HiddenLast(ids)
	if err != nil {
		t.Fatalf("HiddenLast while resBusy: %v", err)
	}
	if be.rf.calls != 0 {
		t.Fatalf("resident HiddenLast was called (%d times) while resBusy was already claimed by "+
			"another generation — this can corrupt the in-flight generation's resident KV", be.rf.calls)
	}
	if len(out) != len(want) {
		t.Fatalf("HiddenLast while busy returned %d dims, want %d (the CPU reference)", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("HiddenLast while busy did not match the CPU reference at [%d]: got %v, want %v", i, out[i], want[i])
		}
	}
	if atomic.LoadInt32(&m.resBusy) != 1 {
		t.Error("resBusy was cleared by the HiddenLast call that lost the CAS — it must leave the winner's claim untouched")
	}
}

// TestSeam_HiddenLastFallsBackToCPUOnResidentDecline exercises the other half of "declines
// gracefully": the resident backend attempted the call and refused (OOM, cap, whatever), and the
// request must still succeed via the CPU path rather than surfacing the resident failure to the
// caller — exactly the posture every other resident decline in this codebase takes.
func TestSeam_HiddenLastFallsBackToCPUOnResidentDecline(t *testing.T) {
	m, be := loadWithFakeHiddenLastResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	if _, own := m.w.arch.ownForward(); own {
		t.Skip("fixture arch has its own runLayers; HiddenLast declines it before touching residency")
	}
	be.rf.fail = true

	ids := []int{1, 2, 3}
	want, err := m.hiddenLastBatched(ids)
	if err != nil {
		if want, err = m.hiddenLastSequential(ids); err != nil {
			t.Fatalf("computing the CPU reference vector: %v", err)
		}
	}

	out, err := m.HiddenLast(ids)
	if err != nil {
		t.Fatalf("HiddenLast did not fall back to the CPU path on a resident decline: %v", err)
	}
	if be.rf.calls != 1 {
		t.Fatalf("resident HiddenLast was called %d times, want exactly 1 (attempted, then declined)", be.rf.calls)
	}
	if len(out) != len(want) {
		t.Fatalf("fallback returned %d dims, want %d (the CPU reference)", len(out), len(want))
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("fallback did not match the CPU reference at [%d]: got %v, want %v", i, out[i], want[i])
		}
	}
	if atomic.LoadInt32(&m.resBusy) != 0 {
		t.Error("resBusy was left claimed after a resident decline — the next generation would see the model permanently busy")
	}
}
