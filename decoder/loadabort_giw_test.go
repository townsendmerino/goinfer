package decoder

import (
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
)

// The load-time swap tripwire's two new consumers (B33): a .giw load's CRC pass and layer walk, and a backend's resident build. Each polls
// Options.LoadAbort and returns an error wrapping ErrLoadAborted; a nil channel never fires.

func closedAbort() <-chan struct{} {
	c := make(chan struct{})
	close(c)
	return c
}

func TestLoadSerialized_abortIsPolledInTheCRCPassAndBetweenLayers(t *testing.T) {
	src := alignFixtureWeights(t)
	blob, err := SerializeWeightsForTarget(src, "abort-poll", GIWTargetNone)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := loadSerializedWeightsAbortable(blob, false, nil, nil); err != nil {
		t.Fatalf("VOID: the fixture does not load with a nil abort: %v", err)
	}
	// A closed channel stops the CRC pass before it reads the payload.
	if _, err := loadSerializedWeightsAbortable(blob, false, nil, closedAbort()); !errors.Is(err, ErrLoadAborted) || !strings.Contains(err.Error(), "CRC pass") {
		t.Errorf("CRC pass with a closed abort: %v, want ErrLoadAborted naming the CRC pass", err)
	}
	// With the CRC already verified the first layer boundary stops it.
	_, err = loadSerializedWeightsAbortable(blob, true, nil, closedAbort())
	if !errors.Is(err, ErrLoadAborted) || !strings.Contains(err.Error(), "layer 0") {
		t.Errorf("layer walk with a closed abort: %v, want ErrLoadAborted at layer 0", err)
	}
	// The unguarded entry points are unchanged.
	if _, err := LoadSerializedWeights(blob); err != nil {
		t.Errorf("LoadSerializedWeights: %v", err)
	}
}

// abortingResidency is a residency backend whose build reports what Model.LoadAbort held and then fails with err.
type abortingResidency struct {
	fakeResidencyBackend
	err      error
	sawAbort atomic.Bool
	sawOpen  atomic.Bool
}

func (b *abortingResidency) BuildResident(m *Model) (ResidentForward, bool, error) {
	b.sawAbort.Store(m.LoadAbort() != nil)
	select {
	case <-m.LoadAbort():
	default:
		b.sawOpen.Store(true)
	}
	return nil, false, b.err
}

// A resident build that returns ErrLoadAborted is the load's error, not a decline that continues on the CPU; any other build error keeps the
// old behaviour (the control: a load that goes on without the resident). The build sees the channel the load was given.
func TestLoad_aResidentBuildAbortIsTheLoadsError(t *testing.T) {
	load := func(name string, err error, abort <-chan struct{}) (*Model, *abortingResidency, error) {
		be := &abortingResidency{err: err}
		RegisterBackend(name, func() (Backend, error) {
			cpu, cerr := NewBackend("cpu")
			if cerr != nil {
				return nil, cerr
			}
			be.Backend = cpu
			return be, nil
		})
		m, lerr := Load(tinyFixture(t), Options{Backend: name, LoadAbort: abort})
		if m != nil {
			t.Cleanup(func() { _ = m.Close() })
		}
		return m, be, lerr
	}

	m, be, err := load("fake-abort-control", fmt.Errorf("cuda: some other build failure"), make(chan struct{}))
	if err != nil || m == nil {
		t.Fatalf("control: an ordinary build failure must still fall back to the CPU, got model %v err %v", m, err)
	}
	if !be.sawAbort.Load() || !be.sawOpen.Load() {
		t.Errorf("the build saw LoadAbort non-nil=%v and still open=%v; want the load's open channel", be.sawAbort.Load(), be.sawOpen.Load())
	}

	m, _, err = load("fake-abort", fmt.Errorf("cuda: build aborted at layer 7: %w", ErrLoadAborted), closedAbort())
	if m != nil || !errors.Is(err, ErrLoadAborted) {
		t.Errorf("an aborted resident build: model %v, err %v; want no model and an error wrapping ErrLoadAborted", m, err)
	}

	var unguarded *Model
	if unguarded = (&Model{}); unguarded.LoadAbort() != nil {
		t.Errorf("a model of an unguarded load reports a non-nil LoadAbort")
	}
}
