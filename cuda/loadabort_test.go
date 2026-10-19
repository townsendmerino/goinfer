//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The swap tripwire's consumer in the CUDA build (B33): Options.LoadAbort, reached through decoder.Model.LoadAbort, stops the build during planning,
// before the layer upload, and in the middle of a streamed upload, and the stop is the load's error (decoder.ErrLoadAborted), not a decline that
// carries on on the CPU.
//
//	go test -count=1 -timeout 10m -tags 'cuda goinfer_testhooks' -run '^TestLoadAbort_' -v ./cuda/

func abortedChan() chan struct{} {
	c := make(chan struct{})
	close(c)
	return c
}

func TestLoadAbort_theBuildStopsWhereItPollsAndNowhereElse(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	load := func(abort <-chan struct{}) (*decoder.Model, error) {
		return decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: 1, ResidentContext: 256, LoadAbort: abort})
	}
	defer func() { planCompleteForTest, allocSeenForTest = nil, nil }()

	// The control: an open channel builds a resident. Without it the aborts below could be a build that never worked.
	m, err := load(make(chan struct{}))
	if err != nil {
		t.Fatalf("VOID: the control load failed: %v", err)
	}
	_ = mustResident(t, m)
	_ = m.Close()

	// Closed before the build: stopped while planning, before one device buffer is taken.
	allocs := 0
	allocSeenForTest = func(int) { allocs++ }
	if m, err = load(abortedChan()); m != nil || !errors.Is(err, decoder.ErrLoadAborted) {
		t.Errorf("closed before the build: model %v err %v; want no model and ErrLoadAborted", m, err)
	}
	if allocs != 0 {
		t.Errorf("an abort during planning allocated %d device buffers", allocs)
	}
	allocSeenForTest = nil

	// Closed the moment planning completes: stopped before the first layer is uploaded. The streamed upload's own poll would also end the build,
	// but only after layer 0's small tensors were allocated; the layer loop's poll stops it at the scratch buffers the job takes before the
	// loop, so the number of buffers allocated after the abort is what tells the two polls apart.
	ch := make(chan struct{})
	var atPlan int
	allocs = 0
	allocSeenForTest = func(int) { allocs++ }
	planCompleteForTest = func() { atPlan = allocs; close(ch) }
	if m, err = load(ch); m != nil || !errors.Is(err, decoder.ErrLoadAborted) {
		t.Errorf("closed after planning: model %v err %v; want no model and ErrLoadAborted", m, err)
	}
	if after := allocs - atPlan; after > 8 {
		t.Errorf("%d device buffers were allocated after the abort closed at the end of planning; the layer loop should stop at its first poll, before any layer buffer", after)
	}
	planCompleteForTest, allocSeenForTest = nil, nil

	// The error is the load's own, not a CPU decline: a load whose build hit an ordinary failure is a different test (the decoder's control).
	if err != nil && errors.Is(err, errKVWontFit) {
		t.Errorf("an abort is not a KV-does-not-fit error: %v", err)
	}
}

// A streamed upload stops between blocks: the block it holds is not uploaded, the staging pool is intact afterwards (nothing leaked, nothing deadlocked),
// and the stop is recorded as the build's error.
func TestLoadAbort_aStreamedUploadStopsBetweenBlocksAndLeavesThePoolWhole(t *testing.T) {
	r := streamRig(t)
	withStage(t, 1<<12) // small blocks: several per tensor
	wm := streamWeights()["int4"]
	plan, err := planWeight(&wm)
	if err != nil {
		t.Fatal(err)
	}

	r.loadAbort = abortedChan()
	_ = r.upW(plan)
	if !errors.Is(r.setupErr, decoder.ErrLoadAborted) {
		t.Errorf("setupErr %v after a streamed upload with the abort closed; want ErrLoadAborted", r.setupErr)
	}
	if got, want := len(r.pk.free), len(r.pk.stages); got != want {
		t.Errorf("%d of %d staging blocks free after an aborted upload: the pool leaked a block", got, want)
	}

	r.loadAbort, r.setupErr = nil, nil
	_ = r.upW(plan) // the same pool, reused: a deadlocked dispatcher would hang here
	if r.setupErr != nil {
		t.Errorf("a normal upload after an aborted one failed: %v", r.setupErr)
	}
}
