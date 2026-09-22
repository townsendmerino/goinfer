package decoder

import (
	"errors"
	"runtime"
	"sync"
	"testing"
)

// task-never-swap-2026-09.md S3: the load-time consumer checks an abort channel
// BETWEEN layers of the resident GGUF build (parallelLayers is the single
// dispatch chokepoint every family's loader goes through — see buildWeightsFromGGUF).
// These gates test parallelLayers directly, independent of any real GGUF, since the
// abort/priority logic lives entirely in its dispatch loop.

func TestParallelLayers_nilAbortIsNoop(t *testing.T) {
	var mu sync.Mutex
	ran := map[int]bool{}
	const n = 5
	err := parallelLayers(n, nil, func(i int) error {
		mu.Lock()
		ran[i] = true
		mu.Unlock()
		return nil
	})
	if err != nil {
		t.Fatalf("nil abort: got error %v, want nil", err)
	}
	if len(ran) != n {
		t.Fatalf("nil abort: %d of %d layers ran, want all %d", len(ran), n, n)
	}
}

// TestParallelLayers_preClosedAbort_runsNothing covers both dispatch shapes: n==1
// (the fast path inside parallelLayers) and n>1 (the worker-pool path). A channel
// closed BEFORE the call is always observed on the very first grab (reading a
// closed channel never blocks), so this is deterministic — no layer may run.
func TestParallelLayers_preClosedAbort_runsNothing(t *testing.T) {
	for _, n := range []int{1, 4} {
		abort := make(chan struct{})
		close(abort)
		var mu sync.Mutex
		ran := map[int]bool{}
		err := parallelLayers(n, abort, func(i int) error {
			mu.Lock()
			ran[i] = true
			mu.Unlock()
			return nil
		})
		if !errors.Is(err, errLoadAborted) {
			t.Errorf("n=%d: got error %v, want errLoadAborted", n, err)
		}
		if len(ran) != 0 {
			t.Errorf("n=%d: %d layers ran on a pre-closed abort, want 0 (ran=%v)", n, len(ran), ran)
		}
	}
}

// TestParallelLayers_midBuildAbort_inFlightFinishesNoNewStarts pins the actual
// contract: a layer already grabbed when abort fires is allowed to finish: only
// NEW grabs are refused. Forced to a single worker (GOMAXPROCS(1)) so the
// interleaving is deterministic — with one worker there is exactly one grab in
// flight at a time, so "abort closes while layer 0 is building" unambiguously
// means "layer 0 finishes, layer 1 never starts."
func TestParallelLayers_midBuildAbort_inFlightFinishesNoNewStarts(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	abort := make(chan struct{})
	started := make(chan struct{})
	proceed := make(chan struct{})
	var mu sync.Mutex
	ran := []int{}

	done := make(chan error, 1)
	go func() {
		done <- parallelLayers(3, abort, func(i int) error {
			mu.Lock()
			ran = append(ran, i)
			mu.Unlock()
			if i == 0 {
				close(started)
				<-proceed
			}
			return nil
		})
	}()

	<-started      // layer 0 has been grabbed and is mid-build
	close(abort)   // trip the tripwire while it is in flight
	close(proceed) // let layer 0 finish

	err := <-done
	if !errors.Is(err, errLoadAborted) {
		t.Fatalf("got error %v, want errLoadAborted", err)
	}
	if want := []int{0}; len(ran) != len(want) || ran[0] != want[0] {
		t.Fatalf("ran layers %v, want exactly %v — abort must let an in-flight layer finish but start no new one", ran, want)
	}
}

// TestParallelLayers_realErrorWinsOverConcurrentAbort pins the priority the
// comment in parallelLayers documents: "a real build error always wins over a
// concurrent abort." Two workers are both occupied (so a third layer's grab
// cannot happen until one frees up), abort is closed while both are still busy,
// then both are released together — one returning a real error, the other nil.
// By construction layer 2's grab can only happen after abort is already closed,
// so it can never run; and firstErr, once set, always wins the final race against
// aborted in parallelLayers' own return order. The only truly deterministic
// observable here is the end result, so that is what this test checks.
func TestParallelLayers_realErrorWinsOverConcurrentAbort(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))

	errBoom := errors.New("boom")
	abort := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(2)
	barrier := make(chan struct{})
	var mu sync.Mutex
	ran := map[int]bool{}

	done := make(chan error, 1)
	go func() {
		done <- parallelLayers(3, abort, func(i int) error {
			mu.Lock()
			ran[i] = true
			mu.Unlock()
			switch i {
			case 0, 1:
				wg.Done()
				<-barrier
				if i == 0 {
					return errBoom
				}
				return nil
			default:
				t.Errorf("layer %d ran: with 2 workers both held on 0/1 until abort closed, layer 2's grab could not have succeeded", i)
				return nil
			}
		})
	}()

	wg.Wait()    // both workers are occupied on layers 0 and 1
	close(abort) // trip the tripwire while no worker can yet observe it
	close(barrier)

	err := <-done
	if !errors.Is(err, errBoom) {
		t.Fatalf("got error %v, want the real build error (errBoom), not errLoadAborted", err)
	}
	if ran[2] {
		t.Fatalf("layer 2 ran: %v", ran)
	}
}

// TestLoadGGUFBytes_abortPropagatesErrLoadAborted is the end-to-end plumbing
// check: Options.LoadAbort -> LoadGGUFBytes -> buildGGUFWeights ->
// buildWeightsFromGGUF -> parallelLayers, through the real public entry point a
// caller (internal/serveapp's future swap-guard consumer) will actually use. A
// pre-closed channel keeps this deterministic without needing to race a real
// swap trip against a real build.
func TestLoadGGUFBytes_abortPropagatesErrLoadAborted(t *testing.T) {
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	abort := make(chan struct{})
	close(abort)
	_, err := LoadGGUFBytes(raw, Options{LoadAbort: abort})
	if !errors.Is(err, ErrLoadAborted) {
		t.Fatalf("got error %v, want errors.Is(err, ErrLoadAborted)", err)
	}
}
