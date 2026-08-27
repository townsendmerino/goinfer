package decoder

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A long test must be legible WHILE it runs: at any moment the operator should be able to tell
// whether it is stuck and roughly how much is left. t.Logf cannot do that — it is buffered until
// the test returns, and dropped entirely for a passing test without -v. That is not a style
// preference; the tests wired to this helper each ran for 2–15 minutes emitting nothing at all,
// which is indistinguishable from a hang until the whole suite ends.
//
// So: os.Stderr, unbuffered, independent of -v, on a TIME ticker rather than an iteration count
// (an iteration count picked for one machine goes silent on a slower one, which is exactly when
// the heartbeat matters most). TEST_PROGRESS_INTERVAL overrides the cadence; "0" silences it.
//
// RUN THESE UNDER -v. `go test` buffers a package's output and DISCARDS it entirely when the
// package passes, so without -v these lines never appear — measured, not assumed: a passing run of
// TestA3FastAttentionDivergence emitted a 55-byte log holding only the "ok" line. os.Stderr does
// not dodge that; what it buys over t.Logf is that under -v the lines stream AS THEY HAPPEN
// (verified: 5s apart by wall clock mid-test) instead of arriving in one dump when the test ends.
const defaultProgressInterval = 45 * time.Second

func progressInterval() time.Duration {
	v := os.Getenv("TEST_PROGRESS_INTERVAL")
	if v == "" {
		return defaultProgressInterval
	}
	if d, err := time.ParseDuration(v); err == nil {
		return d
	}
	if n, err := strconv.Atoi(v); err == nil { // bare seconds, the likelier thing to type
		return time.Duration(n) * time.Second
	}
	return defaultProgressInterval
}

// deadlineCtx derives a context from the test binary's OWN -timeout, expiring 30s early so that a
// hung call fails here — naming this test and the phase it was in — instead of surfacing 30s later
// as a package-wide timeout panic whose goroutine dump names no test at all. Falls back to a plain
// cancellable context when the runner was given no timeout.
func deadlineCtx(t *testing.T) context.Context {
	t.Helper()
	parent := context.Background()
	if d, ok := t.Deadline(); ok {
		ctx, cancel := context.WithDeadline(parent, d.Add(-30*time.Second))
		t.Cleanup(cancel)
		return ctx
	}
	ctx, cancel := context.WithCancel(parent)
	t.Cleanup(cancel)
	return ctx
}

// progress is a heartbeat for a test expected to exceed ~2 minutes. The zero value is not usable;
// call newProgress. Every method is safe from multiple goroutines, because several of these tests
// fan work out across workers.
type progress struct {
	label  string
	total  int64
	uneven bool // items differ in cost, so a linear ETA would lie
	start  time.Time
	done   atomic.Int64
	phase  atomic.Value // string
	stop   chan struct{}
	once   sync.Once
	wg     sync.WaitGroup
}

// newProgress starts the heartbeat and registers its own shutdown, so a test that fails early
// still stops the ticker. total may be 0 when the work is not countable up front: the heartbeat
// then reports phase and elapsed rather than inventing an ETA it cannot support.
func newProgress(t *testing.T, label string, total int) *progress {
	t.Helper()
	p := &progress{label: label, total: int64(total), start: time.Now(), stop: make(chan struct{})}
	p.phase.Store("starting")
	t.Cleanup(p.Done)
	iv := progressInterval()
	if iv <= 0 {
		return p // silenced — the mutators below stay valid, they just never print
	}
	fmt.Fprintf(os.Stderr, "[%s] start%s\n", p.label, p.totalSuffix())
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		tk := time.NewTicker(iv)
		defer tk.Stop()
		for {
			select {
			case <-p.stop:
				return
			case <-tk.C:
				p.emit("")
			}
		}
	}()
	return p
}

func (p *progress) totalSuffix() string {
	if p.total > 0 {
		return fmt.Sprintf(" — %d items", p.total)
	}
	return ""
}

// Uneven marks the items as differing in cost, suppressing the ETA while keeping the n/total
// count. Use it whenever the loop sweeps a size (attention over growing K, say), where a linear
// projection from the cheap early cases is confidently wrong.
func (p *progress) Uneven() *progress { p.uneven = true; return p }

// Step records n completed items.
func (p *progress) Step(n int) { p.done.Add(int64(n)) }

// Phase names the stage now running and prints immediately, so entering a phase is visible even
// if the phase itself is shorter than the tick interval.
func (p *progress) Phase(name string) {
	p.phase.Store(name)
	if progressInterval() > 0 {
		p.emit("")
	}
}

// Done stops the heartbeat and prints a final line with the total elapsed time. Idempotent.
func (p *progress) Done() {
	p.once.Do(func() {
		close(p.stop)
		p.wg.Wait()
		if progressInterval() > 0 {
			p.emit("done")
		}
	})
}

func (p *progress) emit(tag string) {
	el := time.Since(p.start).Round(time.Second)
	ph, _ := p.phase.Load().(string)
	if tag != "" {
		ph = tag
	}
	line := fmt.Sprintf("[%s] %s elapsed=%s", p.label, ph, el)
	if d := p.done.Load(); p.total > 0 {
		line += fmt.Sprintf(" %d/%d", d, p.total)
		if d > 0 && d < p.total && !p.uneven {
			// Linear extrapolation, valid only where items cost the same. Uneven() turns it off
			// rather than printing an ETA that climbs every tick and teaches nobody anything.
			eta := time.Duration(float64(el) / float64(d) * float64(p.total-d)).Round(time.Second)
			line += fmt.Sprintf(" eta=%s", eta)
		}
	} else if d := p.done.Load(); d > 0 {
		line += fmt.Sprintf(" %d done", d)
	}
	fmt.Fprintln(os.Stderr, line)
}
