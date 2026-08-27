package decoder

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
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

	// Guards the sliding-rate window. emit() runs from the ticker goroutine AND from Phase() on
	// the test's own goroutine, so these are shared state, not ticker-local.
	mu         sync.Mutex
	lastAt     time.Time
	lastDone   int64
	lastIO     int64
	lastIOAt   time.Time // SEPARATE from lastAt on purpose — see ioProgress
	lastPerMin float64   // most recent items/minute, so eta and rate agree
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
	fmt.Fprintf(os.Stderr, "%s [%s] start%s\n", time.Now().Format("15:04:05"), p.label, p.totalSuffix())
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
	now := time.Now()
	el := now.Sub(p.start).Round(time.Second)
	ph, _ := p.phase.Load().(string)
	if tag != "" {
		ph = tag
	}
	// Absolute time first: an archived log read a day later has to answer WHEN a run stalled, so
	// it can be lined up against dmesg, a thermal event, or the other machine's log. Elapsed
	// alone cannot do that, and these logs are kept as evidence.
	line := fmt.Sprintf("%s [%s] %s elapsed=%s", now.Format("15:04:05"), p.label, ph, el)
	if d := p.done.Load(); p.total > 0 {
		line += fmt.Sprintf(" %d/%d", d, p.total)
		if d > 0 && d < p.total && !p.uneven {
			// From the RECENT rate, not from total elapsed over total done. A test whose counted
			// work follows a long uncounted phase — an 80B checkpoint load, say — otherwise
			// divides 14 minutes by one finished item and reports eta=2h20m for work that took
			// six. Measured on exactly that run. Uneven() still turns it off entirely where the
			// items differ in cost and no window makes the projection honest.
			if perMin, ok := p.peekRatePerMin(); ok && perMin > 0 {
				eta := time.Duration(float64(p.total-d) / perMin * float64(time.Minute)).Round(time.Second)
				line += fmt.Sprintf(" eta=%s", eta)
			}
		}
	} else if d := p.done.Load(); d > 0 {
		line += fmt.Sprintf(" %d done", d)
	}
	// Rate over the LAST interval, not the whole run. A cumulative average is still digesting
	// cold-cache page-ins minutes in -- measured here: mellum2's cumulative eta read 31m, 25m,
	// 22m, 20m on successive ticks while the machine had not actually changed speed that much. A
	// recent rate tracks what it is doing NOW, and stays honest on uneven work where an ETA cannot.
	if r, ok := p.rate(now); ok {
		line += fmt.Sprintf(" rate=%s", r)
	}
	// Bytes read by this process. A phase with nothing to count -- loading a 162GB checkpoint, say
	// -- otherwise produces a heartbeat that proves only that the process is ALIVE, not that it is
	// getting anywhere, and those are the two states the reader needs to tell apart. This is the
	// exact signal that had to be dug out of /proc/PID/io by hand while the qwen3next oracle sat
	// silent for ten minutes. Linux-only; absent elsewhere, and simply omitted there.
	if io, rate, ok := p.ioProgress(now); ok {
		line += fmt.Sprintf(" io=%s", io)
		if rate != "" {
			line += fmt.Sprintf(" (%s)", rate)
		}
	}
	fmt.Fprintln(os.Stderr, line)
}

// rate returns items/minute since the previous emit, and false on the first one (no window yet).
func (p *progress) rate(now time.Time) (string, bool) {
	perMin, ok := p.ratePerMin(now)
	if !ok {
		return "", false
	}
	if perMin >= 10 {
		return fmt.Sprintf("%.0f/min", perMin), true
	}
	return fmt.Sprintf("%.1f/min", perMin), true
}

func (p *progress) ratePerMin(now time.Time) (float64, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	d := p.done.Load()
	prevAt, prevDone := p.lastAt, p.lastDone
	p.lastAt, p.lastDone = now, d
	if prevAt.IsZero() || d <= prevDone {
		return 0, false
	}
	dt := now.Sub(prevAt).Seconds()
	if dt <= 0 {
		return 0, false
	}
	// Already holding p.mu — do NOT relock here.
	p.lastPerMin = float64(d-prevDone) / dt * 60
	return p.lastPerMin, true
}

// procReadBytes reports bytes this process has read from storage. Linux-only: /proc/self/io does
// not exist on darwin, and the caller simply omits the field there rather than faking one.
func procReadBytes() (int64, bool) {
	b, err := os.ReadFile("/proc/self/io")
	if err != nil {
		return 0, false
	}
	for _, ln := range strings.Split(string(b), "\n") {
		rest, ok := strings.CutPrefix(ln, "read_bytes:")
		if !ok {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSpace(rest), 10, 64)
		if err != nil {
			return 0, false
		}
		return n, true
	}
	return 0, false
}

// ioProgress returns cumulative bytes read and the rate since the previous emit.
func (p *progress) ioProgress(now time.Time) (total, rate string, ok bool) {
	cur, ok := procReadBytes()
	if !ok {
		return "", "", false
	}
	return p.ioProgressFrom(now, cur)
}

// ioProgressFrom is the rate arithmetic, split from the /proc read so it can be driven with known
// values on any platform. The bug it now covers only showed on a real 162GB load.
func (p *progress) ioProgressFrom(now time.Time, cur int64) (total, rate string, ok bool) {
	// Its own timestamp, NOT lastAt. emit() calls rate() first, which sets lastAt = now, so
	// reusing it here made dt zero every time and the io rate never printed once — visible only
	// on a real 162GB load, where the field showed a total and never a rate.
	p.mu.Lock()
	prev, prevAt := p.lastIO, p.lastIOAt
	p.lastIO, p.lastIOAt = cur, now
	p.mu.Unlock()
	total = humanBytes(cur)
	if prev > 0 && cur > prev && !prevAt.IsZero() {
		if dt := now.Sub(prevAt).Seconds(); dt > 0 {
			rate = humanBytes(int64(float64(cur-prev)/dt)) + "/s"
		}
	}
	return total, rate, true
}

func humanBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0fMB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%dKB", n/1024)
	}
}

// peekRatePerMin reports the last computed items/minute WITHOUT advancing the window, so the ETA
// and the rate on one line always describe the same interval.
func (p *progress) peekRatePerMin() (float64, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.lastPerMin, p.lastPerMin > 0
}
