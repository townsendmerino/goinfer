package jobs

import (
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// VendorOptions is a request body's "goinfer" object: goinfer's own request options, ignored by every other server
// (SB11b, docs/server.md "Per-turn timings and cost"; experimental until v1.0).
type VendorOptions struct {
	// Cost asks for the turn's cost report (internal/cost) as the response's "cost".
	Cost bool `json:"cost"`
}

func (v *VendorOptions) WantsCost() bool { return v != nil && v.Cost }

// TurnClock is one turn's stage boundaries, the only places X1's timings are read (docs/completed/task-subsystems-2026-10.md §8,
// "SB11b"): the queue entered, the turn admitted, and, for each generation the turn drives, its start, its first token
// received and its end. It also holds the turn's cost counter when the request asked for one, nil otherwise. It is touched
// only from the handler goroutine that drives the turn. A nil *TurnClock (a genRequest not built by prepare) reports nothing.
//
// A turn that drives more than once (no route does today) sums its drives: prefill_ms and decode_ms add each drive's
// Start-to-first-token and first-token-to-end, a drive with no token adds to neither, and ttft_ms runs from the admission to
// the turn's first token. The cost counter sums over the drives the same way.
type TurnClock struct {
	queued, admitted, first time.Time     // the turn's: queue entry, admission, first token of any drive
	started, driveFirst     time.Time     // the current drive's start and first token
	prefill, decode         time.Duration // summed over the drives that received a token
	drivesWithToken         int
	counter                 *cost.Counter
}

func NewTurnClock(wantCost bool) *TurnClock {
	c := &TurnClock{}
	if wantCost {
		c.counter = cost.New()
	}
	return c
}

// stamp sets *t to now the first time it is called for that boundary; a later call keeps the first.
func (c *TurnClock) stamp(t *time.Time) {
	if c != nil && t.IsZero() {
		*t = time.Now()
	}
}

func (c *TurnClock) Queue() {
	if c != nil {
		c.stamp(&c.queued)
	}
}

func (c *TurnClock) Admit() {
	if c != nil {
		c.stamp(&c.admitted)
	}
}

// Start begins a drive.
func (c *TurnClock) Start() {
	if c != nil {
		c.started, c.driveFirst = time.Now(), time.Time{}
	}
}

// FirstToken stamps the current drive's first token, and the turn's if it is the first; later calls keep both.
func (c *TurnClock) FirstToken() {
	if c != nil {
		c.stamp(&c.driveFirst)
		c.stamp(&c.first)
	}
}

// End closes the current drive, adding its prefill and decode when it received a token.
func (c *TurnClock) End() {
	if c == nil || c.started.IsZero() {
		return
	}
	if !c.driveFirst.IsZero() {
		now := time.Now()
		c.prefill += c.driveFirst.Sub(c.started)
		c.decode += now.Sub(c.driveFirst)
		c.drivesWithToken++
	}
	c.started, c.driveFirst = time.Time{}, time.Time{}
}

// CostCounter is the counter a generation counts into, nil (off) when the request did not ask.
func (c *TurnClock) CostCounter() *cost.Counter {
	if c == nil {
		return nil
	}
	return c.counter
}

// TimingsBlock is the response's "timings", X1's fields in milliseconds. A field whose boundaries were not crossed is absent,
// never 0: no token received means no prefill_ms, ttft_ms or decode_ms.
type TimingsBlock struct {
	QueueWaitMs *float64 `json:"queue_wait_ms,omitempty"`
	PrefillMs   *float64 `json:"prefill_ms,omitempty"`
	DecodeMs    *float64 `json:"decode_ms,omitempty"`
	TtftMs      *float64 `json:"ttft_ms,omitempty"`
}

func spanMs(from, to time.Time) *float64 {
	if from.IsZero() || to.IsZero() {
		return nil
	}
	v := float64(to.Sub(from).Microseconds()) / 1000
	return &v
}

func durMs(d time.Duration) *float64 {
	v := float64(d.Microseconds()) / 1000
	return &v
}

// Timings is the turn's block, nil for a nil clock.
func (c *TurnClock) Timings() *TimingsBlock {
	if c == nil {
		return nil
	}
	b := &TimingsBlock{QueueWaitMs: spanMs(c.queued, c.admitted), TtftMs: spanMs(c.admitted, c.first)}
	if c.drivesWithToken > 0 {
		b.PrefillMs, b.DecodeMs = durMs(c.prefill), durMs(c.decode)
	}
	return b
}

// CostReport is the turn's cost report when the request asked, nil otherwise. A path that counts nothing gives an empty
// report, "cost": {}: asked, and no phase measured.
func (c *TurnClock) CostReport() *cost.Report {
	if c.CostCounter() == nil {
		return nil
	}
	r := c.counter.Report()
	return &r
}

// AddTurn puts the turn's "timings" and, when asked, its "cost" into a response object m beside its usage, and returns m.
func (c *TurnClock) AddTurn(m map[string]any) map[string]any {
	if t := c.Timings(); t != nil {
		m["timings"] = t
	}
	if r := c.CostReport(); r != nil {
		m["cost"] = r
	}
	return m
}
