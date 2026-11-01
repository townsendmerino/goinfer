package serveapp

import (
	"time"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// vendorOptions is a request body's "goinfer" object: goinfer's own request options, ignored by every other server
// (SB11b, docs/server.md "Per-turn timings and cost"; experimental until v1.0).
type vendorOptions struct {
	// Cost asks for the turn's cost report (internal/cost) as the response's "cost".
	Cost bool `json:"cost"`
}

func (v *vendorOptions) wantsCost() bool { return v != nil && v.Cost }

// turnClock is one turn's stage boundaries, the only places X1's timings are read (docs/tasks/task-subsystems-2026-10.md §8,
// "SB11b"): the queue entered, the turn admitted, the generation started, its first token received and its end. It also holds
// the turn's cost counter when the request asked for one, nil otherwise. It is touched only from the handler goroutine
// that drives the turn. A nil *turnClock (a genRequest not built by prepare) reports nothing.
type turnClock struct {
	queued, admitted, started, first, ended time.Time
	counter                                 *cost.Counter
}

func newTurnClock(wantCost bool) *turnClock {
	c := &turnClock{}
	if wantCost {
		c.counter = cost.New()
	}
	return c
}

// stamp sets *t to now the first time it is called for that boundary; a later call keeps the first.
func (c *turnClock) stamp(t *time.Time) {
	if c != nil && t.IsZero() {
		*t = time.Now()
	}
}

func (c *turnClock) queue() {
	if c != nil {
		c.stamp(&c.queued)
	}
}

func (c *turnClock) admit() {
	if c != nil {
		c.stamp(&c.admitted)
	}
}

func (c *turnClock) start() {
	if c != nil {
		c.stamp(&c.started)
	}
}

func (c *turnClock) firstToken() {
	if c != nil {
		c.stamp(&c.first)
	}
}

// end stamps the generation's end; a turn that drives more than once (a retried tool turn) keeps its last end.
func (c *turnClock) end() {
	if c != nil {
		c.ended = time.Now()
	}
}

// costCounter is the counter a generation counts into, nil (off) when the request did not ask.
func (c *turnClock) costCounter() *cost.Counter {
	if c == nil {
		return nil
	}
	return c.counter
}

// timingsBlock is the response's "timings", X1's fields in milliseconds. A field whose two boundaries were not both crossed
// is absent, never 0: no token received means no prefill_ms, ttft_ms or decode_ms.
type timingsBlock struct {
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

// timings is the turn's block, nil for a nil clock.
func (c *turnClock) timings() *timingsBlock {
	if c == nil {
		return nil
	}
	return &timingsBlock{
		QueueWaitMs: spanMs(c.queued, c.admitted),
		PrefillMs:   spanMs(c.started, c.first),
		DecodeMs:    spanMs(c.first, c.ended),
		TtftMs:      spanMs(c.admitted, c.first),
	}
}

// costReport is the turn's cost report when the request asked, nil otherwise. A path that counts nothing gives an empty
// report, "cost": {}: asked, and no phase measured.
func (c *turnClock) costReport() *cost.Report {
	if c.costCounter() == nil {
		return nil
	}
	r := c.counter.Report()
	return &r
}

// addTurn puts the turn's "timings" and, when asked, its "cost" into a response object m beside its usage, and returns m.
func (c *turnClock) addTurn(m map[string]any) map[string]any {
	if t := c.timings(); t != nil {
		m["timings"] = t
	}
	if r := c.costReport(); r != nil {
		m["cost"] = r
	}
	return m
}
