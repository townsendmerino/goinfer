// Package cost is the cost counters' schema (SB11): per phase (prefill, decode) and per operation, how many calls, bytes read
// and written, kernel launches or command buffers, syncs, allocations and time a generation cost. One shape serves tier-0
// tests, gates and, when the wire carries it, X1's timing block (docs/tasks/task-turn-telemetry-2026-09.md): its phase names and
// Timings' fields are X1's.
//
// A nil *Counter is off: every method is a no-op on it, so a backend's hook costs one nil check per operation. What a backend
// does not measure for an operation is listed in that operation's Unmeasured, never reported as 0 (X1's rule: zero means zero),
// and a phase nothing was counted in is absent from the Report.
//
// Imports: the standard library only.
package cost

import (
	"slices"
	"sync"
)

// Phase is a generation's stage: prefill (the prompt) or decode (each new token).
type Phase uint8

const (
	Prefill Phase = iota
	Decode
	nPhases
)

// String is the phase's wire name.
func (p Phase) String() string {
	if p == Prefill {
		return "prefill"
	}
	return "decode"
}

// Counts is one operation's (or one phase's) cost.
type Counts struct {
	Calls        int64 `json:"calls"`
	BytesRead    int64 `json:"bytes_read"`
	BytesWritten int64 `json:"bytes_written"`
	Launches     int64 `json:"launches"` // kernel launches (dispatches)
	Submits      int64 `json:"submits"`  // command buffers submitted
	Syncs        int64 `json:"syncs"`    // host waits on the device
	Allocs       int64 `json:"allocs"`
	AllocBytes   int64 `json:"alloc_bytes"`
	TimeNs       int64 `json:"time_ns"`
	// Unmeasured names the fields above this backend does not count for this operation (by their JSON names, sorted): they
	// read 0 and mean "not counted", not "none".
	Unmeasured []string `json:"unmeasured,omitempty"`
}

// The field names Unmeasured may list: Counts' JSON names.
const (
	FieldBytesRead    = "bytes_read"
	FieldBytesWritten = "bytes_written"
	FieldLaunches     = "launches"
	FieldSubmits      = "submits"
	FieldSyncs        = "syncs"
	FieldAllocs       = "allocs"
	FieldAllocBytes   = "alloc_bytes"
	FieldTime         = "time_ns"
)

// add sums d into c. A field either side lists unmeasured is unmeasured in the sum and reads 0 there: a partial sum would
// report a measured-looking number for what was only partly counted.
func (c *Counts) add(d Counts) {
	c.Calls += d.Calls
	c.BytesRead += d.BytesRead
	c.BytesWritten += d.BytesWritten
	c.Launches += d.Launches
	c.Submits += d.Submits
	c.Syncs += d.Syncs
	c.Allocs += d.Allocs
	c.AllocBytes += d.AllocBytes
	c.TimeNs += d.TimeNs
	for _, f := range d.Unmeasured {
		if !slices.Contains(c.Unmeasured, f) {
			c.Unmeasured = append(c.Unmeasured, f)
		}
	}
	slices.Sort(c.Unmeasured)
	for _, f := range c.Unmeasured {
		if p := c.field(f); p != nil {
			*p = 0
		}
	}
}

// field is the counter whose JSON name is f, or nil.
func (c *Counts) field(f string) *int64 {
	switch f {
	case FieldBytesRead:
		return &c.BytesRead
	case FieldBytesWritten:
		return &c.BytesWritten
	case FieldLaunches:
		return &c.Launches
	case FieldSubmits:
		return &c.Submits
	case FieldSyncs:
		return &c.Syncs
	case FieldAllocs:
		return &c.Allocs
	case FieldAllocBytes:
		return &c.AllocBytes
	case FieldTime:
		return &c.TimeNs
	}
	return nil
}

// PhaseReport is one phase's cost: its totals over every operation, the tokens it covered, each operation's counts, and its
// wall time where a backend measures it around whole tokens (a GPU backend, whose operations' time is not counted; absent
// otherwise). Calls counts operations run; a device count (launches, syncs, …) is in its own field and never in Calls.
type PhaseReport struct {
	Tokens int64             `json:"tokens"`
	WallNs int64             `json:"wall_ns,omitempty"`
	Totals Counts            `json:"totals"`
	Ops    map[string]Counts `json:"ops"`
}

// Report is a generation's cost, by phase. A phase nothing was counted in is nil, absent from the JSON.
type Report struct {
	Prefill *PhaseReport `json:"prefill,omitempty"`
	Decode  *PhaseReport `json:"decode,omitempty"`
}

// Timings are X1's phase durations, in milliseconds: a phase's wall time where it has one, otherwise its operations' summed
// time; absent (nil) for a phase not measured, or one whose time was not counted either way.
type Timings struct {
	PrefillMs *float64 `json:"prefill_ms,omitempty"`
	DecodeMs  *float64 `json:"decode_ms,omitempty"`
}

// Timings is the report's X1 timing fields.
func (r Report) Timings() Timings {
	ms := func(p *PhaseReport) *float64 {
		if p == nil {
			return nil
		}
		ns := p.WallNs
		if ns == 0 {
			if slices.Contains(p.Totals.Unmeasured, FieldTime) {
				return nil
			}
			ns = p.Totals.TimeNs
		}
		v := float64(ns) / 1e6
		return &v
	}
	return Timings{PrefillMs: ms(r.Prefill), DecodeMs: ms(r.Decode)}
}

// Counter accumulates counts. Safe for concurrent use; a nil *Counter is off.
type Counter struct {
	mu     sync.Mutex
	ops    [nPhases]map[string]*Counts
	tokens [nPhases]int64
	wall   [nPhases]int64
}

// New is an empty, enabled Counter.
func New() *Counter { return &Counter{} }

// Add counts d against operation op in phase p.
func (c *Counter) Add(p Phase, op string, d Counts) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ops[p] == nil {
		c.ops[p] = map[string]*Counts{}
	}
	o := c.ops[p][op]
	if o == nil {
		o = &Counts{}
		c.ops[p][op] = o
	}
	o.add(d)
}

// Tokens counts n tokens covered by phase p.
func (c *Counter) Tokens(p Phase, n int) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.tokens[p] += int64(n)
	c.mu.Unlock()
}

// Wall adds ns of wall time, measured around whole tokens, to phase p.
func (c *Counter) Wall(p Phase, ns int64) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.wall[p] += ns
	c.mu.Unlock()
}

// Report is a copy of the counts so far.
func (c *Counter) Report() Report {
	var r Report
	if c == nil {
		return r
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for p := range nPhases {
		if len(c.ops[p]) == 0 && c.tokens[p] == 0 && c.wall[p] == 0 {
			continue
		}
		pr := &PhaseReport{Tokens: c.tokens[p], WallNs: c.wall[p], Ops: make(map[string]Counts, len(c.ops[p]))}
		for name, o := range c.ops[p] {
			cp := *o
			cp.Unmeasured = slices.Clone(o.Unmeasured)
			pr.Ops[name] = cp
			pr.Totals.add(cp)
		}
		if p == Prefill {
			r.Prefill = pr
		} else {
			r.Decode = pr
		}
	}
	return r
}
