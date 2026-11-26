package arch

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
)

// LoadPhase is one named span of a model load.
type LoadPhase struct {
	Name string
	D    time.Duration
}

// LoadProfile accumulates the phases of one model load. The zero value is usable and a nil
// *LoadProfile is safe to record into, so instrumentation never has to be conditional at the call
// site — a load that nobody asked to profile costs a nil check per phase.
type LoadProfile struct {
	mu     sync.Mutex
	phases []LoadPhase
	bytes  int64 // source bytes the load read, when known
}

// Record adds a phase. Safe on a nil receiver.
func (p *LoadProfile) Record(name string, d time.Duration) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phases = append(p.phases, LoadPhase{name, d})
}

// SetBytes records the source size, for the map-phase rate. Safe on a nil receiver.
func (p *LoadProfile) SetBytes(n int64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.bytes = n
}

// Timed runs f and records how long it took under name. Returns f's error unchanged.
func (p *LoadProfile) Timed(name string, f func() error) error {
	t := time.Now()
	err := f()
	p.Record(name, time.Since(t))
	return err
}

// Phases returns the recorded spans in the order they ran.
func (p *LoadProfile) Phases() []LoadPhase {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.phases)
}

// Total is the sum of the recorded phases. It is deliberately the SUM rather than a wall-clock
// span across the whole load: a gap between phases would otherwise be silently attributed to
// whichever phase happened to be adjacent, and an unaccounted gap should show up as phases not
// adding to the caller's own measurement rather than be hidden inside one.
func (p *LoadProfile) Total() time.Duration {
	var t time.Duration
	for _, ph := range p.Phases() {
		t += ph.D
	}
	return t
}

// Dominant returns the costliest phase and its share of the total. That share is the actionable
// part: "load took 9s" tells a user nothing they can act on, "9s, 82% build" tells them the disk
// is not the problem.
func (p *LoadProfile) Dominant() (name string, d time.Duration, share float64) {
	total := p.Total()
	for _, ph := range p.Phases() {
		if ph.D > d {
			name, d = ph.Name, ph.D
		}
	}
	if total > 0 {
		share = float64(d) / float64(total)
	}
	return name, d, share
}

// Summary is the one banner line. Empty when nothing was recorded, so a load path without
// instrumentation prints nothing rather than a misleading zero.
func (p *LoadProfile) Summary() string {
	ph := p.Phases()
	if len(ph) == 0 {
		return ""
	}
	total := p.Total()
	name, d, share := p.Dominant()

	var parts strings.Builder
	for i, x := range ph {
		if i > 0 {
			parts.WriteString(" ")
		}
		parts.WriteString(fmt.Sprintf("%s %s", x.Name, roundDur(x.D)))
	}
	out := fmt.Sprintf("load %s (%s) — %.0f%% %s", roundDur(total), parts.String(), share*100, name)
	_ = d
	if p.bytes > 0 && total > 0 {
		// Source bytes over the WHOLE load, not over `map`. The file is mmap'd, so its pages fault
		// in lazily during `build` rather than being read up front — attributing them to `map`
		// would report a fictional multi-GB/s disk. Called "effective" for that reason: it is
		// throughput of the load, not a disk benchmark.
		gbps := float64(p.bytes) / (1 << 30) / total.Seconds()
		out += fmt.Sprintf("; %.2f GB source, %.2f GB/s effective", float64(p.bytes)/(1<<30), gbps)
	}
	return out
}

// roundDur keeps the banner readable: milliseconds under a second, otherwise two significant places.
func roundDur(d time.Duration) time.Duration {
	if d < time.Second {
		return d.Round(time.Millisecond)
	}
	return d.Round(10 * time.Millisecond)
}
