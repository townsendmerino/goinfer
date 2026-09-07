package decoder

import (
	"fmt"
	"sync"
	"time"
)

// Load-time instrumentation (task: knowing what to download, and how long it takes to load,
// Part B). Before this, the only record of load cost anywhere in the tree was a prose claim in a
// comment — that fanning the GGUF parse across cores turned a 12B's roughly two-minute load into
// seconds. A real and significant result with no measurement behind it, and no way for a user to
// see the number on their own machine.
//
// THE SPLIT IS THE POINT, not the total. A slow load caused by storage and a slow load caused by
// repacking call for completely different responses — a faster disk versus a different quant — and
// a single number cannot tell them apart. The phases are therefore chosen to separate those:
//
//	map    the file becoming addressable: open + mmap + header parse. NOT the file read — see below.
//	build  tensors becoming resident weights: dequantize, quantize, repack. CPU-bound.
//	resident  weights becoming a device-side runner, where a backend builds one. PCIe/GPU-bound.
//
// `map` DOES NOT ISOLATE STORAGE, and the first measurement is what showed it. The loader mmaps, so
// pages fault in lazily during `build`, not during `map` — `map` is header parse alone. Measured
// 2026-09-06 on NVMe: Phi-3-mini (2.23 GB) spent 7ms in `map` and 5.18s in `build`, and the 0.5B
// model spent MORE map time (48ms) on a fifth of the bytes, because that phase tracks metadata
// count rather than size. The storage cost is real but lands inside `build`, where it shows up as
// the cold-minus-warm delta rather than as its own phase.
//
// That delta turned out to be small — 3.3% and 4.4% across the two models — because the repack is
// CPU-bound enough that the kernel's readahead hides most of the I/O behind it. On a spinning disk
// or a network mount that would not hold, which is exactly why the regime has to be recorded
// rather than assumed. Isolating storage properly would need a non-mmap read path or per-phase
// fault accounting; neither is worth it while the answer is "storage is not the problem here".
//
// WHAT THIS CANNOT TELL YOU, and what therefore has to be recorded by whoever measures: whether the
// page cache was cold. A warm-cache `map` phase measures memcpy; a cold one measures the disk, and
// re-running a benchmark gives you a warm one by accident. The profile reports bytes and a rate so
// the regime is at least visible in the number, but the label belongs in docs/benchmarks.md next
// to the storage class, not here.

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

// record adds a phase. Safe on a nil receiver.
func (p *LoadProfile) record(name string, d time.Duration) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phases = append(p.phases, LoadPhase{name, d})
}

// setBytes records the source size, for the map-phase rate. Safe on a nil receiver.
func (p *LoadProfile) setBytes(n int64) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.bytes = n
}

// timed runs f and records how long it took under name. Returns f's error unchanged.
func (p *LoadProfile) timed(name string, f func() error) error {
	t := time.Now()
	err := f()
	p.record(name, time.Since(t))
	return err
}

// Phases returns the recorded spans in the order they ran.
func (p *LoadProfile) Phases() []LoadPhase {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]LoadPhase(nil), p.phases...)
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

	parts := ""
	for i, x := range ph {
		if i > 0 {
			parts += " "
		}
		parts += fmt.Sprintf("%s %s", x.Name, roundDur(x.D))
	}
	out := fmt.Sprintf("load %s (%s) — %.0f%% %s", roundDur(total), parts, share*100, name)
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

// LoadProfile returns how long each phase of this model's load took, or nil if the path that
// produced it was not instrumented (safetensors and .giw are not, today). Every method on the
// result is nil-safe, so a caller can print Summary() unconditionally.
func (m *Model) LoadProfile() *LoadProfile {
	if m == nil {
		return nil
	}
	return m.prof
}
