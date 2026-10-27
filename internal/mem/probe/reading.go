package probe

// Reading is one sample of the machine's memory, in bytes, in two figures for two uses. Available decides loads (the fit guard
// reads it through Available). Reclaimable decides whether a timed run's machine state is valid: the start checks of
// scripts/night.py and scripts/bench_peer.py and the void rule read it through cmd/hostmem -reclaimable
// (docs/measurements/hostmem-available-2026-10/README.md). On macOS Available is the kernel's own free percentage (kern.memorystatus_level, the figure
// memory_pressure prints, as Level) times physical memory, less PressureBuffer: it counts app memory as available, since the kernel
// can compress it, so a load it admits may compress or swap other apps. Reclaimable is what the kernel can hand back without
// compressing or swapping anything (Free + FileBacked + Purgeable): the figure a timed run needs, since compression under a timed run
// changes what it measures. Speculative (part of FileBacked) and App, Wired and Compressed, Activity Monitor's terms, are context. On
// Linux both Available and Reclaimable are the kernel's MemAvailable and the macOS fields are 0.
type Reading struct {
	Physical, Available                      int64
	Level                                    int
	Reclaimable                              int64
	Free, Speculative, FileBacked, Purgeable int64
	App, Wired, Compressed                   int64
	Source                                   string
}

// PressureBuffer is held back from the macOS percentage's figure; the owner's choice (docs/measurements/noise-registry.md §8).
const PressureBuffer = 500 << 20

// TaskVM is another process's memory as task_info(TASK_VM_INFO) reports it, in bytes. Compressed is its pages in the compressor, held
// in memory or swapped to disk: macOS keeps no separate per-process disk-swap figure.
type TaskVM struct {
	Resident, Internal, Compressed, CompressedPeak, PhysFootprint int64
}

// ReadTask reads pid's TaskVM; ok is false where the platform has none or the query was refused.
func ReadTask(pid int) (TaskVM, bool) { return readTask(pid) }

// Read samples the reading now. ok is false where this platform has none or the query failed; callers treat that as "cannot
// judge", never as no memory. On macOS the sample can be up to a second old: XNU rate-limits host_statistics64 for binaries
// that are not Apple's, and once callers system-wide pass 2 to 10 requests in a second it serves the last fresh reply.
func Read() (r Reading, ok bool) { return read() }

// CPUTicks is the machine's CPU time per CPU state, summed over its cores since boot, in the kernel's ticks. Busy states are user,
// nice and system (on Linux also irq, softirq and steal); the others are idle (on Linux also iowait). Two samples are compared only
// through BusyCores.
type CPUTicks struct {
	busy, idle []uint64
	wrap32     bool // the counts are 32-bit and wrap (macOS)
}

// ReadCPU samples CPUTicks now; ok is false where this platform has none or the query failed. On macOS it is
// host_statistics64(HOST_CPU_LOAD_INFO), which reads fresh at a few samples a second.
func ReadCPU() (CPUTicks, bool) { return readCPU() }

// BusyCores is the average number of busy cores from sample a to the later sample b on a machine with cores cores, at most cores.
// Each state's count is differenced on its own: modulo 2^32 where the counts are 32-bit, and clamped at 0 where a count went back
// (Linux's idle and iowait can step back a tick under NOHZ). ok is false when no tick passed or the samples do not match.
func BusyCores(a, b CPUTicks, cores int) (float64, bool) {
	if len(a.busy) != len(b.busy) || len(a.idle) != len(b.idle) || len(b.busy) == 0 || a.wrap32 != b.wrap32 {
		return 0, false
	}
	delta := func(x, y []uint64) (d uint64) {
		for i := range x {
			switch {
			case b.wrap32:
				d += uint64(uint32(y[i]) - uint32(x[i]))
			case y[i] > x[i]:
				d += y[i] - x[i]
			}
		}
		return d
	}
	busy := delta(a.busy, b.busy)
	total := busy + delta(a.idle, b.idle)
	if total == 0 {
		return 0, false
	}
	return float64(busy) / float64(total) * float64(cores), true
}

// Available is this machine's currently available memory, or 0 when it cannot be determined: Read's Available on Linux and
// macOS, MEMORYSTATUSEX.ullAvailPhys on Windows. Not cached: it changes as other processes run. 0 is a real answer every caller
// treats as "proceed", never as "no memory".
func Available() int64 { return available() }

// OracleSample is a reading bracketed by the kernel's own count of available pages under CONFIG_JETSAM (pageable external + free
// pages, read through sysctl), read immediately before and after it in one process: B41 gate 1a's instrument
// (docs/measurements/hostmem-available-2026-10/README.md, amendment A3). JetsamBefore and JetsamAfter are bytes. Fresh is true
// when Ours was computed inside the bracket: the sysctls are not rate-limited and host_statistics64 is (see Read), so Ours is
// read twice, with a page fault between that a fresh reply counts, and Fresh is whether the two replies differ. A sample that is not
// Fresh may predate the bracket and cannot be graded. The kernel figure is compared with Ours.Reclaimable, not Ours.Available.
type OracleSample struct {
	Ours                      Reading
	JetsamBefore, JetsamAfter int64
	Fresh                     bool
}

// ReadWithOracle takes one OracleSample; ok is false where the platform has no oracle.
func ReadWithOracle() (OracleSample, bool) { return readWithOracle() }
