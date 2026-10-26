package decoder

import "github.com/townsendmerino/goinfer/internal/mem/probe"

// HostRAMBytes is this machine's physical RAM, or 0 when it cannot be determined; 0 is a real answer that every caller
// must treat as "proceed", never as "no memory". Read once per process. Inside a container with a cgroup memory limit it
// reports the host's memory, the safe direction for every guard that reads it.
func HostRAMBytes() int64 { return probe.Physical() }

// HostRAMAvailableBytes is this machine's currently available memory, or 0 when it cannot be determined: MemAvailable on
// Linux, the kernel's free percentage of physical memory less a 500 MiB buffer on macOS, ullAvailPhys on Windows. Not
// cached: it changes as other processes run.
func HostRAMAvailableBytes() int64 { return probe.Available() }

// SwapUsedBytes is the machine's swap in use in bytes, or (0, false) when it cannot be determined. Zero swap in use is a
// real reading, distinct from unknown.
func SwapUsedBytes() (usedBytes int64, ok bool) { return probe.SwapUsed() }

// SelfSwapReader returns the per-process swap reader, the swap this process and its descendants hold, or nil and the
// reason where the platform has no per-process figure (macOS, and every platform but Linux), so the swap guard keeps
// watching the system-wide figure.
func SelfSwapReader() (r SwapReader, unavailable string) { return probe.SelfSwapReader() }

// processFaultCounts is this process's cumulative minor and major page faults; a variable so a test can script them.
var processFaultCounts = probe.FaultCounts
