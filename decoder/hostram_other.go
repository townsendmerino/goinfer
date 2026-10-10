//go:build !linux && !darwin && !windows

package decoder

// HostRAMBytes returns 0, unknown, on every platform without a probe here (the BSDs and the rest). The fit guard treats
// 0 as "proceed", so those platforms get no wrong number: say "unknown", never guess. Windows has its own probe
// (hostram_windows.go).
func HostRAMBytes() int64 { return 0 }

// HostRAMAvailableBytes: same "no probe here" answer as HostRAMBytes, for the same reason.
func HostRAMAvailableBytes() int64 { return 0 }
