//go:build !linux && !darwin && !windows

package decoder

// HostRAMBytes returns 0 — unknown — on every platform without a probe here (the BSDs and the rest).
// The fit guard treats 0 as "proceed", so those platforms behave exactly as they did before the
// guard existed rather than getting a wrong number: say "unknown", never guess. Windows has its own
// probe (hostram_windows.go) since 2026-10-03.
func HostRAMBytes() int64 { return 0 }

// HostRAMAvailableBytes: same "no probe here" answer as HostRAMBytes, for the same reason.
func HostRAMAvailableBytes() int64 { return 0 }
