//go:build !linux && !darwin && !windows

package probe

// Physical returns 0, unknown, on every platform without a probe here (the BSDs and the rest). The fit guard treats
// 0 as "proceed", so those platforms get no wrong number: say "unknown", never guess.
func Physical() int64 { return 0 }

// available: the same "no probe here" answer as Physical, for the same reason.
func available() int64 { return 0 }
