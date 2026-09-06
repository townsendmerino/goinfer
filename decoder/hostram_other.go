//go:build !linux && !darwin

package decoder

// HostRAMBytes returns 0 — unknown — on every platform without a probe here (Windows, the BSDs).
// The fit guard treats 0 as "proceed", so those platforms behave exactly as they did before the
// guard existed rather than getting a wrong number. docs/task-fit-to-hardware.md §8 records the
// same position for Windows: say "unknown", never guess.
func HostRAMBytes() int64 { return 0 }
