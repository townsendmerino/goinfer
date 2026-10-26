//go:build !darwin && !linux

package probe

// SwapUsed is the swap probe on platforms with no supported reader: it always reports
// unknown, which StartWatch treats as "do nothing this tick" — never a guess.
func SwapUsed() (usedBytes int64, ok bool) { return 0, false }

// SelfSwapReader reports that there is no per-process swap figure on this platform.
func SelfSwapReader() (r func() (usedBytes int64, ok bool), unavailable string) {
	return nil, "this platform has no per-process swap figure"
}
