//go:build !darwin && !linux

package decoder

// SwapUsedBytes is the swap probe on platforms with no supported reader: it always reports
// unknown, which StartSwapWatch treats as "do nothing this tick" — never a guess.
func SwapUsedBytes() (usedBytes int64, ok bool) { return 0, false }
