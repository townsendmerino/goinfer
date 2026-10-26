//go:build linux

package decoder

import "github.com/townsendmerino/goinfer/internal/mem/probe"

// SelfSwapBytes is the swap this process and every process descended from it hold, in bytes: the sum of VmSwap over the
// pid and its descendants. It is (0, false) when this process's own VmSwap line is missing (a kernel without swap
// accounting) or its status cannot be read, so a caller falls back to the system-wide figure.
func SelfSwapBytes() (usedBytes int64, ok bool) { return probe.SelfSwap() }
