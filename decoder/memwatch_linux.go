//go:build linux

package decoder

import (
	"strconv"
	"strings"
)

// SwapUsedBytes reads /proc/meminfo's SwapTotal - SwapFree as bytes, or (0, false) when either
// field is missing or unparseable. Reuses readMeminfo (hostram_linux.go) rather than a second
// os.ReadFile of the same file. Unlike hostram_linux.go's meminfoField (which folds "missing" and
// "genuinely zero" into the same 0 return — fine for a RAM figure that is never legitimately
// zero on a running machine, wrong here: zero swap-used is a real, common reading, and
// SwapWatch's baseline capture must not mistake "field absent" for "zero swap"), this keeps
// found-ness explicit.
func SwapUsedBytes() (usedBytes int64, ok bool) {
	return parseSwapUsedLinuxMeminfo(readMeminfo())
}

// parseSwapUsedLinuxMeminfo is separated from the file read so it is unit-testable with real,
// committed /proc/meminfo content — hostram_linux.go's own split (readMeminfo / meminfoField),
// applied here.
func parseSwapUsedLinuxMeminfo(meminfo string) (usedBytes int64, ok bool) {
	if meminfo == "" {
		return 0, false
	}
	total, haveTotal := meminfoFieldOK(meminfo, "SwapTotal:")
	free, haveFree := meminfoFieldOK(meminfo, "SwapFree:")
	if !haveTotal || !haveFree || total < free {
		return 0, false
	}
	return total - free, true
}

// meminfoFieldOK is meminfoField's found-aware twin: same "Key:   NNNNN kB" shape, but reports
// whether the key was present and well-formed rather than collapsing that into 0.
func meminfoFieldOK(meminfo, key string) (int64, bool) {
	for line := range strings.SplitSeq(meminfo, "\n") {
		rest, ok := strings.CutPrefix(line, key)
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) != 2 || f[1] != "kB" {
			return 0, false
		}
		kb, err := strconv.ParseInt(f[0], 10, 64)
		if err != nil || kb < 0 {
			return 0, false
		}
		return kb * 1024, true
	}
	return 0, false
}
