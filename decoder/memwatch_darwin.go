//go:build darwin

package decoder

import (
	"os/exec"
	"strconv"
	"strings"
)

// SwapUsedBytes reads `sysctl -n vm.swapusage`'s "used" field as bytes, or (0, false) when it
// cannot be determined. Same shelled-out convention as HostRAMBytes (hostram_darwin.go):
// syscall.Sysctl truncates at the first NUL, and this is a formatted string ("total = 0.00M used
// = 0.00M free = 0.00M (encrypted)"), not a scalar, so there is no bare-syscall alternative. Same
// field this repo's own swapUsed test helper (mellum2_prefill_profile_test.go) already reads as
// a string; this parses the same field into typed bytes for SwapWatchOptions.Read's contract.
func SwapUsedBytes() (usedBytes int64, ok bool) {
	out, err := exec.Command("sysctl", "-n", "vm.swapusage").Output()
	if err != nil {
		return 0, false
	}
	return parseSwapUsageDarwin(string(out))
}

// parseSwapUsageDarwin is separated from the exec.Command call so it is unit-testable with real,
// committed sysctl output — hostram_darwin.go's own split (parseVMStatAvailable), applied here.
func parseSwapUsageDarwin(out string) (usedBytes int64, ok bool) {
	fields := strings.Fields(out)
	for i, f := range fields {
		if f != "used" || i+2 >= len(fields) {
			continue
		}
		mb, err := strconv.ParseFloat(strings.TrimSuffix(fields[i+2], "M"), 64)
		if err != nil || mb < 0 {
			return 0, false
		}
		return int64(mb * 1024 * 1024), true
	}
	return 0, false
}
