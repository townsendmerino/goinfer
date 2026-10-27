//go:build darwin || linux

package probe

import "syscall"

// FaultCounts reads this PROCESS's cumulative minor/major page-fault counts since
// process start, via getrusage(RUSAGE_SELF) — pure syscall package, no x/sys dependency
// (syscall.Rusage exposes Minflt/Majflt identically on darwin and linux). ok is
// false only on the syscall itself failing, which does not happen in practice on either
// platform; every unknown proceeds, as with every reading here.
func FaultCounts() (minflt, majflt int64, ok bool) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0, 0, false
	}
	return int64(ru.Minflt), int64(ru.Majflt), true
}
