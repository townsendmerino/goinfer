//go:build !windows

package decoder

import "syscall"

// cpuSeconds returns this process's user+sys CPU time. Utilization is
// cpu/wall, which is what separates "fast because parallel" from "fast".
func cpuSeconds() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	tv := func(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
	return tv(ru.Utime) + tv(ru.Stime)
}
