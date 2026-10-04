package decoder

import "syscall"

// cpuSeconds returns this process's user+kernel CPU time (GetProcessTimes; Windows has no getrusage). Utilization is
// cpu/wall, which is what separates "fast because parallel" from "fast".
func cpuSeconds() float64 {
	h, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0
	}
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(h, &creation, &exit, &kernel, &user); err != nil {
		return 0
	}
	// A Filetime counts 100 ns intervals.
	ft := func(t syscall.Filetime) float64 {
		return float64(uint64(t.HighDateTime)<<32|uint64(t.LowDateTime)) / 1e7
	}
	return ft(kernel) + ft(user)
}
