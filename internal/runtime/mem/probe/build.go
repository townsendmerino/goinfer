package probe

import "runtime"

// BuildWorkers is how many layers a weight build converts at once on this machine: one per GOMAXPROCS thread.
func BuildWorkers() int { return runtime.GOMAXPROCS(0) }

// ReclaimsReleasedPages reports whether advising a file mapping's pages away (MADV_DONTNEED) frees them on this system. Only
// Linux reclaims them; elsewhere the advice is a hint, and a build that releases its source still holds it.
func ReclaimsReleasedPages() bool { return runtime.GOOS == "linux" }
