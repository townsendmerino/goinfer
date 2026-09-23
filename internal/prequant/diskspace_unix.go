//go:build darwin || linux

package prequant

import "syscall"

// statfsFreeBytes reports free space on the filesystem holding dir. ok is false when the
// statfs itself fails (a permissions issue, a path that does not exist yet) — every unknown
// proceeds (fitguard.go's own rule for this repo), never refuses on a guess.
func statfsFreeBytes(dir string) (free int64, ok bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, false
	}
	return int64(st.Bavail) * int64(st.Bsize), true
}
