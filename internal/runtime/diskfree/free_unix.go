//go:build darwin || linux

package diskfree

import "syscall"

// Free reports the space an unprivileged writer has on the filesystem holding dir. ok is false when the statfs itself
// fails (a permissions issue, a path that does not exist yet); a caller then proceeds, never refusing on a guess.
func Free(dir string) (free int64, ok bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, false
	}
	return int64(st.Bavail) * int64(st.Bsize), true
}
