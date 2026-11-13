//go:build goinfer_testhooks

package decoder

import (
	"os"
	"syscall"
	"time"
)

// fileChange is st's change time, inode and device, for checkpointStamp.
func fileChange(st os.FileInfo) (ctime time.Time, ino, dev uint64, ok bool) {
	s, ok := st.Sys().(*syscall.Stat_t)
	if !ok {
		return time.Time{}, 0, 0, false
	}
	return time.Unix(s.Ctimespec.Sec, s.Ctimespec.Nsec), s.Ino, uint64(s.Dev), true
}
