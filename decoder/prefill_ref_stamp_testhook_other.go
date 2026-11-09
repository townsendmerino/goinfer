//go:build goinfer_testhooks && !darwin && !linux

package decoder

import (
	"os"
	"time"
)

// fileChange reports no change time here (Windows reports none), so checkpointStamp vouches for no file and every hash is computed.
func fileChange(os.FileInfo) (ctime time.Time, ino, dev uint64, ok bool) {
	return time.Time{}, 0, 0, false
}
