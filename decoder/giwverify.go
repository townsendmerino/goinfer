package decoder

import (
	"fmt"
	"os"
	"strings"
)

// A .giw's trailing CRC-32 covers every byte of the weight payload, and checking it forces a
// read of the whole mapped file. On a streamed load that is the entire load time and the
// entire page-cache footprint: measured 2026-09-23, a 22 GB M35 .giw read over a ~10 MB/s link
// spent 27-28 minutes in that one check (docs/measurements/moe-pager-m35-smb-2026-09-23.md), and
// on a 5.17 GB local .giw the CRC was ~100% of LoadSerializedWeights' cost.
//
// The CRC's job is to catch a corrupt or truncated file. Truncation is already caught for free by
// the lengths recorded in the header; what only the CRC catches is silent corruption inside the
// weights. That is a property of the FILE, not of a load, so it need be checked once per file, not
// once per start: after a load passes the CRC, a small marker next to the file records the
// (size, mtime) it passed at, and later loads of an unchanged file skip the pass.
//
// The marker is best-effort in both directions. A marker that cannot be written (read-only
// directory, an SMB mount) simply means the CRC runs every load, as before; a missing, malformed
// or mismatched marker means the CRC runs. It can only ever remove work from a file that has
// already been verified at exactly this size and mtime.

// giwVerifiedSuffix is appended to a .giw's path to name its marker.
const giwVerifiedSuffix = ".verified"

// giwVerifyEnv forces the full CRC on every load when set to "always" — for anyone who wants the
// whole-file check back regardless of the marker (a suspect disk, a file restored by a tool that
// preserves mtimes).
const giwVerifyEnv = "GOINFER_GIW_VERIFY"

// GIWVerifiedMarkerPath is the marker's path for the .giw at path. Exported so a caller that
// renames a .giw into place (prequant's temp-then-rename publish) can carry the marker along.
func GIWVerifiedMarkerPath(path string) string { return path + giwVerifiedSuffix }

func giwMarkerBody(fi os.FileInfo) string {
	return fmt.Sprintf("giwv1 %d %d\n", fi.Size(), fi.ModTime().UnixNano())
}

// giwVerified reports whether path has a marker matching fi — the file's stat taken BEFORE it was
// loaded, so a file replaced mid-load can never inherit a marker it did not earn.
func giwVerified(path string, fi os.FileInfo) bool {
	if os.Getenv(giwVerifyEnv) == "always" {
		return false
	}
	b, err := os.ReadFile(GIWVerifiedMarkerPath(path))
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(b)) == strings.TrimSpace(giwMarkerBody(fi))
}

// markGIWVerified records that path passed the full CRC at fi's size and mtime. Errors are
// ignored: see the file comment, an unwritable marker only costs the next load a CRC pass.
func markGIWVerified(path string, fi os.FileInfo) {
	_ = os.WriteFile(GIWVerifiedMarkerPath(path), []byte(giwMarkerBody(fi)), 0o644)
}
