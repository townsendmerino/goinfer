package giw

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestReadTokFile_rejectsSignedInt64Wrap(t *testing.T) {
	p := filepath.Join(t.TempDir(), "wrap.giw")
	var hdr [32]byte
	copy(hdr[:5], bundleMagic)
	binary.LittleEndian.PutUint32(hdr[5:9], bundleVersion)
	// wLen = 2^64 - bundleBlobOffsetV3 wraps signed int64 addition back to offset 0.
	binary.LittleEndian.PutUint64(hdr[9:17], ^uint64(bundleBlobOffsetV3-1))
	if err := os.WriteFile(p, hdr[:], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTokFile(p); err == nil {
		t.Fatal("expected signed int64 wrap in weights length to be rejected")
	}
}
