package giw

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadTokFile_rejectsSignedInt64Wrap(t *testing.T) {
	p := filepath.Join(t.TempDir(), "wrap.giw")
	// Make a file >= bundleBlobOffsetV3+4 bytes and craft wLen so signed int64 addition
	// (bundleBlobOffsetV3 + int64(wLen)) wraps back to offset 20, where a valid tokLen=4 + "FAKE"
	// sits: without the uint64 bounds check, ReadTokFile would succeed and return "FAKE".
	var buf [bundleBlobOffsetV3 + 16]byte
	copy(buf[:5], bundleMagic)
	binary.LittleEndian.PutUint32(buf[5:9], bundleVersion)
	const wrappedOff = 20
	wLen := ^uint64(bundleBlobOffsetV3 - wrappedOff - 1)
	binary.LittleEndian.PutUint64(buf[9:17], wLen)
	binary.LittleEndian.PutUint32(buf[wrappedOff:wrappedOff+4], 4)
	copy(buf[wrappedOff+4:wrappedOff+8], "FAKE")
	if err := os.WriteFile(p, buf[:], 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := ReadTokFile(p)
	if err == nil || !strings.Contains(err.Error(), "weights length") {
		t.Fatalf("ReadTokFile wrapped wLen = (%q, %v), want weights length error", got, err)
	}

	shortPath := filepath.Join(t.TempDir(), "short.giw")
	if err := os.WriteFile(shortPath, buf[:32], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadTokFile(shortPath); err == nil || !strings.Contains(err.Error(), "too short") {
		t.Fatalf("ReadTokFile short file err = %v, want too short error", err)
	}
}
