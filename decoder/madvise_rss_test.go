//go:build linux

package decoder

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// vmRSSKB returns this process's total resident set (VmRSS) in KB. VmRSS counts
// both file-backed (RssFile) and tmpfs/shmem-backed (RssShmem) resident pages, so
// it's correct whether the temp file lands on disk or on a tmpfs t.TempDir().
func vmRSSKB(t *testing.T) int64 {
	t.Helper()
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skipf("no /proc/self/status: %v", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			if f := strings.Fields(line); len(f) >= 2 {
				kb, _ := strconv.ParseInt(f[1], 10, 64)
				return kb
			}
		}
	}
	return -1
}

// TestMadvise_dontneedFreesRSS proves the load-bearing memory claim the whole
// program rests on: MADV_DONTNEED on the read-only mapping must actually FREE
// resident pages, not merely preserve correctness. (TestMadvise_dontneedRefaultsIntact
// only checks the bytes survive a re-fault — true even if DONTNEED were a no-op.)
// We map a file, fault it all in, DONTNEED it, and require resident RAM to drop by
// most of the mapping.
func TestMadvise_dontneedFreesRSS(t *testing.T) {
	const sz = 256 << 20 // 256 MiB
	path := filepath.Join(t.TempDir(), "big.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	chunk := make([]byte, 1<<20)
	for i := range chunk {
		chunk[i] = byte(i*31 + 7) // non-zero pattern (avoid any zero-page sharing)
	}
	for written := 0; written < sz; written += len(chunk) {
		if _, err := f.Write(chunk); err != nil {
			t.Fatal(err)
		}
	}
	f.Close()

	m, err := mmapReadOnly(path)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	defer munmap(m)

	// Fault every page in (sum into a sink the compiler can't drop).
	var sink uint64
	for i := 0; i < len(m); i += 4096 {
		sink += uint64(m[i])
	}
	if sink == 0 {
		t.Fatal("unexpected zero checksum") // also keeps the fault loop live
	}
	before := vmRSSKB(t)

	if err := madviseBytes(m, false); err != nil { // MADV_DONTNEED the whole mapping
		t.Fatalf("MADV_DONTNEED: %v", err)
	}
	after := vmRSSKB(t)

	freed := before - after
	t.Logf("VmRSS %d → %d KB (freed %d KB of ~%d KB mapped)", before, after, freed, sz>>10)
	if freed < int64(sz>>10)/2 {
		t.Fatalf("MADV_DONTNEED freed only %d KB of %d KB — eviction does NOT reduce resident RAM", freed, sz>>10)
	}
}
