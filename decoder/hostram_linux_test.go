//go:build linux

package decoder

import "testing"

// R13-follow-on (docs/measurements/cold-user-2026-09-07-macbook-arm64.md's live re-run of the R13
// fix): HostRAMAvailableBytes exists because HostRAMBytes (total physical RAM) is what the fit
// guard budgeted against, and a fixed fraction of TOTAL RAM assumes nothing else on the machine
// ever needs more than the rest — false on a real machine with a browser or IDE open. This tests
// the shared meminfo-field parser both readers now use.
const sampleMeminfo = `MemTotal:       16311288 kB
MemFree:         1234567 kB
MemAvailable:    5766224 kB
Buffers:          234567 kB
Cached:          3456789 kB
`

func TestMeminfoField_realShape(t *testing.T) {
	if got, want := meminfoField(sampleMeminfo, "MemTotal:"), int64(16311288)*1024; got != want {
		t.Errorf("MemTotal: got %d, want %d", got, want)
	}
	if got, want := meminfoField(sampleMeminfo, "MemAvailable:"), int64(5766224)*1024; got != want {
		t.Errorf("MemAvailable: got %d, want %d", got, want)
	}
}

func TestMeminfoField_missingKeyIsUnknown(t *testing.T) {
	if got := meminfoField(sampleMeminfo, "MemAvailable:"); got == 0 {
		t.Fatal("test fixture should have MemAvailable — this would make the next assertion vacuous")
	}
	if got := meminfoField(sampleMeminfo, "NoSuchKey:"); got != 0 {
		t.Errorf("missing key: got %d, want 0 (unknown)", got)
	}
}

func TestMeminfoField_unexpectedUnitIsUnknown(t *testing.T) {
	// A missing or unexpected unit means "we do not know", never a guess off by 1024.
	const badUnit = "MemTotal:       16311288 MB\n"
	if got := meminfoField(badUnit, "MemTotal:"); got != 0 {
		t.Errorf("unexpected unit: got %d, want 0 (unknown, not off by 1024)", got)
	}
}

// HostRAMAvailableBytes must actually read live /proc/meminfo on this real Linux box — not
// cached like HostRAMBytes, since availability is the whole point of the field existing.
func TestHostRAMAvailableBytes_readsRealMeminfo(t *testing.T) {
	got := HostRAMAvailableBytes()
	if got <= 0 {
		t.Fatalf("HostRAMAvailableBytes() = %d on a real Linux box with a real /proc/meminfo", got)
	}
	total := HostRAMBytes()
	if total <= 0 {
		t.Fatal("HostRAMBytes() = 0 on a real Linux box — test environment problem, not this code's")
	}
	if got > total {
		t.Errorf("available (%d) > total (%d) — impossible", got, total)
	}
}
