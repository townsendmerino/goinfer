package hostmem

import (
	"encoding/hex"
	"testing"
)

// A vm_statistics64 that host_statistics64 returned on a 16 GiB M1 Pro (2026-10-21, 16 KiB pages), and its fields and Activity
// Monitor split as Python's struct.unpack('<4I9Q2I4Q4IQ') decoded them independently of this package.
const sampleVMStats64 = "49290000c8920300d4760300eadd0100b7c93d40000000002703d506000000005933d1040000000010cd04000000000030596499000000002af4870e" +
	"0000000000000000000000000000000000000000e166db000000000042000000bd170000b53a6d05000000001f6ecc0600000000b6d30d0000000000c5c316" +
	"00000000009e54060000000000f454020065cc0400c5060c0000000000"

func TestDecodeVMStats64_sample(t *testing.T) {
	b, err := hex.DecodeString(sampleVMStats64)
	if err != nil {
		t.Fatal(err)
	}
	s, ok := decodeVMStats64(b)
	want := vmStats64{free: 10569, active: 234184, inactive: 227028, wire: 122346, purgeable: 66, speculative: 6077, compressor: 414878,
		external: 152820, internal: 314469}
	if !ok || s != want {
		t.Fatalf("decoded %+v, want %+v", s, want)
	}
	if _, ok := decodeVMStats64(b[:151]); ok {
		t.Error("a short buffer decoded")
	}
}

func TestFromVMStats64_activityMonitorSplit(t *testing.T) {
	b, _ := hex.DecodeString(sampleVMStats64)
	s, _ := decodeVMStats64(b)
	r := fromVMStats64(s, 16384, 17179869184)
	if r.App != 5151178752 || r.Wired != 2004516864 || r.Compressed != 6797361152 || r.Available != 3226812416 {
		t.Fatalf("got %+v; want app 5151178752, wired 2004516864, compressed 6797361152, available 3226812416", r)
	}
	// File-backed (external) pages are not in the sum: twice the cache leaves the reading unchanged, twice the app memory lowers it.
	c := s
	c.external *= 2
	if fromVMStats64(c, 16384, 17179869184).Available != r.Available {
		t.Error("cached file pages lowered the reading")
	}
	a := s
	a.internal += 65536 // 1 GiB of anonymous pages
	if got := r.Available - fromVMStats64(a, 16384, 17179869184).Available; got != 1<<30 {
		t.Errorf("1 GiB of app memory lowered the reading by %d", got)
	}
}

func TestRead_live(t *testing.T) {
	r, ok := Read()
	if !ok {
		t.Skip("no reading on this platform")
	}
	if r.Physical <= 0 || r.Available <= 0 || r.Available > r.Physical {
		t.Fatalf("implausible reading %+v", r)
	}
	t.Logf("%+v", r)
}
