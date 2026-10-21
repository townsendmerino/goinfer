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

func TestFromVMStats64_addsUpReclaimable(t *testing.T) {
	b, _ := hex.DecodeString(sampleVMStats64)
	s, _ := decodeVMStats64(b)
	// The sample's anonymous and file-backed pages are its active, inactive and speculative pages: speculative is inside file-backed.
	if s.internal+s.external != s.active+s.inactive+s.speculative {
		t.Fatalf("sample: internal+external %d, active+inactive+speculative %d", s.internal+s.external, s.active+s.inactive+s.speculative)
	}
	r := fromVMStats64(s, 16384, 17179869184)
	// free 10569 less speculative 6077 + file-backed 152820 + purgeable 66 = 157378 pages
	if r.Available != 2578481152 || r.Free != 73596928 || r.App != 5151178752 || r.Wired != 2004516864 || r.Compressed != 6797361152 {
		t.Fatalf("got %+v; want available 2578481152, free 73596928, app 5151178752, wired 2004516864, compressed 6797361152", r)
	}
	// A GiB more of cached file pages is a GiB more available; a GiB more of app pages, wired or compressed pages changes nothing.
	c := s
	c.external += 65536
	if got := fromVMStats64(c, 16384, 17179869184).Available - r.Available; got != 1<<30 {
		t.Errorf("1 GiB of file-backed pages raised the reading by %d", got)
	}
	for name, f := range map[string]func(*vmStats64){"app": func(x *vmStats64) { x.internal += 65536 }, "wired": func(x *vmStats64) { x.wire += 65536 },
		"compressed": func(x *vmStats64) { x.compressor += 65536 }} {
		a := s
		f(&a)
		if fromVMStats64(a, 16384, 17179869184).Available != r.Available {
			t.Errorf("1 GiB of %s pages changed the reading", name)
		}
	}
	// A free page that becomes speculative read-ahead stays in the free count and joins the file-backed count: still one page. The
	// sample has 4492 free pages that are not speculative; 4096 of them turn.
	sp := s
	sp.speculative += 4096
	sp.external += 4096
	if got := fromVMStats64(sp, 16384, 17179869184).Available; got != r.Available {
		t.Errorf("64 MiB of free pages turned speculative moved the reading by %d", got-r.Available)
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

func TestReadWithOracle_live(t *testing.T) {
	o, ok := ReadWithOracle()
	if !ok {
		t.Skip("no oracle on this platform")
	}
	if o.JetsamBefore <= 0 || o.JetsamAfter <= 0 || o.JetsamBefore > o.Ours.Physical || o.Ours.Available <= 0 {
		t.Fatalf("implausible sample %+v", o)
	}
	t.Logf("ours %d, jetsam %d then %d (speculative %d, purgeable %d)", o.Ours.Available, o.JetsamBefore, o.JetsamAfter,
		o.Ours.Speculative, o.Ours.Purgeable)
	// Ours less purgeable is the kernel's figure (free + pageable external pages): a speculative page counted twice shows here.
	if d := o.JetsamAfter - o.JetsamBefore; d < -oracleStill || d > oracleStill {
		t.Skipf("memory moved %d bytes across the bracket", d)
	}
	if diff := o.Ours.Available - o.Ours.Purgeable - (o.JetsamBefore+o.JetsamAfter)/2; diff < -oracleTolerance || diff > oracleTolerance {
		t.Errorf("ours less purgeable is %d bytes from the kernel's figure (speculative %d)", diff, o.Ours.Speculative)
	}
}

// oracleStill is gate 1a's bracket (a sample is graded only when the kernel's figure moved at most this much across it), and
// oracleTolerance its bar (docs/measurements/hostmem-available-2026-10/README.md, amendment A5).
const oracleStill, oracleTolerance = (1 << 30) / 50, (1 << 30) / 20
