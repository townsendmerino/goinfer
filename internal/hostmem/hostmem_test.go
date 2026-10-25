package hostmem

import (
	"encoding/binary"
	"encoding/hex"
	"runtime"
	"testing"
	"time"
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
	if r.Reclaimable != 2578481152 || r.Free != 73596928 || r.App != 5151178752 || r.Wired != 2004516864 || r.Compressed != 6797361152 {
		t.Fatalf("got %+v; want reclaimable 2578481152, free 73596928, app 5151178752, wired 2004516864, compressed 6797361152", r)
	}
	// A GiB more of cached file pages is a GiB more reclaimable; a GiB more of app pages, wired or compressed pages changes nothing.
	c := s
	c.external += 65536
	if got := fromVMStats64(c, 16384, 17179869184).Reclaimable - r.Reclaimable; got != 1<<30 {
		t.Errorf("1 GiB of file-backed pages raised the reading by %d", got)
	}
	for name, f := range map[string]func(*vmStats64){"app": func(x *vmStats64) { x.internal += 65536 }, "wired": func(x *vmStats64) { x.wire += 65536 },
		"compressed": func(x *vmStats64) { x.compressor += 65536 }} {
		a := s
		f(&a)
		if fromVMStats64(a, 16384, 17179869184).Reclaimable != r.Reclaimable {
			t.Errorf("1 GiB of %s pages changed the reading", name)
		}
	}
	// A free page that becomes speculative read-ahead stays in the free count and joins the file-backed count: still one page. The
	// sample has 4492 free pages that are not speculative; 4096 of them turn.
	sp := s
	sp.speculative += 4096
	sp.external += 4096
	if got := fromVMStats64(sp, 16384, 17179869184).Reclaimable; got != r.Reclaimable {
		t.Errorf("64 MiB of free pages turned speculative moved the reading by %d", got-r.Reclaimable)
	}
}

// TestDecodeTaskVM_fields lays out struct task_vm_info (mach/task_info.h) field by field with a distinct value in each, so a decoder
// that read a field's neighbour (resident_size_peak for resident_size, compressed_lifetime for compressed_peak) fails.
func TestDecodeTaskVM_fields(t *testing.T) {
	fields := []string{"virtual_size", "region_count+page_size", "resident_size", "resident_size_peak", "device", "device_peak",
		"internal", "internal_peak", "external", "external_peak", "reusable", "reusable_peak", "purgeable_volatile_pmap",
		"purgeable_volatile_resident", "purgeable_volatile_virtual", "compressed", "compressed_peak", "compressed_lifetime",
		"phys_footprint"}
	b := make([]byte, 8*len(fields))
	val := map[string]int64{}
	for i, f := range fields {
		v := int64(i+1) << 20
		binary.LittleEndian.PutUint64(b[8*i:], uint64(v))
		val[f] = v
	}
	if len(b) != 152 {
		t.Fatalf("layout is %d bytes, want 152 (through phys_footprint)", len(b))
	}
	got := decodeTaskVM(b)
	want := TaskVM{Resident: val["resident_size"], Internal: val["internal"], Compressed: val["compressed"],
		CompressedPeak: val["compressed_peak"], PhysFootprint: val["phys_footprint"]}
	if got != want {
		t.Fatalf("decoded %+v, want %+v", got, want)
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

func TestBusyCores(t *testing.T) {
	lin := func(busy, idle uint64) CPUTicks { return CPUTicks{busy: []uint64{busy, 0}, idle: []uint64{idle, 0}} }
	mac := func(busy, idle uint64) CPUTicks {
		return CPUTicks{busy: []uint64{busy}, idle: []uint64{idle}, wrap32: true}
	}
	for _, c := range []struct {
		name string
		a, b CPUTicks
		want float64
		ok   bool
	}{
		{"half busy", lin(100, 900), lin(500, 1300), 4, true},
		{"no tick", lin(100, 900), lin(100, 900), 0, false},
		{"idle steps back a tick (NOHZ): clamped", lin(100, 900), lin(500, 899), 8, true},
		{"busy count wraps at 2^32", mac(1<<32-100, 900), mac(300, 1300), 4, true},
		{"idle count wraps at 2^32", mac(100, 1<<32-200), mac(500, 200), 4, true},
		{"mismatched samples", lin(1, 1), mac(2, 2), 0, false},
	} {
		got, ok := BusyCores(c.a, c.b, 8)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("%s: %v, %v; want %v, %v", c.name, got, ok, c.want, c.ok)
		}
	}
}

// TestReadCPU_live: two goroutines spinning for 400 ms read at least 1.0 busy cores more than the 400 ms windows on either side of
// them, so other processes' load cannot satisfy it alone.
func TestReadCPU_live(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs two Ps to spin two goroutines at once")
	}
	if _, ok := ReadCPU(); !ok {
		t.Skip("no CPU reading on this platform")
	}
	window := func(spin bool) float64 {
		a, _ := ReadCPU()
		stop := time.Now().Add(400 * time.Millisecond)
		done := make(chan struct{}, 2)
		if spin {
			for range 2 {
				go func() {
					for time.Now().Before(stop) {
					}
					done <- struct{}{}
				}()
			}
			<-done
			<-done
		} else {
			time.Sleep(time.Until(stop))
		}
		b, ok := ReadCPU()
		got, bok := BusyCores(a, b, runtime.NumCPU())
		if !ok || !bok {
			t.Fatalf("CPU reading failed: %v, %v", ok, bok)
		}
		return got
	}
	before, spun, after := window(false), window(true), window(false)
	if spun < max(before, after)+1.0 {
		t.Fatalf("busy cores: %.2f before, %.2f with two spinning goroutines, %.2f after; want the spin at least 1.0 above both", before, spun, after)
	}
	t.Logf("busy cores: %.2f before, %.2f spinning, %.2f after", before, spun, after)
}

func TestReadWithOracle_live(t *testing.T) {
	var o OracleSample
	for attempt := 1; ; attempt++ {
		var ok bool
		if o, ok = ReadWithOracle(); !ok {
			t.Skip("no oracle on this platform")
		}
		d := o.JetsamAfter - o.JetsamBefore
		t.Logf("attempt %d: ours %d, jetsam %d then %d, fresh %v (speculative %d, purgeable %d)", attempt, o.Ours.Reclaimable,
			o.JetsamBefore, o.JetsamAfter, o.Fresh, o.Ours.Speculative, o.Ours.Purgeable)
		if o.Fresh && d >= -oracleStill && d <= oracleStill {
			break
		}
		if attempt == oracleAttempts {
			t.Skipf("no gradable sample in %d attempts: the last was fresh %v, memory moved %d bytes across it", attempt, o.Fresh, d)
		}
		time.Sleep(rateLimitWindow) // a new window, so this process's next request is served fresh unless others crowd it
	}
	if o.JetsamBefore <= 0 || o.JetsamAfter <= 0 || o.JetsamBefore > o.Ours.Physical || o.Ours.Reclaimable <= 0 || o.Ours.Available <= 0 {
		t.Fatalf("implausible sample %+v", o)
	}
	// Ours less purgeable is the kernel's figure (free + pageable external pages): a speculative page counted twice shows here.
	if diff := o.Ours.Reclaimable - o.Ours.Purgeable - (o.JetsamBefore+o.JetsamAfter)/2; diff < -oracleTolerance || diff > oracleTolerance {
		t.Errorf("ours less purgeable is %d bytes from the kernel's figure (speculative %d)", diff, o.Ours.Speculative)
	}
}

// TestReadWithOracle_rateLimitedIsNotFresh: once this process has used up the rate limit's window, the kernel serves the cached reply,
// and the oracle must say so rather than offer it for grading.
func TestReadWithOracle_rateLimitedIsNotFresh(t *testing.T) {
	if _, ok := ReadWithOracle(); !ok {
		t.Skip("no oracle on this platform")
	}
	stale := 0
	for i := 0; i < 5; i++ {
		for j := 0; j < 25; j++ { // past the most requests a window serves fresh (10)
			Read()
		}
		if o, _ := ReadWithOracle(); !o.Fresh {
			stale++
		}
	}
	// Another process's fresh request between the oracle's two reads makes a sample fresh, so one stale sample in five is the bar.
	if stale == 0 {
		t.Errorf("no sample reported stale after the window was used up: the oracle cannot see a cached reply")
	}
	t.Logf("%d of 5 samples stale", stale)
}

// oracleStill is gate 1a's bracket (a sample is graded only when the kernel's figure moved at most this much across it), and
// oracleTolerance its bar (docs/measurements/hostmem-available-2026-10/README.md, amendment A5). A sample that is stale or moved is
// retaken up to oracleAttempts times, rateLimitWindow apart: XNU's host_statistics64 window is one second (see Read).
const (
	oracleStill, oracleTolerance = (1 << 30) / 50, (1 << 30) / 20
	oracleAttempts               = 3
	rateLimitWindow              = 1100 * time.Millisecond
)

func TestFromLevel(t *testing.T) {
	const phys = 17179869184 // 16 GiB
	for _, c := range []struct {
		level int
		want  int64
	}{{83, 13735003422}, {100, 16655581184}, {3, 0}, {0, 0}} { // 16 GiB × level / 100, less 524,288,000 bytes
		if got := fromLevel(c.level, phys); got != c.want {
			t.Errorf("level %d: got %d, want %d", c.level, got, c.want)
		}
	}
}
