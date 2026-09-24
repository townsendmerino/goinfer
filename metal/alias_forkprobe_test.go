//go:build darwin

package metal

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/mmap"
)

// TestAliasForkProbe is the decisive experiment for the M26 alias collapse
// (docs/measurements/s6-alias-2026-09-24.md, "The M26 alias arm collapses"): does a fork() of a process
// that holds a GPU-wired no-copy buffer over a PROT_READ|MAP_PRIVATE file mapping copy the WHOLE mapping
// eagerly? XNU says it must — IOPL wiring faults each page with write intent (COW copy into a wired shadow,
// vm_pageout.c vm_object_iopl_request), marks the object true_share, and vm_map_fork then routes the entry
// to slow_vm_map_fork_copy → vm_object_copy_slowly of the entire entry (vm_map.c, vm_object.c). The serving
// swap guard forks every 2 s (exec of `sysctl`), so on a 15 GB mapping that copy is the collapse.
//
// No model, no server. A fresh 2 GiB file of random bytes written with F_NOCACHE (uncached, so page-ins are
// countable), mapped exactly as decoder.Load maps a .giw (aikit mmap.MapReadOnly), ONE no-copy buffer over
// a 16 MiB page-aligned window mid-file, one dispatch touching one float per page of THAT WINDOW ONLY, and
// fork+exec(/usr/bin/true) timed: before the mapping (t0), after the buffer exists but before any GPU touch
// (t1), three times after the touch (t2a-c), and after the buffer is released and the file unmapped (t3).
// An external shell samples vm_stat every 50 ms throughout, so a transient anonymous copy made during a
// slow fork is seen even though the child frees it at exec.
//
// Readings: t1 ≈ t0 and wired +≈1,024 pages at the touch, then t2 = seconds with system page-ins ≈ the
// file's 131,072 pages and a transient Anonymous rise of the same size ⇒ H1 (window-precise wiring; fork
// copies the whole entry) — the fix is a fork-free swap guard (and/or VM_INHERIT_NONE on the mapping).
// wired +≈131,072 pages at the touch ⇒ H2 (the driver wires the whole region). t2 ≈ t0 ⇒ neither; the
// M26 collapse needs another explanation. t3 ≈ t0 closes the loop: the wired buffer was the cause.
//
//	GOINFER_ALIAS_FORK_PROBE=1 go test -tags goinfer_testhooks ./metal/ -run TestAliasForkProbe -v -count=1
func TestAliasForkProbe(t *testing.T) {
	if os.Getenv("GOINFER_ALIAS_FORK_PROBE") == "" {
		t.Skip("set GOINFER_ALIAS_FORK_PROBE=1 (writes a 2 GiB scratch file, maps it, makes a ~2 GiB transient copy during a fork)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const page = 16384
	const fileBytes = int64(2) << 30 // 2 GiB = 131,072 pages
	const windowBytes = 16 << 20     // 16 MiB = 1,024 pages
	dir := os.Getenv("GOINFER_ALIAS_FORK_PROBE_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	path := filepath.Join(dir, "forkprobe.bin")
	writeUncachedRandom(t, path, fileBytes)
	defer os.Remove(path)

	// The external sampler: its own process, so Go's ForkLock and the slow fork under test never block it.
	samplePath := filepath.Join(dir, "vmstat-samples.log")
	sampler := exec.Command("/bin/sh", "-c",
		`while :; do perl -MTime::HiRes=time -e 'printf "T %.3f\n", time'; vm_stat | grep -E 'Pages free|wired down|Anonymous|File-backed|compressor|Pageins'; sleep 0.05; done`)
	sf, err := os.Create(samplePath)
	if err != nil {
		t.Fatal(err)
	}
	sampler.Stdout = sf
	if err := sampler.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = sampler.Process.Kill(); _ = sampler.Wait(); sf.Close() }()
	time.Sleep(300 * time.Millisecond)

	forkMS := func() float64 {
		t0 := time.Now()
		if err := exec.Command("/usr/bin/true").Run(); err != nil {
			t.Fatalf("fork+exec: %v", err)
		}
		return float64(time.Since(t0).Microseconds()) / 1000
	}
	median3 := func() float64 {
		v := []float64{forkMS(), forkMS(), forkMS()}
		sort.Float64s(v)
		return v[1]
	}
	mark := func(label string) time.Time {
		now := time.Now()
		t.Logf("%s  @%.3f  vm_stat: %s", label, float64(now.UnixNano())/1e9, vmStatLine(t))
		return now
	}

	// P0: nothing mapped.
	mark("P0 baseline")
	t0 := median3()
	t.Logf("P0 fork+exec, no mapping: %.2f ms", t0)

	// P1: map the whole file as decoder.Load does; one no-copy buffer over a 16 MiB window mid-file.
	data, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	base := uintptr(unsafe.Pointer(&data[0]))
	if base%page != 0 {
		t.Fatalf("mapping base %#x not page-aligned", base)
	}
	off := (len(data) / 2) &^ (page - 1)
	win := data[off : off+windowBytes]
	lib, err := d.CompileLibrary(forkProbeTouchSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "touch_pages")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	wBeforeBuf := vmStat(t)["wired"]
	buf := d.NewBufferNoCopy(unsafe.Pointer(&win[0]), windowBytes)
	mark("P1 mapped + no-copy buffer created (untouched)")
	t1 := median3()
	t.Logf("P1 fork+exec, buffer created, no GPU touch: %.2f ms (wired Δ since P0: %+d pages)", t1, vmStat(t)["wired"]-wBeforeBuf)

	// P2: the GPU touches one float per page of the window only. NOTHING forks between the touch and the
	// first timed fork (vm_stat is itself a fork+exec — the first version of this probe called it right
	// after the touch and that untimed fork paged in the whole file; see the record).
	const nPages = windowBytes / page
	out := d.NewBufferLen(nPages)
	q := d.NewCommandQueue()
	stride := gpu.NewBufferOf(d, []uint32{uint32(page / 4)})
	pageinsBefore := vmStat(t)["pageins"]
	wBeforeTouch := vmStat(t)["wired"]
	tTouch := time.Now()
	q.Run1D(pipe, nPages, 256, buf, out, stride)
	_ = out.Floats()[0] // completion + readback
	touchMS := float64(time.Since(tTouch).Microseconds()) / 1000
	tFirst := time.Now()
	t2first := forkMS() // the FIRST fork after the touch, while the window is (presumably) still wired
	s := vmStat(t)
	t.Logf("P2 GPU touched the 16 MiB window in %.1f ms; FIRST fork+exec right after it (started %.3f): %.2f ms; then wired Δ %+d pages (window = %d, file = %d), system pageins Δ %+d since before the touch",
		touchMS, float64(tFirst.UnixNano())/1e9, t2first, s["wired"]-wBeforeTouch, nPages, fileBytes/page, s["pageins"]-pageinsBefore)
	time.Sleep(2 * time.Second)
	mark("P2 +2 s (idle)")
	t2idle := median3()
	t.Logf("P2 fork+exec 2 s after the touch, GPU idle: %.2f ms", t2idle)

	// P3: hold the window wired by dispatching continuously (what a decode does: a command buffer is always
	// in flight), and time forks DURING that.
	stop := make(chan struct{})
	done := make(chan int)
	go func() {
		n := 0
		for {
			select {
			case <-stop:
				done <- n
				return
			default:
				q.Run1D(pipe, nPages, 256, buf, out, stride)
				n++
			}
		}
	}()
	time.Sleep(300 * time.Millisecond)
	var t3 [3]float64
	var pi [3]int64
	for i := range t3 {
		before := vmStat(t)["pageins"] // (this fork runs while wired too; it is timed separately below)
		start := mark(fmt.Sprintf("P3.%d fork start, dispatch loop running", i+1))
		t3[i] = forkMS()
		pi[i] = vmStat(t)["pageins"] - before
		t.Logf("P3.%d fork+exec DURING continuous dispatch: %.2f ms (system pageins Δ %+d pages; started %.3f)", i+1, t3[i], pi[i], float64(start.UnixNano())/1e9)
		time.Sleep(1 * time.Second)
	}
	close(stop)
	nDispatch := <-done
	t.Logf("P3 dispatch loop: %d dispatches", nDispatch)
	time.Sleep(2 * time.Second)
	t3idle := median3()
	t.Logf("P3 fork+exec 2 s after the loop stopped: %.2f ms", t3idle)
	_ = data[off] // keep the mapping alive across everything above (nil deallocator)

	// P4: release the buffer (Metal's ledger), then unmap; fork again.
	d.ReleaseAll()
	time.Sleep(500 * time.Millisecond)
	tRel := median3()
	t.Logf("P4a fork+exec after ReleaseAll (mapping still present): %.2f ms", tRel)
	_ = mmap.Unmap(data)
	t4 := median3()
	t.Logf("P4b fork+exec after munmap: %.2f ms", t4)

	// P5: the candidate fix — map the same file again, mark the mapping VM_INHERIT_NONE (minherit(2): absent
	// from any child, which is right for weights no child ever needs), wire it by dispatching continuously,
	// and time forks during that. If the eager copy is fork inheriting a wired entry, this makes it vanish.
	data2, err := mmap.MapReadOnly(path)
	if err != nil {
		t.Fatal(err)
	}
	const vmInheritNone = 2 // mach/vm_inherit.h VM_INHERIT_NONE
	if _, _, e := syscall.Syscall(syscall.SYS_MINHERIT, uintptr(unsafe.Pointer(&data2[0])), uintptr(len(data2)), vmInheritNone); e != 0 {
		t.Fatalf("minherit(VM_INHERIT_NONE): %v", e)
	}
	win2 := data2[off : off+windowBytes]
	buf2 := d.NewBufferNoCopy(unsafe.Pointer(&win2[0]), windowBytes)
	out2 := d.NewBufferLen(nPages) // P4a's ReleaseAll freed out/stride; fresh ones (a use-after-free SIGSEGVs in the binding)
	stride2 := gpu.NewBufferOf(d, []uint32{uint32(page / 4)})
	q.Run1D(pipe, nPages, 256, buf2, out2, stride2)
	_ = out2.Floats()[0]
	t5first := forkMS()
	t.Logf("P5 minherit(NONE) mapping: FIRST fork+exec right after the GPU touch: %.2f ms", t5first)
	stop2 := make(chan struct{})
	done2 := make(chan int)
	go func() {
		n := 0
		for {
			select {
			case <-stop2:
				done2 <- n
				return
			default:
				q.Run1D(pipe, nPages, 256, buf2, out2, stride2)
				n++
			}
		}
	}()
	time.Sleep(300 * time.Millisecond)
	var t5 [3]float64
	for i := range t5 {
		t5[i] = forkMS()
		time.Sleep(500 * time.Millisecond)
	}
	close(stop2)
	n2 := <-done2
	t.Logf("P5 minherit(NONE) mapping: fork+exec DURING continuous dispatch (%d dispatches): %.2f/%.2f/%.2f ms", n2, t5[0], t5[1], t5[2])
	_ = data2[off]
	d.ReleaseAll()
	_ = mmap.Unmap(data2)
	if (t5[0]+t5[1]+t5[2])/3 < 5*t0 && t5first < 5*t0 {
		t.Logf("=> minherit(VM_INHERIT_NONE) on the mapping removes the eager copy: forks stay ~%.1f ms while the window is wired.", (t5[0]+t5[1]+t5[2])/3)
	} else {
		t.Logf("=> minherit(VM_INHERIT_NONE) did NOT remove the slow fork (%.0f ms while wired) — the copy is not inheritance-driven.", (t5[0]+t5[1]+t5[2])/3)
	}

	// Sampler: peak Anonymous / wired / pageins across the run, with the sample times, so the transient
	// copy during a slow fork is visible.
	_ = sampler.Process.Kill()
	_ = sampler.Wait()
	sf.Close()
	reportSamples(t, samplePath)

	// Reading, stated in the log; this is a probe, so no assertion decides it.
	filePages := int64(fileBytes / page)
	slowHeld := (t3[0]+t3[1]+t3[2])/3 > 20*t0
	t.Logf("READING: fork ms — no mapping %.2f | buffer untouched %.2f | FIRST after touch %.2f | idle 2 s after touch %.2f | "+
		"during continuous dispatch %.2f/%.2f/%.2f | idle 2 s after loop %.2f | after ReleaseAll %.2f | after munmap %.2f; "+
		"wired at touch %+d of %d file pages; pageins during held forks %+d/%+d/%+d",
		t0, t1, t2first, t2idle, t3[0], t3[1], t3[2], t3idle, tRel, t4, s["wired"]-wBeforeTouch, filePages, pi[0], pi[1], pi[2])
	switch {
	case slowHeld && t3idle < 5*t0:
		t.Logf("=> H1 (refined): a fork made WHILE a no-copy page is wired copies the whole mapping eagerly (seconds, page-ins ≈ the file the first time); forks with the GPU idle are fast because the driver unwires between command buffers. A decode always has a command buffer in flight, so the 2 s swap-guard fork on M26 always pays it: 15 GB per tick.")
	case slowHeld:
		t.Logf("=> fork is slow whenever the buffer exists after its first touch (wire persists while idle too); same fix.")
	case t2first > 20*t0:
		t.Logf("=> only the very first fork after a touch is slow; later ones are not — the wire is released quickly. M26 would still pay it once per dispatch burst.")
	case s["wired"]-wBeforeTouch > filePages/2:
		t.Logf("=> H2: the GPU touch wired ~the whole mapping, not the window.")
	default:
		t.Logf("=> neither: forks stayed fast and wiring stayed window-sized; the collapse needs another explanation.")
	}
}

const forkProbeTouchSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void touch_pages(device const float* buf[[buffer(0)]], device float* out[[buffer(1)]],
    constant uint& strideFloats[[buffer(2)]], uint gid[[thread_position_in_grid]]) {
    out[gid] = buf[(uint)gid*strideFloats];
}
`

// writeUncachedRandom writes n bytes of pseudo-random data with F_NOCACHE so the file's pages are NOT in
// the unified buffer cache afterwards (a later read of them is a real page-in). Random so APFS cannot
// compress or sparse it away.
func writeUncachedRandom(t *testing.T, path string, n int64) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, _, e := syscall.Syscall(syscall.SYS_FCNTL, f.Fd(), syscall.F_NOCACHE, 1); e != 0 {
		t.Fatalf("F_NOCACHE: %v", e)
	}
	chunk := make([]byte, 16<<20)
	x := uint64(0x9E3779B97F4A7C15)
	t0 := time.Now()
	for written := int64(0); written < n; written += int64(len(chunk)) {
		for i := 0; i+8 <= len(chunk); i += 8 {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			chunk[i] = byte(x)
			chunk[i+1] = byte(x >> 8)
			chunk[i+2] = byte(x >> 16)
			chunk[i+3] = byte(x >> 24)
			chunk[i+4] = byte(x >> 32)
			chunk[i+5] = byte(x >> 40)
			chunk[i+6] = byte(x >> 48)
			chunk[i+7] = byte(x >> 56)
		}
		if _, err := f.Write(chunk); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %d MiB uncached random to %s in %.1f s", n>>20, path, time.Since(t0).Seconds())
}

// vmStat returns the vm_stat counters this probe reads, in pages.
func vmStat(t *testing.T) map[string]int64 {
	t.Helper()
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		t.Fatalf("vm_stat: %v", err)
	}
	m := map[string]int64{}
	sc := bufio.NewScanner(strings.NewReader(string(out)))
	for sc.Scan() {
		line := sc.Text()
		i := strings.Index(line, ":")
		if i < 0 {
			continue
		}
		v, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(line[i+1:]), "."), 10, 64)
		if err != nil {
			continue
		}
		switch {
		case strings.HasPrefix(line, "Pages free"):
			m["free"] = v
		case strings.HasPrefix(line, "Pages wired down"):
			m["wired"] = v
		case strings.HasPrefix(line, "Anonymous pages"):
			m["anon"] = v
		case strings.HasPrefix(line, "File-backed pages"):
			m["file"] = v
		case strings.HasPrefix(line, "Pages occupied by compressor"):
			m["compressor"] = v
		case strings.HasPrefix(line, "Pageins"):
			m["pageins"] = v
		}
	}
	return m
}

func vmStatLine(t *testing.T) string {
	m := vmStat(t)
	return fmt.Sprintf("free=%d wired=%d anon=%d file=%d compressor=%d pageins=%d", m["free"], m["wired"], m["anon"], m["file"], m["compressor"], m["pageins"])
}

// reportSamples prints, from the external sampler's log, the baseline and the peak of each counter with
// the time it was seen, so a transient copy during a fork shows even though the child frees it at exec.
func reportSamples(t *testing.T, path string) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Logf("sampler log: %v", err)
		return
	}
	defer f.Close()
	type sample struct {
		at   float64
		vals map[string]int64
	}
	var samples []sample
	var cur *sample
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "T ") {
			at, _ := strconv.ParseFloat(strings.TrimSpace(line[2:]), 64)
			samples = append(samples, sample{at: at, vals: map[string]int64{}})
			cur = &samples[len(samples)-1]
			continue
		}
		if cur == nil {
			continue
		}
		i := strings.Index(line, ":")
		if i < 0 {
			continue
		}
		v, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(line[i+1:]), "."), 10, 64)
		if err != nil {
			continue
		}
		switch {
		case strings.HasPrefix(line, "Pages free"):
			cur.vals["free"] = v
		case strings.HasPrefix(line, "Pages wired down"):
			cur.vals["wired"] = v
		case strings.HasPrefix(line, "Anonymous pages"):
			cur.vals["anon"] = v
		case strings.HasPrefix(line, "File-backed pages"):
			cur.vals["file"] = v
		case strings.HasPrefix(line, "Pages occupied by compressor"):
			cur.vals["compressor"] = v
		case strings.HasPrefix(line, "Pageins"):
			cur.vals["pageins"] = v
		}
	}
	if len(samples) < 2 {
		t.Logf("sampler: %d samples", len(samples))
		return
	}
	t.Logf("sampler: %d samples over %.1f s (every ~%.0f ms)", len(samples), samples[len(samples)-1].at-samples[0].at,
		1000*(samples[len(samples)-1].at-samples[0].at)/float64(len(samples)-1))
	for _, k := range []string{"anon", "wired", "file", "compressor", "free"} {
		base := samples[0].vals[k]
		peak, at, low, atLow := base, samples[0].at, base, samples[0].at
		for _, s := range samples {
			if s.vals[k] > peak {
				peak, at = s.vals[k], s.at
			}
			if s.vals[k] < low {
				low, atLow = s.vals[k], s.at
			}
		}
		t.Logf("  %-10s baseline %9d  peak %9d (%+d pages = %+.2f GB) @%.3f   low %9d (%+d) @%.3f",
			k, base, peak, peak-base, float64(peak-base)*16384/(1<<30), at, low, low-base, atLow)
	}
}
