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
)

// TestAliasSharedMappingProbe asks the question the M26 root cause left open
// (docs/measurements/m26-alias-fork-collapse-2026-09-24.md §5): does a no-copy buffer over a MAP_SHARED
// file mapping give S6 what it wanted — GPU-readable weights that stay FILE-BACKED — where MAP_PRIVATE
// cannot (IOKit wires with write intent, so every touched private page becomes a wired anonymous COW copy)?
//
// For each mapping mode, over a fresh uncached 2 GiB file: one no-copy buffer over a 256 MiB window;
// (1) one GPU touch of one float per page, and the values compared with what the CPU reads at the same
// offsets (a read-only shared mapping might not be wireable at all, and a GPU that silently reads zeros
// is the failure to rule out); (2) a 3 s continuous-dispatch phase; (3) 3 s idle. System vm_stat is
// sampled every ~70 ms by a SEPARATE process — never from this one, because a fork of this process
// while a private window is wired copies the whole mapping (the M26 mechanism), which would contaminate
// both the measurement and the machine. Per phase: the change in anonymous, file-backed and wired pages
// against the pre-touch baseline. Fork+exec is timed once during dispatch, deliberately, in each mode.
//
// "Pages copy-on-write" (cumulative COW faults, system-wide) is the discriminating counter: it cannot be
// moved by memory pressure the way anon/file/free are, and the private arm's wire should add ~one per window
// page. Reading: MAP_SHARED with COW Δ ≈ 0 and file Δ ≈ +window, values correct, fork fast ⇒ S6's premise holds
// for a shared mapping. Anon Δ ≈ +window in either mode ⇒ it is a copy. Values wrong ⇒ the mode is unusable.
//
//	GOINFER_ALIAS_SHARED_PROBE=1 go test -tags goinfer_testhooks ./metal/ -run TestAliasSharedMappingProbe -v -count=1
func TestAliasSharedMappingProbe(t *testing.T) {
	if os.Getenv("GOINFER_ALIAS_SHARED_PROBE") == "" {
		t.Skip("set GOINFER_ALIAS_SHARED_PROBE=1 (writes a 2 GiB scratch file; wires up to 256 MiB twice)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const page = 16384
	const fileBytes = int64(2) << 30
	const windowBytes = 256 << 20 // 16,384 pages: large against vm_stat noise
	dir := os.Getenv("GOINFER_ALIAS_FORK_PROBE_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	lib, err := d.CompileLibrary(forkProbeTouchSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "touch_pages")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()

	for _, mode := range []struct {
		name  string
		flags int
	}{{"MAP_PRIVATE", syscall.MAP_PRIVATE}, {"MAP_SHARED", syscall.MAP_SHARED}} {
		t.Run(mode.name, func(t *testing.T) {
			path := filepath.Join(dir, "sharedprobe.bin")
			writeUncachedRandom(t, path, fileBytes) // fresh and uncached for each mode
			defer os.Remove(path)

			samplePath := filepath.Join(dir, "sharedprobe-"+mode.name+".log")
			stopSampler := startVMStatSampler(t, samplePath)
			taskPath := filepath.Join(dir, "sharedprobe-task-"+mode.name+".log")
			stopTask := startTaskSampler(t, taskPath, os.Getpid())
			time.Sleep(600 * time.Millisecond)

			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			data, err := syscall.Mmap(int(f.Fd()), 0, int(fileBytes), syscall.PROT_READ, mode.flags)
			f.Close()
			if err != nil {
				t.Fatalf("mmap %s: %v", mode.name, err)
			}
			defer syscall.Munmap(data)
			off := (len(data) / 2) &^ (page - 1)
			win := data[off : off+windowBytes]
			buf := d.NewBufferNoCopy(unsafe.Pointer(&win[0]), windowBytes)
			const nPages = windowBytes / page
			out := d.NewBufferLen(nPages)
			stride := gpu.NewBufferOf(d, []uint32{uint32(page / 4)})

			marks := map[string]time.Time{}
			marks["baseline-end"] = time.Now()
			time.Sleep(300 * time.Millisecond)

			// (1) one touch, then correctness.
			tTouch := time.Now()
			q.Run1D(pipe, nPages, 256, buf, out, stride)
			got := append([]float32(nil), out.Floats()[:nPages]...)
			touchMS := float64(time.Since(tTouch).Microseconds()) / 1000
			marks["touch-end"] = time.Now()
			wrong := 0
			for i := range nPages {
				want := *(*float32)(unsafe.Pointer(&win[i*page]))
				if got[i] != want && !(got[i] != got[i] && want != want) { // NaN == NaN for this purpose
					wrong++
				}
			}
			t.Logf("%s: GPU touch of %d pages took %.1f ms; %d/%d values differ from the CPU's read", mode.name, nPages, touchMS, wrong, nPages)
			if wrong > 0 {
				t.Errorf("%s: the GPU read %d wrong values through the no-copy buffer — this mode is not usable", mode.name, wrong)
			}
			time.Sleep(1 * time.Second)
			marks["after-touch-end"] = time.Now()

			// (2) continuous dispatch, with one timed fork+exec in the middle.
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
			marks["dispatch-start"] = time.Now()
			time.Sleep(1500 * time.Millisecond)
			forkMS := -1.0 // MAP_PRIVATE's fork cost is known (1.1-2.5 s, a whole-mapping copy); not repeated
			if mode.flags == syscall.MAP_SHARED {
				tf := time.Now()
				if err := exec.Command("/usr/bin/true").Run(); err != nil {
					t.Fatal(err)
				}
				forkMS = float64(time.Since(tf).Microseconds()) / 1000
			}
			time.Sleep(1500 * time.Millisecond)
			close(stop)
			nDisp := <-done
			marks["dispatch-end"] = time.Now()
			t.Logf("%s: %d dispatches; fork+exec during dispatch: %.1f ms", mode.name, nDisp, forkMS)

			// (3) idle.
			time.Sleep(3 * time.Second)
			marks["idle-end"] = time.Now()
			_ = win[0]
			d.ReleaseBuf(buf)
			stopSampler()
			stopTask()

			phases := []struct{ name, from, to string }{
				{"baseline", "", "baseline-end"},
				{"after one touch (idle 1 s)", "touch-end", "after-touch-end"},
				{"during continuous dispatch", "dispatch-start", "dispatch-end"},
				{"idle 0-3 s after dispatch", "dispatch-end", "idle-end"},
			}
			// THIS process's own COW faults (task->cow_faults, charged to the task whose thread takes the fault —
			// the one that commits the command buffer and so runs the driver's wire). Desktop background cannot
			// move it. Reported as the change from the last pre-touch sample to the last sample of each phase.
			tm := phaseLast(t, taskPath, marks, phases)
			for _, p := range phases[1:] {
				t.Logf("%s | %-28s THIS PROCESS: COW faults %+7.0f  faults %+7.0f  page-ins %+7.0f  (window = %d pages)",
					mode.name, p.name, tm[p.name]["cow"]-tm["baseline"]["cow"], tm[p.name]["faults"]-tm["baseline"]["faults"], tm[p.name]["pageins"]-tm["baseline"]["pageins"], nPages)
			}
			means := phaseMeans(t, samplePath, marks, phases)
			base := means["baseline"]
			windowPages := float64(nPages)
			for _, p := range phases[1:] {
				m := means[p.name]
				t.Logf("%s | %-28s COW faults %+8.0f  wired %+8.0f  anon %+8.0f  file %+8.0f  free %+8.0f pages  (window = %.0f pages)",
					mode.name, p.name, m["cow"]-base["cow"], m["wired"]-base["wired"], m["anon"]-base["anon"], m["file"]-base["file"], m["free"]-base["free"], windowPages)
			}
		})
	}
}

// startVMStatSampler runs vm_stat in a loop in a SEPARATE process, one line per sample:
// "<unix time> anon=<n> file=<n> wired=<n> free=<n>". Returns a stop function.
func startVMStatSampler(t *testing.T, path string) func() {
	t.Helper()
	script := `while :; do vm_stat | awk -v t="$(perl -MTime::HiRes=time -e 'printf "%.3f", time')" '
		/Anonymous pages/{a=$3} /File-backed pages/{f=$3} /Pages wired down/{w=$4} /Pages free/{fr=$3} /copy-on-write/{c=$NF}
		END{gsub("\\.","",a); gsub("\\.","",f); gsub("\\.","",w); gsub("\\.","",fr); gsub("\\.","",c); print t, "anon=" a, "file=" f, "wired=" w, "free=" fr, "cow=" c}'; sleep 0.05; done`
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("/bin/sh", "-c", script)
	cmd.Stdout = f
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	return func() { _ = cmd.Process.Kill(); _ = cmd.Wait(); f.Close() }
}

// phaseMeans averages each counter over the samples whose time falls in [marks[from], marks[to]] (from ""
// = the log's start).
func phaseMeans(t *testing.T, path string, marks map[string]time.Time, phases []struct{ name, from, to string }) map[string]map[string]float64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	type sample struct {
		at   float64
		vals map[string]float64
	}
	var samples []sample
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fs := strings.Fields(sc.Text())
		if len(fs) < 5 {
			continue
		}
		at, err := strconv.ParseFloat(fs[0], 64)
		if err != nil {
			continue
		}
		s := sample{at: at, vals: map[string]float64{}}
		for _, kv := range fs[1:] {
			k, v, ok := strings.Cut(kv, "=")
			if !ok {
				continue
			}
			x, err := strconv.ParseFloat(v, 64)
			if err == nil {
				s.vals[k] = x
			}
		}
		samples = append(samples, s)
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i].at < samples[j].at })
	out := map[string]map[string]float64{}
	for _, p := range phases {
		lo := 0.0
		if p.from != "" {
			lo = float64(marks[p.from].UnixNano()) / 1e9
		}
		hi := float64(marks[p.to].UnixNano()) / 1e9
		sum := map[string]float64{}
		n := 0
		for _, s := range samples {
			if s.at >= lo && s.at <= hi {
				for k, v := range s.vals {
					sum[k] += v
				}
				n++
			}
		}
		if n == 0 {
			t.Fatalf("phase %q: no vm_stat samples between the marks", p.name)
		}
		m := map[string]float64{}
		for k, v := range sum {
			m[k] = v / float64(n)
		}
		out[p.name] = m
		_ = fmt.Sprint(n)
	}
	return out
}

// startTaskSampler samples one process's cumulative cow/faults/pageins with top, from a SEPARATE process
// (so the process under test never forks): "<unix time> cow=<n> faults=<n> pageins=<n>".
func startTaskSampler(t *testing.T, path string, pid int) func() {
	t.Helper()
	script := fmt.Sprintf(`while :; do top -l 1 -pid %d -stats pid,cow,faults,pageins 2>/dev/null | awk -v t="$(perl -MTime::HiRes=time -e 'printf "%%.3f", time')" '$1 == "%d" {gsub("[^0-9]","",$2); gsub("[^0-9]","",$3); gsub("[^0-9]","",$4); print t, "cow=" $2, "faults=" $3, "pageins=" $4}'; done`, pid, pid)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("/bin/sh", "-c", script)
	cmd.Stdout = f
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	return func() { _ = cmd.Process.Kill(); _ = cmd.Wait(); f.Close() }
}

// phaseLast returns, per phase, the LAST sample at or before marks[to] (cumulative counters).
func phaseLast(t *testing.T, path string, marks map[string]time.Time, phases []struct{ name, from, to string }) map[string]map[string]float64 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]map[string]float64{}
	for _, p := range phases {
		hi := float64(marks[p.to].UnixNano()) / 1e9
		var best map[string]float64
		bestAt := -1.0
		for _, line := range strings.Split(string(raw), "\n") {
			fs := strings.Fields(line)
			if len(fs) < 4 {
				continue
			}
			at, err := strconv.ParseFloat(fs[0], 64)
			if err != nil || at > hi || at < bestAt {
				continue
			}
			v := map[string]float64{}
			for _, kv := range fs[1:] {
				if k, x, ok := strings.Cut(kv, "="); ok {
					if n, err := strconv.ParseFloat(x, 64); err == nil {
						v[k] = n
					}
				}
			}
			best, bestAt = v, at
		}
		if best == nil {
			t.Fatalf("phase %q: no task sample before its end mark", p.name)
		}
		out[p.name] = best
	}
	return out
}
