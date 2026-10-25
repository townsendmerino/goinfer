//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// otherGPUTimes is the GPU time each GPU client of another process has accumulated, in nanoseconds, keyed by the client's
// registry entry ID, from the GPU driver's per-process clients: on macOS the IORegistry's AGXDeviceUserClient entries, whose
// AppUsage accumulatedGPUTime is the figure Activity Monitor's GPU column reads. Keying by entry rather than by process keeps a
// client that closes from cancelling another client's time. ok is false on other platforms, if ioreg fails, or if no entry of
// this process is listed: a test that has a GPU device open owns one, so its absence means the parse found nothing.
func otherGPUTimes() (map[string]int64, bool) {
	if runtime.GOOS != "darwin" {
		return nil, false
	}
	out, err := exec.Command("ioreg", "-r", "-c", "AGXDeviceUserClient", "-l").Output()
	if err != nil {
		return nil, false
	}
	return parseAGXClients(string(out), os.Getpid())
}

// parseAGXClients is otherGPUTimes over ioreg's text, with self as this process's pid.
func parseAGXClients(out string, self int) (map[string]int64, bool) {
	// ioreg prints each entry's properties in name order (AppUsage before IOUserClientCreator), so the two are matched within an
	// entry, which starts at its "+-o" line and carries its registry ID there.
	id := regexp.MustCompile(`^AGXDeviceUserClient\s+<class AGXDeviceUserClient, id (0x[0-9a-f]+),`)
	creator := regexp.MustCompile(`"IOUserClientCreator" = "pid (\d+),`)
	acc := regexp.MustCompile(`"accumulatedGPUTime"=(\d+)`)
	own, times := false, map[string]int64{}
	for _, entry := range strings.Split(out, "+-o ")[1:] {
		e, c := id.FindStringSubmatch(entry), creator.FindStringSubmatch(entry)
		if e == nil || c == nil {
			continue
		}
		if pid, _ := strconv.Atoi(c[1]); pid == self {
			own = true
			continue
		}
		var ns int64
		for _, a := range acc.FindAllStringSubmatch(entry, -1) {
			v, _ := strconv.ParseInt(a[1], 10, 64)
			ns += v
		}
		times[e[1]] = ns
	}
	return times, own
}

// otherGPUWindow brackets one timed arm with otherGPUTimes: end returns the GPU time other processes' clients accumulated during
// the arm as a fraction of its wall time. A client opened inside the arm counts in full; one closed inside it is not seen. ok is
// false where there is no reading; the test fails if the reading worked at the start and not at the end. Open it before the
// CPU window and close it after, so the ioreg call's own CPU stays outside the arm's busy-core reading.
func otherGPUWindow(t *testing.T) (end func() (frac float64, ok bool)) {
	t.Helper()
	a, ok := otherGPUTimes()
	t0 := time.Now()
	return func() (float64, bool) {
		if !ok {
			return 0, false
		}
		wall := time.Since(t0)
		b, ok2 := otherGPUTimes()
		if !ok2 {
			t.Fatal("the GPU-time reading failed at the end of an arm")
		}
		return float64(otherGPUDelta(a, b)) / float64(wall.Nanoseconds()), true
	}
}

// otherGPUDelta is the GPU time the clients in b accumulated since a, per client: one new in b counts in full, one gone from b
// counts nothing (and cancels nothing).
func otherGPUDelta(a, b map[string]int64) (ns int64) {
	for id, v := range b {
		if d := v - a[id]; d > 0 {
			ns += d
		}
	}
	return ns
}

// TestParseAGXClients: the parse keys GPU time by client, skips this process's clients and requires one of them; a client that
// closes between two samples does not cancel another client of the same process.
func TestParseAGXClients(t *testing.T) {
	entry := func(id string, pid int, ns ...int64) string {
		var u []string
		for _, n := range ns {
			u = append(u, fmt.Sprintf(`{"API"="Metal","accumulatedGPUTime"=%d}`, n))
		}
		return fmt.Sprintf("+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id %s, !registered, active>\n"+
			"    {\n      \"AppUsage\" = (%s)\n      \"IOUserClientCreator\" = \"pid %d, proc\"\n    }\n", id, strings.Join(u, ","), pid)
	}
	const self = 9
	if _, ok := parseAGXClients(entry("0x1", 7, 100), self); ok {
		t.Error("no client of this process listed: ok, want not ok")
	}
	if _, ok := parseAGXClients("", self); ok {
		t.Error("empty listing: ok, want not ok")
	}
	a, ok := parseAGXClients(entry("0x1", 7, 100, 50)+entry("0x2", 7, 1000)+entry("0x3", self, 5), self)
	if !ok || a["0x1"] != 150 || a["0x2"] != 1000 || len(a) != 2 {
		t.Fatalf("parse: %v, %v; want 0x1: 150, 0x2: 1000, this process's client skipped", a, ok)
	}
	// Client 0x2 closes; 0x1 runs 400 ns; a new client 0x4 of another process runs 30 ns.
	b, _ := parseAGXClients(entry("0x1", 7, 500, 50)+entry("0x4", 8, 30)+entry("0x3", self, 900), self)
	if got := otherGPUDelta(a, b); got != 430 {
		t.Errorf("delta %d ns, want 430 (0x1's 400 and the new 0x4's 30; the closed 0x2 cancels nothing)", got)
	}
}

// TestOtherGPUWindow_seesAnotherProcess: otherGPUWindow reads a second process's GPU work. A child process (this binary, under
// GOINFER_GPUOTHERS_CHILD=1) decodes llama-tiny on WebGPU for 8 seconds; a 3-second window opened 2 seconds in, with the child
// still running at both ends, must read at least 5% of its wall time as other processes' GPU time. A process that exits inside
// the window takes its client, and its GPU time, out of the registry before the window closes, so it is not seen. macOS only.
func TestOtherGPUWindow_seesAnotherProcess(t *testing.T) {
	if os.Getenv("GOINFER_GPUOTHERS_CHILD") == "1" {
		gpuOthersChildLoad(t)
		return
	}
	if runtime.GOOS != "darwin" {
		t.Skip("no per-process GPU time on this platform")
	}
	c := newOrSkipHW(t) // the window needs this process's own GPU client listed
	defer c.Close()
	cmd := exec.Command(os.Args[0], "-test.run", "^TestOtherGPUWindow_seesAnotherProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOINFER_GPUOTHERS_CHILD=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	time.Sleep(2 * time.Second)
	end := otherGPUWindow(t)
	time.Sleep(3 * time.Second)
	frac, ok := end()
	if err := cmd.Wait(); err != nil {
		t.Fatalf("child: %v", err)
	}
	if !ok || frac < 0.05 {
		t.Fatalf("another process decoded on the GPU for 3 s and the window read %.4f of its wall time (ok %v); want at least 0.05", frac, ok)
	}
	t.Logf("other processes' GPU time over the window: %.1f%% of its wall time", 100*frac)
}

// gpuOthersChildLoad is the child's GPU work: llama-tiny's WebGPU resident decoding for 8 seconds.
func gpuOthersChildLoad(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatalf("not a WebGPU resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, V := m.Dims()
	for stop, pos := time.Now().Add(8*time.Second), 0; time.Now().Before(stop); pos = (pos + 1) % 64 {
		if pos == 0 {
			rd.Reset()
		}
		if _, err := rd.Forward(m.EmbedResidentForTest((pos*37+5)%V), pos); err != nil {
			t.Fatal(err)
		}
	}
}
