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

// agxClient is one GPU client of another process: its process and the GPU time it has accumulated, in nanoseconds.
type agxClient struct {
	pid int
	ns  int64
}

// otherGPUTimes is every other process's GPU clients, keyed by registry entry ID, from the GPU driver's per-process clients: on
// macOS the IORegistry's AGXDeviceUserClient entries, whose AppUsage accumulatedGPUTime is the figure Activity Monitor's GPU
// column reads. Keying by entry rather than by process keeps a client that closes from cancelling another client's time. ok is
// false on other platforms, if ioreg fails, or if no client of this process is listed with a GPU time that parses: a test that
// has run GPU work owns one, so its absence means the parse no longer reads the listing (a renamed key, a changed format), and
// every other client would otherwise read 0.
func otherGPUTimes() (map[string]agxClient, bool) {
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
func parseAGXClients(out string, self int) (map[string]agxClient, bool) {
	// ioreg prints each entry's properties in name order (AppUsage before IOUserClientCreator), so the two are matched within an
	// entry, which starts at its "+-o" line and carries its registry ID there.
	id := regexp.MustCompile(`^AGXDeviceUserClient\s+<class AGXDeviceUserClient, id (0x[0-9a-f]+),`)
	creator := regexp.MustCompile(`"IOUserClientCreator" = "pid (\d+),`)
	acc := regexp.MustCompile(`"accumulatedGPUTime"=(\d+)`)
	own, clients := false, map[string]agxClient{}
	for _, entry := range strings.Split(out, "+-o ")[1:] {
		e, c := id.FindStringSubmatch(entry), creator.FindStringSubmatch(entry)
		if e == nil || c == nil {
			continue
		}
		pid, _ := strconv.Atoi(c[1])
		times := acc.FindAllStringSubmatch(entry, -1)
		if pid == self {
			own = own || len(times) > 0
			continue
		}
		cl := agxClient{pid: pid}
		for _, a := range times {
			v, _ := strconv.ParseInt(a[1], 10, 64)
			cl.ns += v
		}
		clients[e[1]] = cl
	}
	return clients, own
}

// otherGPUDelta is the GPU time the clients in b accumulated since a, in total and by process, per client: one new in b counts
// in full. A client in a and gone from b took its time with it, so what it did between the samples is unknown; vanished lists
// those clients' processes, and a caller must not read the interval as quiet when it is not empty.
func otherGPUDelta(a, b map[string]agxClient) (total int64, byPid map[int]int64, vanished []int) {
	byPid = map[int]int64{}
	for id, cl := range b {
		if d := cl.ns - a[id].ns; d > 0 {
			total += d
			byPid[cl.pid] += d
		}
	}
	for id, cl := range a {
		if _, ok := b[id]; !ok {
			vanished = append(vanished, cl.pid)
		}
	}
	return total, byPid, vanished
}

// gpuArm is one otherGPUWindow reading: other processes' GPU time as a fraction of the window's wall time, by process, and the
// processes whose clients closed inside it (their time is unknown).
type gpuArm struct {
	frac     float64
	byPid    map[int]int64
	wall     time.Duration
	vanished []int
}

// otherGPUWindow brackets one timed arm with otherGPUTimes; end returns its gpuArm. A client opened inside the arm counts in full;
// one present at the start and closed before the end is reported in vanished, and the caller voids the arm. What still passes
// unseen is a client opened and closed between the two samples. ok is false where there is no reading; the test fails if the
// reading worked at the start and not at the end. Open it before the CPU window and close it after, so the ioreg call's own CPU
// stays outside the arm's busy-core reading.
func otherGPUWindow(t *testing.T) (end func() (arm gpuArm, ok bool)) {
	t.Helper()
	a, ok := otherGPUTimes()
	t0 := time.Now()
	return func() (gpuArm, bool) {
		if !ok {
			return gpuArm{}, false
		}
		b, ok2 := otherGPUTimes()
		wall := time.Since(t0)
		if !ok2 {
			t.Fatal("the GPU-time reading failed at the end of an arm")
		}
		total, byPid, vanished := otherGPUDelta(a, b)
		return gpuArm{frac: float64(total) / float64(wall.Nanoseconds()), byPid: byPid, wall: wall, vanished: vanished}, true
	}
}

// TestParseAGXClients: the parse keys GPU time by client, skips this process's clients and requires one of them with a GPU time
// that parses; a client that closes between two samples does not cancel another client of the same process.
func TestParseAGXClients(t *testing.T) {
	entryAs := func(id string, pid int, usage string) string {
		return fmt.Sprintf("+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id %s, !registered, active>\n"+
			"    {\n      \"AppUsage\" = (%s)\n      \"IOUserClientCreator\" = \"pid %d, proc\"\n    }\n", id, usage, pid)
	}
	entry := func(id string, pid int, ns ...int64) string {
		var u []string
		for _, n := range ns {
			u = append(u, fmt.Sprintf(`{"API"="Metal","accumulatedGPUTime"=%d}`, n))
		}
		return entryAs(id, pid, strings.Join(u, ","))
	}
	const self = 9
	for _, c := range []struct{ name, out string }{
		{"no client of this process", entry("0x1", 7, 100)},
		{"empty listing", ""},
		{"this process's client with no GPU time", entry("0x1", 7, 100) + entry("0x3", self)},
		{"the GPU-time format changed", entryAs("0x1", 7, `{"accumulatedGPUTime" = 100}`) + entryAs("0x3", self, `{"accumulatedGPUTime" = 5}`)},
	} {
		if _, ok := parseAGXClients(c.out, self); ok {
			t.Errorf("%s: ok, want not ok", c.name)
		}
	}
	a, ok := parseAGXClients(entry("0x1", 7, 100, 50)+entry("0x2", 7, 1000)+entry("0x3", self, 5), self)
	if !ok || a["0x1"].ns != 150 || a["0x2"].ns != 1000 || len(a) != 2 {
		t.Fatalf("parse: %v, %v; want 0x1: 150, 0x2: 1000, this process's client skipped", a, ok)
	}
	// Client 0x2 closes; 0x1 runs 400 ns; a new client 0x4 of another process runs 30 ns.
	b, _ := parseAGXClients(entry("0x1", 7, 500, 50)+entry("0x4", 8, 30)+entry("0x3", self, 900), self)
	total, by, vanished := otherGPUDelta(a, b)
	if total != 430 || by[7] != 400 || by[8] != 30 {
		t.Errorf("delta %d ns (%v), want 430: 400 for pid 7 and 30 for the new pid 8; the closed 0x2 cancels nothing", total, by)
	}
	if len(vanished) != 1 || vanished[0] != 7 {
		t.Errorf("vanished %v, want [7]: client 0x2 closed between the samples, and what it ran is unknown", vanished)
	}
	if _, _, v := otherGPUDelta(a, a); len(v) != 0 {
		t.Errorf("the same sample twice: vanished %v, want none", v)
	}
}

// TestOtherGPUWindow_seesAnotherProcess: otherGPUWindow sees a second process's GPU work. This process opens its own GPU client
// with GPU time first, then starts a child (this binary under GOINFER_GPUOTHERS_CHILD=1) that decodes llama-tiny on WebGPU until
// it is killed at cleanup, waits until the child's client shows GPU time, and reads a 3-second otherGPUWindow: the child's own
// clients must account for at least 5% of its wall time, so another process's load cannot satisfy it, and no client may vanish.
// It skips only where, with this process's own GPU work running, the registry lists no Apple GPU client (a virtualised GPU). macOS
// only.
func TestOtherGPUWindow_seesAnotherProcess(t *testing.T) {
	if os.Getenv("GOINFER_GPUOTHERS_CHILD") == "1" {
		run, done := gpuOthersLoad(t)
		defer done()
		run(2 * time.Minute) // until the parent kills it
		return
	}
	if runtime.GOOS != "darwin" {
		t.Skip("no per-process GPU time on this platform")
	}
	newOrSkipHW(t).Close()
	run, done := gpuOthersLoad(t) // this process's own client, with a GPU time, kept open through the test
	defer done()
	run(300 * time.Millisecond)
	if out, err := exec.Command("ioreg", "-r", "-c", "AGXDeviceUserClient").Output(); err != nil || !strings.Contains(string(out), "AGXDeviceUserClient") {
		t.Skipf("no AGXDeviceUserClient entries with this process's GPU work running (a virtualised GPU, as on CI's macOS runners): %v", err)
	}
	if _, ok := otherGPUTimes(); !ok {
		t.Fatal("the GPU-time reading fails with this process's own GPU work listed")
	}
	cmd := exec.Command(os.Args[0], "-test.run", "^TestOtherGPUWindow_seesAnotherProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), "GOINFER_GPUOTHERS_CHILD=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	child := cmd.Process.Pid
	for deadline := time.Now().Add(15 * time.Second); ; {
		end := otherGPUWindow(t)
		time.Sleep(200 * time.Millisecond)
		if arm, _ := end(); arm.byPid[child] > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the child's GPU work never showed in the registry")
		}
	}
	end := otherGPUWindow(t)
	time.Sleep(3 * time.Second)
	arm, ok := end()
	if !ok {
		t.Fatal("no GPU-time reading")
	}
	childFrac := float64(arm.byPid[child]) / float64(arm.wall.Nanoseconds())
	if childFrac < 0.05 || len(arm.vanished) != 0 {
		t.Fatalf("the child decoded on the GPU through the window: its clients read %.4f of the wall time (all others %.4f), vanished %v; want at least 0.05 and none vanished",
			childFrac, arm.frac, arm.vanished)
	}
	t.Logf("the child's GPU time over the window: %.1f%% of its wall time (all other processes %.1f%%)", 100*childFrac, 100*arm.frac)
}

// gpuOthersLoad loads llama-tiny's WebGPU resident; run decodes on it for d, and done closes it.
func gpuOthersLoad(t *testing.T) (run func(d time.Duration), done func()) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		m.Close()
		t.Fatalf("not a WebGPU resident: %s", m.ResidentDecline())
	}
	_, _, _, _, _, _, V := m.Dims()
	run = func(d time.Duration) {
		for stop, pos := time.Now().Add(d), 0; time.Now().Before(stop); pos = (pos + 1) % 64 {
			if pos == 0 {
				rd.Reset()
			}
			if _, err := rd.Forward(m.EmbedResidentForTest((pos*37+5)%V), pos); err != nil {
				t.Fatal(err)
			}
		}
	}
	return run, func() { m.Close() }
}
