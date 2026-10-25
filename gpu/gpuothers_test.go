//go:build gpu

package gpu

import (
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

// otherGPUTimes is the GPU time each process other than this one has accumulated, in nanoseconds, from the GPU driver's
// per-process clients: on macOS the IORegistry's AGXDeviceUserClient entries, whose AppUsage accumulatedGPUTime is the figure
// Activity Monitor's GPU column reads. ok is false on other platforms or if ioreg fails.
func otherGPUTimes() (map[int]int64, bool) {
	if runtime.GOOS != "darwin" {
		return nil, false
	}
	out, err := exec.Command("ioreg", "-r", "-c", "AGXDeviceUserClient", "-l").Output()
	if err != nil {
		return nil, false
	}
	// ioreg prints each entry's properties in name order (AppUsage before IOUserClientCreator), so the two are matched within an
	// entry, which starts at its "+-o" line.
	creator := regexp.MustCompile(`"IOUserClientCreator" = "pid (\d+),`)
	acc := regexp.MustCompile(`"accumulatedGPUTime"=(\d+)`)
	self, times := os.Getpid(), map[int]int64{}
	for _, entry := range strings.Split(string(out), "+-o ")[1:] {
		m := creator.FindStringSubmatch(entry)
		if m == nil {
			continue
		}
		pid, _ := strconv.Atoi(m[1])
		if pid == self {
			continue
		}
		for _, a := range acc.FindAllStringSubmatch(entry, -1) {
			v, _ := strconv.ParseInt(a[1], 10, 64)
			times[pid] += v
		}
	}
	return times, true
}

// otherGPUWindow brackets one timed arm with otherGPUTimes: end returns the GPU time every other process accumulated during the
// arm as a fraction of its wall time (a process that started inside the arm counts in full). ok is false where there is no
// reading; the test fails if the reading worked at the start and not at the end.
func otherGPUWindow(t *testing.T) (end func() (frac float64, ok bool)) {
	t.Helper()
	a, ok := otherGPUTimes()
	t0 := time.Now()
	return func() (float64, bool) {
		if !ok {
			return 0, false
		}
		b, ok2 := otherGPUTimes()
		wall := time.Since(t0)
		if !ok2 {
			t.Fatal("the GPU-time reading failed at the end of an arm")
		}
		var ns int64
		for pid, v := range b {
			if d := v - a[pid]; d > 0 {
				ns += d
			}
		}
		return float64(ns) / float64(wall.Nanoseconds()), true
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
	if _, ok := otherGPUTimes(); !ok {
		t.Skip("no per-process GPU time on this platform")
	}
	newOrSkipHW(t).Close()
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
