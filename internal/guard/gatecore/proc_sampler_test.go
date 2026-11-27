package gatecore

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestParseProcStat(t *testing.T) {
	name, ppid, ok := ParseProcStat("4242 (we(ird) name.test) S 17 4242 4242 0 -1")
	if !ok || name != "we(ird) name.test" || ppid != 17 {
		t.Errorf("ParseProcStat: %q %d %v", name, ppid, ok)
	}
}

func TestProcSampler_findsTheTestBinaryUnderItsParent(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("reads /proc")
	}
	bin, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	if !strings.HasSuffix(filepath.Base(bin), ".test") {
		t.Skipf("test binary %s is not named .test", bin)
	}
	t.Cleanup(SetProcSamplePeriod(20 * time.Millisecond))
	cmd := exec.Command("sh", "-c", `"$0" -test.run='^TestProcSampler_helperSleeps$' -procsampler.helper; true`, bin)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	stop := StartProcSampler(cmd.Process.Pid, func() (string, time.Duration, bool) { return "TestFake/sub", 0, true })
	_ = cmd.Wait()
	p := stop()
	if p.Samples == 0 || p.HWMKB == 0 || p.PID == cmd.Process.Pid {
		t.Fatalf("sampled %+v; want the grandchild's VmHWM", p)
	}
	if p.ByTest["TestFake"] == 0 || len(p.ByTest) != 1 {
		t.Errorf("attribution %v; want the top-level test only", p.ByTest)
	}
}

var procSamplerHelper = flag.Bool("procsampler.helper", false, "run TestProcSampler_helperSleeps as the sampler test's grandchild")

func TestProcSampler_helperSleeps(t *testing.T) {
	if !*procSamplerHelper {
		t.Skip("only as TestProcSampler_findsTheTestBinaryUnderItsParent's grandchild")
	}
	time.Sleep(time.Second)
}
