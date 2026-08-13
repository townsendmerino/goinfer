//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentCloseSettleTime measures how long free VRAM takes to stop rising after Close returns.
//
// WHY THIS EXISTS, and it is the one datum the A12 retraction produced. While disproving the "leak",
// the tracer's tail showed free VRAM still climbing as the process exited — 2.4 -> 4.3 -> 6.2 GiB
// across three 50 ms samples. Close() had returned; the driver had not finished. If that interval is
// long relative to the gap between tests, the next test's Load starts inside it and sees a card that
// is still handing memory back, which reproduces every symptom the CUDA tier shows: passes alone,
// fails in suite, VRAM signature, and no leak anywhere.
//
// Cross-package parallelism is already excluded — gpu_gate.sh passes -p 1 on every CUDA invocation,
// targets the single ./cuda/ package rather than ./cuda/..., and the package contains zero
// t.Parallel() calls. So the tests ARE sequential, and "sequential" is exactly what makes a
// non-instant teardown matter: nothing else is running, but the previous test may not be finished.
//
// It reports rather than asserts. A threshold pulled out of one machine's timing would be the
// stale-constant shape this queue keeps finding; what a stabilisation wait should be, if the tier
// needs one at all, follows from the number rather than preceding it.
func TestResidentCloseSettleTime(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no 7B gguf at %s", gguf)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no device: %v", err)
	}
	probe, err := dev.Primary()
	if err != nil {
		t.Skipf("primary ctx: %v", err)
	}
	free := func() uint64 {
		f, _, e := probe.MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return f
	}

	m, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !m.ResidentActive() {
		_ = m.Close()
		t.Skip("cuda declined this model")
	}
	rf := m.ResidentForwardForTest()
	for step := range 16 {
		if _, e := rf.Forward(m.EmbedResidentForTest(1), step); e != nil {
			t.Fatalf("step %d: forward: %v", step, e)
		}
	}

	beforeClose := free()
	start := time.Now()
	if e := m.Close(); e != nil {
		t.Fatalf("close: %v", e)
	}
	closeReturned := time.Since(start)
	atReturn := free()

	// Poll until free stops rising. "Stopped" is three consecutive identical readings, so a single
	// flat sample mid-release cannot end the measurement early.
	const settleTimeout = 30 * time.Second
	var stable uint64
	var settled time.Duration
	same := 0
	last := atReturn
	for time.Since(start) < settleTimeout {
		time.Sleep(5 * time.Millisecond)
		f := free()
		if f == last {
			same++
			if same >= 3 {
				stable, settled = f, time.Since(start)
				break
			}
			continue
		}
		same, last = 0, f
	}
	if stable == 0 {
		stable, settled = last, time.Since(start)
		t.Logf("WARNING: free VRAM never stabilised within %v — the figure below is a floor, not the settle time", settleTimeout)
	}

	MiB := func(b uint64) float64 { return float64(b) / (1 << 20) }
	t.Logf("free before Close      %10.1f MiB", MiB(beforeClose))
	t.Logf("Close() returned after %v", closeReturned)
	t.Logf("free at Close's return %10.1f MiB  (+%.1f MiB released synchronously)", MiB(atReturn), MiB(atReturn)-MiB(beforeClose))
	t.Logf("free once stable       %10.1f MiB  (+%.1f MiB more, ASYNCHRONOUSLY)", MiB(stable), MiB(stable)-MiB(atReturn))
	t.Logf("SETTLE TIME after Close returned: %v", settled-closeReturned)
	t.Logf("SCOPE: model=qwen2.5-7b-instruct-q4_k_m quant=int4 backend=cuda workload=16-step decode, " +
		"one Close, this box. Says nothing about other models, MoE expert caches, or the staged path.")
}
