package gatecore

import (
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestCellHeartbeatEmitsWhileRunning(t *testing.T) {
	defer SetCellHeartbeatInterval(20 * time.Millisecond)()
	t.Setenv("GOINFER_GATE_HEARTBEAT", "")

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	origErr := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = origErr }()

	res := NewResults()
	res.Cur = "testcell"
	res.LiveDone.Store(7)
	res.LiveLast.Store("TestSomethingSlow")

	stop := StartCellHeartbeat("testcell", res, time.Now())
	time.Sleep(120 * time.Millisecond)
	stop()
	w.Close()
	os.Stderr = origErr

	out, _ := io.ReadAll(r)
	got := string(out)
	for _, want := range []string{"testcell", "7 tests finished", "TestSomethingSlow", "elapsed"} {
		if !strings.Contains(got, want) {
			t.Errorf("heartbeat output missing %q:\n%s", want, got)
		}
	}
	if n := strings.Count(got, "tests finished"); n < 2 {
		t.Errorf("heartbeat fired %d time(s); want repeated emissions:\n%s", n, got)
	}
}

func TestCellHeartbeatCanBeSilenced(t *testing.T) {
	t.Setenv("GOINFER_GATE_HEARTBEAT", "0")
	r, w, _ := os.Pipe()
	origErr := os.Stderr
	os.Stderr = w
	res := NewResults()
	stop := StartCellHeartbeat("quiet", res, time.Now())
	time.Sleep(60 * time.Millisecond)
	stop()
	w.Close()
	os.Stderr = origErr
	out, _ := io.ReadAll(r)
	if len(out) != 0 {
		t.Errorf("GOINFER_GATE_HEARTBEAT=0 still printed:\n%s", out)
	}
}

func TestCellHeartbeatNamesLongestRunningTest(t *testing.T) {
	defer SetCellHeartbeatInterval(20 * time.Millisecond)()
	t.Setenv("GOINFER_GATE_HEARTBEAT", "")

	res := NewResults()
	res.Cur = "cell"
	res.Add(TestEvent{Action: "run", Package: "p", Test: "TestSlowParent"})
	time.Sleep(30 * time.Millisecond)
	res.Add(TestEvent{Action: "run", Package: "p", Test: "TestRecentlyStarted"})
	res.Add(TestEvent{Action: "run", Package: "p", Test: "TestAlreadyDone"})
	res.Add(TestEvent{Action: "pass", Package: "p", Test: "TestAlreadyDone"})

	r, w, _ := os.Pipe()
	origErr := os.Stderr
	os.Stderr = w
	stop := StartCellHeartbeat("cell", res, time.Now())
	time.Sleep(80 * time.Millisecond)
	stop()
	w.Close()
	os.Stderr = origErr
	got, _ := io.ReadAll(r)
	out := string(got)

	if !strings.Contains(out, "in flight: TestSlowParent") {
		t.Errorf("heartbeat must name the longest-running in-flight test:\n%s", out)
	}
	if strings.Contains(out, "in flight: TestRecentlyStarted") {
		t.Errorf("named the newer test instead of the longest-running one:\n%s", out)
	}
	if strings.Contains(out, "in flight: TestAlreadyDone") {
		t.Errorf("a finished test is not in flight — it was not cleared:\n%s", out)
	}

	res.Add(TestEvent{Action: "pass", Package: "p", Test: "TestSlowParent"})
	res.Add(TestEvent{Action: "pass", Package: "p", Test: "TestRecentlyStarted"})
	if name, _, ok := res.InFlight(); ok {
		t.Errorf("InFlight() = %q after every test finished, want none", name)
	}
}
