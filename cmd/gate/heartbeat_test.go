package main

import (
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// The heartbeat exists because `go test -json` reports a cell only when it
// FINISHES, and this gate's realckpt cells run 55-90 minutes. Without it a
// reader cannot distinguish a working gate from a hung one. So the thing worth
// testing is that it actually emits while work is still in flight — a heartbeat
// nobody has watched fire is indistinguishable from no heartbeat.
func TestCellHeartbeatEmitsWhileRunning(t *testing.T) {
	old := cellHeartbeatInterval
	cellHeartbeatInterval = 20 * time.Millisecond
	defer func() { cellHeartbeatInterval = old }()
	t.Setenv("GOINFER_GATE_HEARTBEAT", "") // no override; use the var

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	origErr := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = origErr }()

	res := newResults()
	res.cur = "testcell"
	res.liveDone.Store(7)
	res.liveLast.Store("TestSomethingSlow")

	stop := startCellHeartbeat("testcell", res, time.Now())
	time.Sleep(120 * time.Millisecond) // several intervals
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
		t.Errorf("heartbeat fired %d time(s) over ~6 intervals — it must repeat, not print once:\n%s", n, got)
	}
}

// "0" must silence it — the rule says make the interval configurable rather than
// deleting the heartbeat when it is noisy somewhere.
func TestCellHeartbeatCanBeSilenced(t *testing.T) {
	t.Setenv("GOINFER_GATE_HEARTBEAT", "0")
	r, w, _ := os.Pipe()
	origErr := os.Stderr
	os.Stderr = w
	res := newResults()
	stop := startCellHeartbeat("quiet", res, time.Now())
	time.Sleep(60 * time.Millisecond)
	stop()
	w.Close()
	os.Stderr = origErr
	out, _ := io.ReadAll(r)
	if len(out) != 0 {
		t.Errorf("GOINFER_GATE_HEARTBEAT=0 still printed:\n%s", out)
	}
}

// The count of finished tests answers "is it moving?" but not "what is holding it up?" — and
// during the v0.15.0 sweep those diverged: the count sat at 430 for four minutes while the line
// kept naming a test that had already completed. So the heartbeat must name what is IN FLIGHT,
// and must prefer the longest-running one, because a slow parent's subtests churn beneath it and
// the parent is the name worth printing.
func TestCellHeartbeatNamesLongestRunningTest(t *testing.T) {
	old := cellHeartbeatInterval
	cellHeartbeatInterval = 20 * time.Millisecond
	defer func() { cellHeartbeatInterval = old }()
	t.Setenv("GOINFER_GATE_HEARTBEAT", "")

	res := newResults()
	res.cur = "cell"
	// Two tests started, the slow one first; a third started and already finished.
	res.add(testEvent{Action: "run", Package: "p", Test: "TestSlowParent"})
	time.Sleep(30 * time.Millisecond)
	res.add(testEvent{Action: "run", Package: "p", Test: "TestRecentlyStarted"})
	res.add(testEvent{Action: "run", Package: "p", Test: "TestAlreadyDone"})
	res.add(testEvent{Action: "pass", Package: "p", Test: "TestAlreadyDone"})

	r, w, _ := os.Pipe()
	origErr := os.Stderr
	os.Stderr = w
	stop := startCellHeartbeat("cell", res, time.Now())
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

	// And once everything terminal-s, there is nothing in flight to report.
	res.add(testEvent{Action: "pass", Package: "p", Test: "TestSlowParent"})
	res.add(testEvent{Action: "pass", Package: "p", Test: "TestRecentlyStarted"})
	if name, _, ok := res.inFlight(); ok {
		t.Errorf("inFlight() = %q after every test finished, want none", name)
	}
}
