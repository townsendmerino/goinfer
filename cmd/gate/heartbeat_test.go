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
