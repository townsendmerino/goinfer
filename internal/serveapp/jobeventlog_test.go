package serveapp

import (
	"testing"
	"time"
)

// TestJobEventLog_replayFromStart: everything appended before a reader ever shows up must still
// come back on that reader's first snapshot(0) — the "replay" half of J3's re-attach.
func TestJobEventLog_replayFromStart(t *testing.T) {
	l := newJobEventLog()
	l.append([]byte("one"))
	l.append([]byte("two"))
	l.append([]byte("three"))

	events, _, done := l.snapshot(0)
	if done {
		t.Fatal("log reported done before markDone")
	}
	if len(events) != 3 {
		t.Fatalf("got %d events, want 3", len(events))
	}
	for i, want := range []string{"one", "two", "three"} {
		if string(events[i]) != want {
			t.Fatalf("event %d = %q, want %q", i, events[i], want)
		}
	}
}

// TestJobEventLog_partialReplayThenLive: a reader that already has the first N events asks for
// events[N:] only — the "then continues live" half, exercised without any real waiting.
func TestJobEventLog_partialReplayThenLive(t *testing.T) {
	l := newJobEventLog()
	l.append([]byte("a"))
	l.append([]byte("b"))

	events, _, _ := l.snapshot(1)
	if len(events) != 1 || string(events[0]) != "b" {
		t.Fatalf("snapshot(1) = %q, want [\"b\"]", events)
	}

	l.append([]byte("c"))
	events, _, _ = l.snapshot(2)
	if len(events) != 1 || string(events[0]) != "c" {
		t.Fatalf("snapshot(2) after a new append = %q, want [\"c\"]", events)
	}
}

// TestJobEventLog_waiterUnblocksOnAppend is the mechanism's own gate: a reader blocked on the
// wait channel from an EMPTY snapshot must wake the instant append() fires, not on some poll
// interval — proving the channel-swap actually signals rather than the test passing by luck.
func TestJobEventLog_waiterUnblocksOnAppend(t *testing.T) {
	l := newJobEventLog()
	_, wait, done := l.snapshot(0)
	if done {
		t.Fatal("fresh log reported done")
	}

	woke := make(chan struct{})
	go func() {
		<-wait
		close(woke)
	}()

	select {
	case <-woke:
		t.Fatal("waiter woke before any append — false positive")
	case <-time.After(20 * time.Millisecond):
	}

	l.append([]byte("x"))

	select {
	case <-woke:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("waiter did not wake within 200ms of append")
	}
}

// TestJobEventLog_waiterUnblocksOnMarkDone: same gate, for the "job ended with no further events"
// case — a reader must not be left hanging forever just because nothing more was ever appended.
func TestJobEventLog_waiterUnblocksOnMarkDone(t *testing.T) {
	l := newJobEventLog()
	_, wait, _ := l.snapshot(0)

	woke := make(chan struct{})
	go func() {
		<-wait
		close(woke)
	}()

	l.markDone()

	select {
	case <-woke:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("waiter did not wake within 200ms of markDone")
	}

	_, _, done := l.snapshot(0)
	if !done {
		t.Fatal("snapshot should report done after markDone")
	}
}

// TestJobEventLog_noMissedAppendBetweenSnapshotAndWait: the race the "same lock" doc comment
// exists to prevent — an append landing in the narrow window between reading events and reading
// the wait channel must still be visible, either in the events slice itself or by the wait
// channel already being closed. Run under -race to additionally confirm no data race.
func TestJobEventLog_noMissedAppendBetweenSnapshotAndWait(t *testing.T) {
	l := newJobEventLog()
	const n = 200
	go func() {
		for i := 0; i < n; i++ {
			l.append([]byte{byte(i)})
		}
		l.markDone()
	}()

	pos := 0
	seen := 0
	for {
		events, wait, done := l.snapshot(pos)
		seen += len(events)
		pos += len(events)
		if done && pos >= n {
			break
		}
		if len(events) == 0 {
			<-wait
		}
	}
	if seen != n {
		t.Fatalf("read %d events across the whole run, want %d — one was missed", seen, n)
	}
}
