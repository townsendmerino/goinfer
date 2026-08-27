package decoder

import (
	"io"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"
)

// capture swaps os.Stderr for a pipe and returns what fn wrote to it.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stderr
	os.Stderr = w
	fn()
	os.Stderr = orig
	w.Close()
	out, _ := io.ReadAll(r)
	return string(out)
}

// A heartbeat nobody has watched fire is indistinguishable from no heartbeat, so assert on the
// bytes rather than trusting that the ticker was wired up.
func TestProgressLineCarriesClockCountAndRate(t *testing.T) {
	t.Setenv("TEST_PROGRESS_INTERVAL", "20ms")
	got := captureStderr(t, func() {
		p := newProgress(t, "unit", 10)
		p.Step(4)
		time.Sleep(70 * time.Millisecond) // several ticks, so a rate window exists
		p.Step(3)
		time.Sleep(70 * time.Millisecond)
		p.Done()
	})

	// Absolute wall clock: an archived log has to say WHEN, not only how long.
	if !regexp.MustCompile(`\d\d:\d\d:\d\d \[unit\]`).MatchString(got) {
		t.Errorf("no HH:MM:SS timestamp before the label:\n%s", got)
	}
	for _, want := range []string{"[unit]", "elapsed=", "/10", "rate=", "done"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if n := strings.Count(got, "elapsed="); n < 2 {
		t.Errorf("emitted %d line(s); a heartbeat must repeat:\n%s", n, got)
	}
}

// The ETA is only defensible where items cost the same; Uneven() must drop it and keep the count.
func TestProgressUnevenDropsETAKeepsCount(t *testing.T) {
	t.Setenv("TEST_PROGRESS_INTERVAL", "20ms")
	got := captureStderr(t, func() {
		p := newProgress(t, "sweep", 4).Uneven()
		p.Step(1)
		time.Sleep(70 * time.Millisecond)
		p.Done()
	})
	if strings.Contains(got, "eta=") {
		t.Errorf("Uneven() must suppress the eta:\n%s", got)
	}
	if !strings.Contains(got, "1/4") {
		t.Errorf("Uneven() must keep the n/total count:\n%s", got)
	}
}

// "0" silences it — configurable rather than deleted, per the long-test rules.
func TestProgressCanBeSilenced(t *testing.T) {
	t.Setenv("TEST_PROGRESS_INTERVAL", "0")
	got := captureStderr(t, func() {
		p := newProgress(t, "quiet", 3)
		p.Step(1)
		p.Phase("still quiet")
		time.Sleep(40 * time.Millisecond)
		p.Done()
	})
	if got != "" {
		t.Errorf("TEST_PROGRESS_INTERVAL=0 still printed:\n%s", got)
	}
}
