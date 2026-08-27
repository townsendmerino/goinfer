package decoder

import (
	"io"
	"os"
	"regexp"
	"runtime"
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

func TestHumanBytes(t *testing.T) {
	for _, c := range []struct {
		n    int64
		want string
	}{
		{162_700_000_000, "151.5GB"},
		{5 << 20, "5MB"},
		{2048, "2KB"},
	} {
		if got := humanBytes(c.n); got != c.want {
			t.Errorf("humanBytes(%d) = %s, want %s", c.n, got, c.want)
		}
	}
}

// The io= field is Linux-only. Assert the contract in BOTH directions rather than only where this
// happens to run: on the Linux box it must actually produce a number (that box is where the 162GB
// loads happen and where the field earns its keep), and on darwin it must degrade silently rather
// than print a zero that reads like "no progress".
func TestIOProgressMatchesPlatform(t *testing.T) {
	p := newProgress(t, "io", 0)
	total, _, ok := p.ioProgress(time.Now())
	if runtime.GOOS == "linux" {
		if !ok || total == "" {
			t.Errorf("on linux ioProgress must report bytes; got ok=%v total=%q", ok, total)
		}
		return
	}
	if ok {
		t.Errorf("off linux ioProgress must report nothing; got ok=%v total=%q", ok, total)
	}
}

// emit() calls rate() before ioProgress(), and rate() stamps its own window. When both read the
// same timestamp field, ioProgress always saw dt == 0 and silently never printed a rate — the
// total appeared, the rate never did, and it took a 162GB load to notice. The windows are separate
// now, and this drives the arithmetic with known values so the regression cannot return unseen.
func TestIORateSurvivesTheItemRateWindow(t *testing.T) {
	p := newProgress(t, "io-rate", 0)
	t0 := time.Now()
	if _, rate, _ := p.ioProgressFrom(t0, 1<<30); rate != "" {
		t.Errorf("first sample cannot have a rate, got %q", rate)
	}
	// The interleaving that broke it: rate() stamps lastAt between the two io samples.
	p.Step(1)
	_, _ = p.rate(t0.Add(30 * time.Second))
	total, rate, ok := p.ioProgressFrom(t0.Add(30*time.Second), 1<<30+30*(1<<20))
	if !ok {
		t.Fatal("ioProgressFrom returned !ok")
	}
	if rate == "" {
		t.Error("io rate was suppressed — the io window is sharing rate()'s timestamp again")
	}
	if want := "1MB/s"; rate != want {
		t.Errorf("rate = %q, want %q (30MB over 30s)", rate, want)
	}
	if total == "" {
		t.Error("cumulative total must always print")
	}
}

// The ETA is built from the recent rate. That matters when counted work follows a long UNCOUNTED
// phase: the qwen3next oracle spends ~13 minutes loading an 80B checkpoint before its first
// countable token, and a cumulative estimate divided that whole span by one finished item and
// announced eta=2h20m for work that took six minutes. The recent window has to dominate.
func TestRateTracksRecentWindowNotAllHistory(t *testing.T) {
	p := newProgress(t, "eta", 11)
	t0 := time.Now()
	p.Step(1)
	p.ratePerMin(t0) // establish the window

	p.Step(1)
	slow, ok := p.ratePerMin(t0.Add(14 * time.Minute)) // 1 item across the load
	if !ok {
		t.Fatal("no rate after the slow interval")
	}
	p.Step(2)
	fast, ok := p.ratePerMin(t0.Add(15 * time.Minute)) // 2 items in the next minute
	if !ok {
		t.Fatal("no rate after the fast interval")
	}
	if fast <= slow*10 {
		t.Errorf("recent rate %.3f/min barely moved from %.3f/min — the window is averaging over "+
			"all history, so the ETA will stay poisoned by the load phase", fast, slow)
	}
	if want := 2.0; fast < want*0.9 || fast > want*1.1 {
		t.Errorf("recent rate = %.3f/min, want ~%.1f (2 items in 1 minute)", fast, want)
	}
}
