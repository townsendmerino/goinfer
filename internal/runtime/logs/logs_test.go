package logs

import (
	"context"
	"io"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"testing"
)

// captureStderr returns what f writes to os.Stderr, swapped as the tests that read a banner swap it.
func captureStderr(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	defer func() { os.Stderr = old }()
	f()
	w.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// capture is a slog handler that keeps its records. minLevel is the lowest it wants.
type capture struct {
	minLevel slog.Level
	records  []slog.Record
}

func (c *capture) Enabled(_ context.Context, l slog.Level) bool { return l >= c.minLevel }
func (c *capture) Handle(_ context.Context, r slog.Record) error {
	c.records = append(c.records, r)
	return nil
}
func (c *capture) WithAttrs([]slog.Attr) slog.Handler { return c }
func (c *capture) WithGroup(string) slog.Handler      { return c }

func useLogger(t *testing.T, h slog.Handler) {
	t.Helper()
	Set(slog.New(h))
	t.Cleanup(func() { Set(nil) })
}

func TestDefault_isTheTextOnStderr(t *testing.T) {
	got := captureStderr(t, func() {
		Debugf("debug %d", 1)
		Infof("info %s", "two")
		Warnf("warn 100%%")
		InfoText("a line\nand another\n")
		WarnText("no newline at the end")
		DebugText("")
	})
	const want = "debug 1\ninfo two\nwarn 100%\na line\nand another\nno newline at the end"
	if got != want {
		t.Errorf("stderr %q, want %q", got, want)
	}
}

func TestLogger_getsLevelTextAndCaller(t *testing.T) {
	c := &capture{minLevel: slog.LevelDebug}
	useLogger(t, c)
	got := captureStderr(t, func() {
		Debugf("debug %d", 1)
		Infof("info")
		Warnf("warn")
		WarnText("a line\nand another\n\n")
		InfoText("\n")
	})
	if got != "" {
		t.Errorf("with a logger set, stderr got %q", got)
	}
	want := []struct {
		level slog.Level
		msg   string
	}{{slog.LevelDebug, "debug 1"}, {slog.LevelInfo, "info"}, {slog.LevelWarn, "warn"}, {slog.LevelWarn, "a line\nand another"}}
	if len(c.records) != len(want) {
		t.Fatalf("%d records, want %d", len(c.records), len(want))
	}
	for i, w := range want {
		r := c.records[i]
		if r.Level != w.level || r.Message != w.msg {
			t.Errorf("record %d: %v %q, want %v %q", i, r.Level, r.Message, w.level, w.msg)
		}
		f, _ := runtime.CallersFrames([]uintptr{r.PC}).Next()
		if !strings.HasSuffix(f.File, "logs_test.go") {
			t.Errorf("record %d is attributed to %s:%d, not its caller", i, f.File, f.Line)
		}
	}
}

// A debug line on a hot path must cost nothing for a logger that drops it.
func TestLogger_droppedLevelIsNotFormatted(t *testing.T) {
	c := &capture{minLevel: slog.LevelWarn}
	useLogger(t, c)
	formatted := false
	Debugf("%v", formatter(func() { formatted = true }))
	Infof("%v", formatter(func() { formatted = true }))
	if formatted {
		t.Error("a level the logger drops was formatted")
	}
	DebugText("dropped")
	Warnf("kept")
	if len(c.records) != 1 || c.records[0].Message != "kept" {
		t.Errorf("records %v, want only the warning", c.records)
	}
}

type formatter func()

func (f formatter) String() string { f(); return "" }

func TestSet_nilRestoresTheDefault(t *testing.T) {
	c := &capture{}
	Set(slog.New(c))
	Set(nil)
	if got := captureStderr(t, func() { Infof("back") }); got != "back\n" {
		t.Errorf("stderr %q", got)
	}
	if len(c.records) != 0 {
		t.Errorf("the logger got %d records after Set(nil)", len(c.records))
	}
}
