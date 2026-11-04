package decoder

import (
	"context"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type recordsHandler struct{ records []slog.Record }

func (h *recordsHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *recordsHandler) Handle(_ context.Context, r slog.Record) error {
	h.records = append(h.records, r)
	return nil
}
func (h *recordsHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *recordsHandler) WithGroup(string) slog.Handler      { return h }

// A real load's message reaches the logger, attributed to the line that wrote it, and stderr stays clean.
func TestSetLogger_takesWhatALoadWrites(t *testing.T) {
	h := &recordsHandler{}
	SetLogger(slog.New(h))
	t.Cleanup(func() { SetLogger(nil) })
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	m, err := Load(filepath.Join("..", "testdata", "llama-tiny"), Options{StreamWeights: true})
	os.Stderr = old
	w.Close()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	stderr, _ := io.ReadAll(r)
	if len(stderr) != 0 {
		t.Errorf("with a logger set, stderr got %q", stderr)
	}
	for _, rec := range h.records {
		if !strings.Contains(rec.Message, "--stream-weights ignored") {
			continue
		}
		if rec.Level != slog.LevelWarn {
			t.Errorf("level %v, want warn", rec.Level)
		}
		f, _ := runtime.CallersFrames([]uintptr{rec.PC}).Next()
		if !strings.HasSuffix(f.File, filepath.Join("decoder", "model.go")) {
			t.Errorf("attributed to %s:%d, not the line in decoder/model.go that wrote it", f.File, f.Line)
		}
		return
	}
	t.Errorf("no record of the ignored flag among %d", len(h.records))
}
