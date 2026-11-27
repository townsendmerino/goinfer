package streamframe

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSSEWriter_heartbeatAndHandlerDoNotRace(t *testing.T) {
	rec := httptest.NewRecorder()
	ss := NewSSEWriter(rec, rec)

	old := HeartbeatInterval
	HeartbeatInterval = 50 * time.Microsecond
	defer func() { HeartbeatInterval = old }()

	stop := Heartbeat(ss)
	var wg sync.WaitGroup
	var pinged atomic.Bool
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		for off := 0; !pinged.Load(); time.Sleep(time.Millisecond) {
			ss.Lock()
			b := rec.Body.Bytes()
			pinged.Store(bytes.Contains(b[max(off-len(": ping"), 0):], []byte(": ping")))
			off = len(b)
			ss.Unlock()
		}
	}()
	start := time.Now()
	for i := range 4 {
		wg.Go(func() {
			for j := 0; j < 200 || time.Since(start) < 20*time.Millisecond || (!pinged.Load() && time.Since(start) < 5*time.Second); j++ {
				SendData(ss, map[string]any{"delta": fmt.Sprintf("w%d-%d", i, j)})
			}
		})
	}
	wg.Wait()
	stop()
	stop() // idempotent
	pinged.Store(true)
	<-watchDone

	body := rec.Body.String()
	for line := range strings.SplitSeq(body, "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") && !strings.HasPrefix(line, ": ping") {
			t.Fatalf("torn frame in the stream: %q", line)
		}
		if strings.HasPrefix(line, "data: ") && strings.Contains(line, ": ping") {
			t.Fatalf("heartbeat spliced into a data frame: %q", line)
		}
	}
	if !strings.Contains(body, ": ping") {
		t.Error("no heartbeat frame observed")
	}
}

type blockingWriter struct{ release <-chan struct{} }

func (b blockingWriter) Header() http.Header { return http.Header{} }
func (b blockingWriter) WriteHeader(int)     {}
func (b blockingWriter) Write([]byte) (int, error) {
	return 0, fmt.Errorf("write: connection timed out")
}

type nopFlusher struct{}

func (nopFlusher) Flush() {}

func TestSSEWriter_stalledClientFailsTheWriteInsteadOfBlocking(t *testing.T) {
	old := WriteTimeout
	WriteTimeout = 30 * time.Millisecond
	defer func() { WriteTimeout = old }()

	blocked := make(chan struct{})
	ss := NewSSEWriterRaw(blockingWriter{blocked}, nopFlusher{})
	done := make(chan error, 1)
	go func() { done <- ss.Frame("data: %s\n\n", "x") }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a failing write reported success")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Frame() never returned")
	}
	close(blocked)

	if ss.Err() == nil {
		t.Error("write error is not sticky")
	}
	if err := ss.Frame("data: %s\n\n", "y"); err == nil {
		t.Error("a frame after a failed write reported success")
	}
}

func TestJSONSender_cancelsOnWriteFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var cancelled atomic.Int32
	send := JSONSender(blockingWriter{}, nopFlusher{}, func() {
		cancelled.Store(1)
		cancel()
	})
	send("start", map[string]string{"file": "x"})
	if cancelled.Load() == 0 || ctx.Err() == nil {
		t.Fatal("failed write did not cancel context")
	}

	cancelled.Store(0)
	rec := httptest.NewRecorder()
	send2 := JSONSender(rec, rec, func() { cancelled.Store(1) })
	send2("start", map[string]string{"file": "x"})
	if cancelled.Load() != 0 {
		t.Error("successful write unexpectedly called cancel")
	}
	if !strings.Contains(rec.Body.String(), `"file":"x"`) {
		t.Errorf("frame not written: %q", rec.Body.String())
	}
}

func TestSSEWriter_setsAWriteDeadlineOnARealConnection(t *testing.T) {
	old := WriteTimeout
	WriteTimeout = 40 * time.Millisecond
	defer func() { WriteTimeout = old }()

	errc := make(chan error, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f, ok := w.(http.Flusher)
		if !ok {
			errc <- fmt.Errorf("no flusher")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		ss := NewSSEWriter(w, f)
		big := strings.Repeat("x", 64<<10)
		var err error
		for i := 0; i < 2000 && err == nil; i++ {
			err = ss.Frame("data: %s\n\n", big)
		}
		errc <- err
	}))
	defer srv.Close()

	resp, getErr := http.Get(srv.URL)
	if getErr == nil {
		defer resp.Body.Close()
	}
	select {
	case err := <-errc:
		if getErr != nil && err == nil {
			t.Fatalf("get: %v, yet handler reported no write error", getErr)
		}
		if err == nil {
			t.Skip("socket buffer absorbed every frame on this host")
		}
	case <-time.After(15 * time.Second):
		t.Fatal("handler never returned against a stalled reader")
	}
}
