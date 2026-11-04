package inproc

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func client(h http.HandlerFunc) *http.Client { return &http.Client{Transport: Transport(h)} }

// TestTransport_requestAndResponse: the handler sees the method, path, query, headers and body the client sent, and the
// client gets the status, headers and body the handler wrote, with no listener anywhere.
func TestTransport_requestAndResponse(t *testing.T) {
	c := client(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" || r.URL.Query().Get("beta") != "true" ||
			r.Header.Get("Authorization") != "Bearer k" || r.Host != "goinfer.local" || r.RequestURI != "/v1/chat/completions?beta=true" {
			t.Errorf("handler saw %s %s host %q uri %q auth %q", r.Method, r.URL, r.Host, r.RequestURI, r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Goinfer-Tools-Dropped", "web_search")
		w.WriteHeader(http.StatusTeapot)
		w.Header().Set("X-Late", "ignored") // after the headers went out
		fmt.Fprintf(w, `{"echo":%q}`, b)
	})
	req, _ := http.NewRequest(http.MethodPost, "http://goinfer.local/v1/chat/completions?beta=true", strings.NewReader("hello"))
	req.Header.Set("Authorization", "Bearer k")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusTeapot || string(got) != `{"echo":"hello"}` || resp.Header.Get("X-Goinfer-Tools-Dropped") != "web_search" || resp.Header.Get("X-Late") != "" {
		t.Errorf("status %d, body %q, headers %v", resp.StatusCode, got, resp.Header)
	}
}

// TestTransport_streamsEachEvent: a server-sent-events handler's frames reach the client one at a time, each before the
// handler writes the next. The handler waits for the client's acknowledgement of a frame before it sends another, so a
// transport that buffered the reply would deadlock here and fail on the timeout.
func TestTransport_streamsEachEvent(t *testing.T) {
	ack := make(chan struct{})
	c := client(func(w http.ResponseWriter, r *http.Request) {
		f, ok := w.(http.Flusher)
		if !ok {
			t.Error("the response writer is not an http.Flusher; serve's stream handlers require one")
			return
		}
		if err := http.NewResponseController(w).SetWriteDeadline(time.Now().Add(time.Second)); err != nil {
			t.Errorf("SetWriteDeadline: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f.Flush()
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: %d\n\n", i)
			f.Flush()
			select {
			case <-ack:
			case <-time.After(5 * time.Second):
				t.Errorf("frame %d was not read before the next was due: the reply is buffered", i)
				return
			}
		}
	})
	resp, err := c.Get("http://x/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	var got []string
	for sc.Scan() {
		if line := sc.Text(); line != "" {
			got = append(got, line)
			ack <- struct{}{}
		}
	}
	if strings.Join(got, "|") != "data: 0|data: 1|data: 2" {
		t.Errorf("frames %q", got)
	}
}

// TestTransport_closingTheBodyCancelsTheHandler: the caller stops reading mid-stream and closes the body; the handler's
// request context ends and its blocked write returns, so a generation stops and its goroutine is not left behind.
func TestTransport_closingTheBodyCancelsTheHandler(t *testing.T) {
	done := make(chan error, 1)
	c := client(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "first")
		for {
			if _, err := w.Write(make([]byte, 1024)); err != nil {
				<-r.Context().Done()
				done <- r.Context().Err()
				return
			}
		}
	})
	resp, err := c.Get("http://x/")
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 5)
	if _, err := io.ReadFull(resp.Body, buf); err != nil || string(buf) != "first" {
		t.Fatalf("read %q, %v", buf, err)
	}
	resp.Body.Close()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("handler's context ended with %v, want Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the handler did not return after the body was closed")
	}
}

// TestTransport_contextCancelledBeforeTheReply: the caller's context ends while the handler has written nothing; Do
// returns the context's error and the handler's context ends too.
func TestTransport_contextCancelledBeforeTheReply(t *testing.T) {
	ended := make(chan struct{})
	c := client(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
		close(ended)
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://x/", nil)
	if _, err := c.Do(req); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Do = %v, want the deadline's error", err)
	}
	select {
	case <-ended:
	case <-time.After(5 * time.Second):
		t.Fatal("the handler's context did not end")
	}
}

// TestTransport_edgesOfAHandlersReply: a handler that writes nothing answers 200 with an empty body; a Write with no
// WriteHeader is a 200; a panic before any write is Do's error; a panic after a write ends the body with an error, and
// neither takes the test process down.
func TestTransport_edgesOfAHandlersReply(t *testing.T) {
	resp, err := client(func(http.ResponseWriter, *http.Request) {}).Get("http://x/")
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := io.ReadAll(resp.Body); resp.StatusCode != 200 || len(b) != 0 {
		t.Errorf("silent handler: status %d, body %q", resp.StatusCode, b)
	}
	resp.Body.Close()

	resp, err = client(func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "hi") }).Get("http://x/")
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := io.ReadAll(resp.Body); resp.StatusCode != 200 || string(b) != "hi" {
		t.Errorf("implicit 200: status %d, body %q", resp.StatusCode, b)
	}
	resp.Body.Close()

	if _, err := client(func(http.ResponseWriter, *http.Request) { panic("boom") }).Get("http://x/"); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("a panic before any write: Do = %v, want an error naming it", err)
	}

	resp, err = client(func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, "partial")
		panic("late")
	}).Get("http://x/")
	if err != nil {
		t.Fatal(err)
	}
	b, rerr := io.ReadAll(resp.Body)
	if string(b) != "partial" || rerr == nil || !strings.Contains(rerr.Error(), "late") {
		t.Errorf("a panic after a write: body %q, read error %v; want the bytes written and an error naming the panic", b, rerr)
	}
	resp.Body.Close()
}
