// Package inproc gives an http.Client a transport that calls an http.Handler in the same process: no listener, no
// port, no connection. A response streams as the handler writes it, so server-sent events arrive one at a time, and
// closing the response body (or cancelling the request's context) cancels the handler's request context.
package inproc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// Transport returns a RoundTripper that serves each request with h.
//
// The handler runs on its own goroutine and sees a request whose context ends when the caller's does or when the caller
// closes the response body. Its writes block until the caller reads them, as on a connection whose peer is slow, so a
// caller must read or close every body. A handler that panics before it writes is reported as RoundTrip's error; one
// that panics later ends the body with an error.
func Transport(h http.Handler) http.RoundTripper { return transport{h} }

type transport struct{ h http.Handler }

func (t transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL == nil {
		return nil, errors.New("inproc: request has no URL")
	}
	ctx, cancel := context.WithCancel(req.Context())
	sreq := req.Clone(ctx)
	sreq.RequestURI = req.URL.RequestURI()
	sreq.RemoteAddr = "inproc"
	if sreq.Host == "" {
		sreq.Host = req.URL.Host
	}
	if sreq.Body == nil {
		sreq.Body = http.NoBody
	}
	pr, pw := io.Pipe()
	w := &writer{header: http.Header{}, pw: pw, sent: make(chan struct{})}
	go func() {
		defer sreq.Body.Close()
		defer func() {
			if p := recover(); p != nil {
				err := fmt.Errorf("inproc: handler panicked: %v", p)
				if p == http.ErrAbortHandler {
					err = errors.New("inproc: handler aborted the response")
				}
				w.fail(err)
				return
			}
			w.WriteHeader(http.StatusOK) // a handler that wrote nothing answered 200 with no body
			pw.Close()
		}()
		t.h.ServeHTTP(w, sreq)
	}()
	select {
	case <-w.sent:
	case <-req.Context().Done():
		cancel()
		pr.CloseWithError(req.Context().Err()) // unblocks a handler mid-write
		return nil, req.Context().Err()
	}
	if w.err != nil {
		cancel()
		return nil, w.err
	}
	return &http.Response{
		Status:        fmt.Sprintf("%d %s", w.code, http.StatusText(w.code)),
		StatusCode:    w.code,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        w.sentHeader,
		Body:          &body{pr: pr, cancel: cancel},
		ContentLength: -1,
		Request:       req,
	}, nil
}

// writer is the handler's http.ResponseWriter. Headers are fixed at the first WriteHeader, Write or Flush, which is
// also when RoundTrip returns the response.
type writer struct {
	header http.Header
	pw     *io.PipeWriter

	once       sync.Once
	sent       chan struct{} // closed once the status and headers are fixed, or the handler failed before that
	code       int
	sentHeader http.Header
	err        error // the handler failed before it sent headers
}

func (w *writer) Header() http.Header { return w.header }

func (w *writer) WriteHeader(code int) {
	if code >= 100 && code < 200 {
		return // informational responses are not the answer
	}
	w.once.Do(func() {
		w.code, w.sentHeader = code, w.header.Clone()
		close(w.sent)
	})
}

func (w *writer) Write(p []byte) (int, error) {
	w.WriteHeader(http.StatusOK)
	return w.pw.Write(p)
}

// Flush sends the headers if nothing has yet; written bytes need no flushing, since each Write hands them to the reader.
func (w *writer) Flush() { w.WriteHeader(http.StatusOK) }

// SetWriteDeadline and SetReadDeadline let http.ResponseController callers proceed: there is no connection to time out.
func (w *writer) SetWriteDeadline(time.Time) error { return nil }
func (w *writer) SetReadDeadline(time.Time) error  { return nil }

// fail reports a handler that ended by panicking: as RoundTrip's error if no headers went out, as the body's otherwise.
func (w *writer) fail(err error) {
	w.once.Do(func() {
		w.err = err
		close(w.sent)
	})
	w.pw.CloseWithError(err)
}

// body is the response body. Close cancels the handler's request context and unblocks a write in progress.
type body struct {
	pr     *io.PipeReader
	cancel context.CancelFunc
}

func (b *body) Read(p []byte) (int, error) { return b.pr.Read(p) }

func (b *body) Close() error {
	b.cancel()
	return b.pr.Close()
}
