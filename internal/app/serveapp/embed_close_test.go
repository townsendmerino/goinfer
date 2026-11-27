package serveapp

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

// blockEncoder is an embedder that is not a decoderEmbedder: its EncodeBatch blocks until release, and its Close only records
// that it ran, with no lock of its own, so only the handler's gate keeps Close off a running encode.
type blockEncoder struct {
	stubEncoder
	started, release chan struct{}
	inEncode         atomic.Bool
	closedUnder      atomic.Bool // Close ran while an encode was in progress
	closed           atomic.Bool
}

func (e *blockEncoder) EncodeBatch(texts []string, q []bool, c int) ([][]float32, error) {
	e.inEncode.Store(true)
	defer e.inEncode.Store(false)
	e.started <- struct{}{}
	<-e.release
	return e.stubEncoder.EncodeBatch(texts, q, c)
}

func (e *blockEncoder) Close() error {
	e.closedUnder.Store(e.inEncode.Load())
	e.closed.Store(true)
	return nil
}

// handlerCloser.Close waits for a running /v1/embeddings encode before closing the embedder, and a request after it is a 503.
func TestEmbeddings_closeWaitsForTheEncodeAndLaterRequestsGet503(t *testing.T) {
	enc := &blockEncoder{stubEncoder: stubEncoder{dim: 4}, started: make(chan struct{}, 1), release: make(chan struct{})}
	s := &server{embedService: embedService{embed: enc, embedDim: 4, embedID: "block-embed"}}
	running := make(chan int, 1)
	go func() { running <- postEmbed(t, s, `{"model":"block-embed","input":["a"]}`).Code }()
	<-enc.started

	_, stop := context.WithCancelCause(context.Background())
	closed := make(chan error, 1)
	go func() { closed <- (&handlerCloser{srv: s, stop: stop}).Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned while an encode was running")
	case <-time.After(50 * time.Millisecond):
	}
	close(enc.release)
	if code := <-running; code != http.StatusOK {
		t.Errorf("the running request: %d, want 200", code)
	}
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !enc.closed.Load() || enc.closedUnder.Load() {
		t.Fatalf("embedder closed %v, closed under a running encode %v; want closed after it", enc.closed.Load(), enc.closedUnder.Load())
	}
	if rr := postEmbed(t, s, `{"model":"block-embed","input":["b"]}`); rr.Code != http.StatusServiceUnavailable {
		t.Errorf("a request after Close: %d %s, want 503", rr.Code, rr.Body.String())
	}
}
