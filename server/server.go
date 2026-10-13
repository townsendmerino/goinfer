// Package server is goinfer-serve's HTTP API as a library: the same routes, served by a handler in your own
// process, with no listener, port or daemon.
//
// Pair NewHandler with Transport and any client that accepts an *http.Client talks to a local model in process: the
// OpenAI and Anthropic Go SDKs and the frameworks built on them take one. The three API dialects, structured output,
// tool calling and reasoning separation are the server's own.
//
//	h, closer, err := server.NewHandler("-model", "model.gguf")
//	if err != nil { ... }
//	defer closer.Close()
//	client := &http.Client{Transport: server.Transport(h)}
//	// point an SDK at http://goinfer/v1 with this client
//
// A GPU backend is linked the way the binaries link it: import the backend's module for its registration
// (github.com/townsendmerino/goinfer/metal, .../cuda or .../gpu) and pass "-backend".
//
// Experimental: this package is outside the compatibility guarantee (docs/api-tiers.md) and may change in any release.
package server

import (
	"io"
	"net/http"

	"github.com/townsendmerino/goinfer/internal/inproc"
	"github.com/townsendmerino/goinfer/internal/serveapp"
)

// NewHandler loads the models the flags name and returns the server's HTTP API as a handler. The flags are
// goinfer-serve's own, with the same meanings and defaults (docs/flags.md). Those that configure the binary's
// listener or process, such as -addr, the TLS pair, -web and the admin surfaces, are refused. -api-key is honoured
// when given; the environment is not read for one.
//
// Close cancels the generations in flight, saves sessions when -session-dir is set, and unloads the models. Do not
// use the handler after it.
func NewHandler(flags ...string) (http.Handler, io.Closer, error) {
	return serveapp.NewHandler(flags)
}

// Transport returns a RoundTripper that serves each request with h in this process. A response streams as the
// handler writes it, so server-sent events arrive as they are produced. Cancelling the request's context, or closing
// the response body, stops the generation. The request's host is ignored; "http://goinfer" is as good as any.
func Transport(h http.Handler) http.RoundTripper { return inproc.Transport(h) }
