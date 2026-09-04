package serveapp

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/pull"
)

// The local web UI (docs/completed/task-model-pull.md §4, option B).
//
// It rides the server that already exists: the page is a single embedded HTML file with no
// external stylesheet, font or script, and it talks to the SAME /v1/models and
// /v1/chat/completions routes any other client uses. That is the point — it adds no second
// inference path to keep in sync, and it cannot drift from the API, because it IS a client
// of it. Being asset-free also keeps the offline story intact: a CDN reference would make
// the UI of an offline-capable engine require the network.
//
// A native desktop app was considered and rejected in the same design note: every realistic
// toolkit needs cgo or a bundled webview runtime, which is the one property this project is
// built to avoid.
//
// OFF BY DEFAULT (-web). The page itself is static and harmless; the pull route is not — it
// triggers an outbound download of a caller-named repo and writes it to disk. It therefore
// sits behind the same two gates -allow-admin uses: an explicit opt-in flag, and the
// startup rule that a non-loopback bind must carry an -api-key. Loopback stays key-free so
// the ordinary single-user desktop case has no auth friction.

//go:embed webui/index.html
var webUIPage []byte

// pullState serialises pulls. One at a time, deliberately: the endpoint starts a
// multi-gigabyte transfer, so without this a handful of clicks (or requests) queue unbounded
// concurrent downloads against the same disk. Single-flight also makes the progress stream
// unambiguous — there is only ever one thing to report on.
type pullState struct {
	mu      sync.Mutex
	running bool
}

func (p *pullState) acquire() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.running {
		return false
	}
	p.running = true
	return true
}

func (p *pullState) release() {
	p.mu.Lock()
	p.running = false
	p.mu.Unlock()
}

// sameOrigin refuses a cross-origin request, the same guard N-26 added to the demo/agent web
// app for its own mutating routes (cmd/agent-web/main.go). V-20 (docs/review-2026-09-04.md):
// the two routes below (list, pull) act — pull triggers a caller-named multi-gigabyte download
// — and on the key-free loopback default (requireAuth is a no-op when -api-key is unset) they
// had NO protection at all: any page open in the same browser can send a cross-origin POST
// (browsers block reading the response, not sending the request), so a malicious or compromised
// page could drive a multi-GB download onto the user's disk with no visible prompt. A browser's
// own fetch()/XHR from the web UI's page always carries a same-origin Origin header; a
// cross-origin POST either carries a foreign one (refused here) or, for a same-site plain form
// post, none at all outside a browser context — which is why (like N-26) this only checks an
// Origin header that IS present, rather than requiring one.
func sameOrigin(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if o := r.Header.Get("Origin"); o != "" {
			u, err := url.Parse(o)
			if err != nil || u.Host != r.Host {
				writeErr(w, http.StatusForbidden, "cross-origin request refused")
				return
			}
		}
		h(w, r)
	}
}

func (s *server) handleWebUI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// The page is generated per build and embeds no secrets, but it is served from the same
	// origin as the API, so keep the browser from sniffing it into anything else.
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(webUIPage)
}

type webPullReq struct {
	Repo  string `json:"repo"`
	Quant string `json:"quant"`
	File  string `json:"file"`
}

// handleWebList answers the UI's "what does this repo publish?" step. Separate from the
// pull itself so the user chooses a concrete file before anything multi-gigabyte starts.
func (s *server) handleWebList(w http.ResponseWriter, r *http.Request) {
	if !s.webEnabled(w) {
		return
	}
	var req webPullReq
	if !decodeJSON(w, r, &req) {
		return
	}
	ref, err := pull.ParseRef(req.Repo)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	if err := pull.CheckAccess(ctx, ref.Repo); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	files, err := pull.List(ctx, ref.Repo)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	out := make([]map[string]any, 0, len(files))
	for _, f := range files {
		out = append(out, map[string]any{"path": f.Path, "size": f.Size, "human": pull.HumanBytes(f.Size), "sha256": f.SHA256})
	}
	writeJSON(w, http.StatusOK, map[string]any{"repo": ref.Repo, "files": out})
}

// handleWebPull streams download progress as SSE. It reuses the pull package unchanged,
// so the digest verification and the .part-then-rename behaviour are identical to the CLI's
// — the UI is a second front end on one implementation, not a second implementation.
func (s *server) handleWebPull(w http.ResponseWriter, r *http.Request) {
	if !s.webEnabled(w) {
		return
	}
	var req webPullReq
	if !decodeJSON(w, r, &req) {
		return
	}
	spec := req.Repo
	switch {
	case req.File != "":
		spec += ":" + req.File
	case req.Quant != "":
		spec += ":" + req.Quant
	}
	ref, err := pull.ParseRef(spec)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if ref.File == "" && ref.Quant == "" {
		writeErr(w, http.StatusBadRequest, "name a quant or a file to pull")
		return
	}
	if !s.pulls.acquire() {
		writeErr(w, http.StatusConflict, "a model pull is already running")
		return
	}
	defer s.pulls.release()

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)

	send := func(event string, payload any) {
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}

	// r.Context() dies when the browser tab closes, which cancels the transfer and lets
	// Download clean up its .part file — no orphaned multi-GB write after a closed tab.
	ctx := r.Context()
	if err := pull.CheckAccess(ctx, ref.Repo); err != nil {
		send("error", map[string]string{"message": err.Error()})
		return
	}
	files, err := pull.List(ctx, ref.Repo)
	if err != nil {
		send("error", map[string]string{"message": err.Error()})
		return
	}
	f, err := pull.Select(files, ref)
	if err != nil {
		send("error", map[string]string{"message": err.Error()})
		return
	}
	dir, err := pull.CacheDir(ref.Repo)
	if err != nil {
		send("error", map[string]string{"message": err.Error()})
		return
	}
	send("start", map[string]any{"file": f.Path, "size": f.Size, "human": pull.HumanBytes(f.Size), "sha256": f.SHA256, "dir": dir})

	start := time.Now()
	path, err := pull.Download(ctx, ref.Repo, f, dir, func(done, total int64) {
		el := time.Since(start).Seconds()
		var rate float64
		if el > 0 {
			rate = float64(done) / el
		}
		p := map[string]any{"done": done, "total": total, "human": pull.HumanBytes(done), "rate": pull.HumanBytes(int64(rate)) + "/s"}
		// ETA only once the sample is long enough to mean something. A confidently wrong
		// estimate is worse than none: it gets planned around.
		if total > 0 && rate > 0 && el > 3 {
			p["eta"] = time.Duration(float64(total-done) / rate * float64(time.Second)).Round(time.Second).String()
		}
		send("progress", p)
	})
	if err != nil {
		if ctx.Err() != nil {
			return // client went away; nothing useful to send down a dead stream
		}
		send("error", map[string]string{"message": err.Error()})
		return
	}
	send("done", map[string]any{
		"path":     path,
		"elapsed":  time.Since(start).Round(time.Second).String(),
		"verified": f.SHA256 != "",
	})
}

// webEnabled mirrors adminEnabled: the routes exist only when -web was passed, and say so
// in the same shape the rest of the API uses rather than 404-ing.
func (s *server) webEnabled(w http.ResponseWriter) bool {
	if !s.cfg.web {
		writeErr(w, http.StatusForbidden, "web UI is disabled; start the server with -web")
		return false
	}
	return true
}
