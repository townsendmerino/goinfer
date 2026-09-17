package serveapp

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/pull"
)

// The local web UI (docs/completed/task-model-pull.md §4, option B).
//
// It rides the server that already exists: the page is a small set of embedded files — index.html
// plus its stylesheet and script under webui/ui/ — with no external stylesheet, font or script,
// and it talks to the SAME /v1/models and
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

// ONE DIRECTORY, NOT ONE FILE (docs/tasks/task-web-ui-2026-09.md §6.1). The page was a single
// 1,828-line HTML file; the web-UI plan roughly doubles it, so it is split into index.html plus
// webui/ui/app.css and webui/ui/app.js. What is deliberately NOT given up: still no build step, no
// bundler, no toolchain — plain files, embedded verbatim, one binary, fully offline. The assets
// live under ui/ and are referenced relatively ("ui/app.js"), so the same page also loads from
// file:// with no server, which is how a headless browser can check it on a box whose sandbox
// blocks loopback HTTP.
//
//go:embed webui
var webUIFS embed.FS

// webUIPage is index.html, read once. A missing file is a build defect, not a runtime condition,
// so it panics at init rather than serving an empty page.
var webUIPage = func() []byte {
	b, err := webUIFS.ReadFile("webui/index.html")
	if err != nil {
		panic("serveapp: embedded web UI is missing index.html: " + err.Error())
	}
	return b
}()

// webUIAssetTypes is the complete set of asset kinds the page may load, with the Content-Type each
// is served as. An allow-list, not mime.TypeByExtension: the page is served from the API's own
// origin, so an embedded file of an unexpected type must 404 rather than be sniffed into
// something executable. Add a type here deliberately, with the file that needs it.
var webUIAssetTypes = map[string]string{
	".css": "text/css; charset=utf-8",
	".js":  "text/javascript; charset=utf-8",
}

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

// handleWebAsset serves the page's own stylesheet and script from webui/ui/. UNAUTHENTICATED, for
// the same reason as the page (V-02, see main.go): a browser's plain subresource load sends no
// Authorization header, and without these files the page — the only place the key can be typed —
// cannot work at all. They are static, embedded at build time, and hold no secrets.
//
// Only a single path segment under ui/ with an allow-listed extension is served; anything else,
// including a traversal attempt or a directory, is a plain 404. No directory listing, ever.
func (s *server) handleWebAsset(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("file")
	ctype, ok := webUIAssetTypes[path.Ext(name)]
	if !ok || name == "" || strings.ContainsAny(name, `/\`) || strings.HasPrefix(name, ".") || !fs.ValidPath(name) {
		http.NotFound(w, r)
		return
	}
	b, err := webUIFS.ReadFile("webui/ui/" + name)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", ctype)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// no-store, matching the page: the UI and the binary are one version, and a cached app.js from
	// the previous build talking to a newer server is exactly the drift being a client of the API
	// is supposed to rule out.
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(b)
}

type webPullReq struct {
	Repo  string `json:"repo"`
	Quant string `json:"quant"`
	File  string `json:"file"`
}

// freeBytesForActiveBackend reports free memory for the server's own currently-configured
// backend. decoder.FreeBytesFor's registry covers every GPU backend, but "cpu" (and this
// server's own "" default) has never gone through it — decoder/backend.go's own doc comment says
// so: it predates the registry and is used by decoder/fitguard.go directly — so that one case is
// special-cased here rather than silently reporting "unknown" for the single most common backend.
func (s *server) freeBytesForActiveBackend() (int64, bool) {
	if s.cfg.backend == "" || s.cfg.backend == "cpu" {
		b := decoder.HostRAMAvailableBytes()
		return b, b > 0
	}
	return decoder.FreeBytesFor(s.cfg.backend)
}

// fitEstimate is a COARSE per-file verdict for the file listing, using only the file's own
// on-disk size against free memory for the backend this server actually runs — no header
// parsing, no model load (docs/tasks/task-fit-to-hardware.md §3's own words: "the file table can
// say fits / needs streaming / will not fit per row from the listing alone, before the
// multi-gigabyte transfer"). NOT exact, and the UI note beside this field says so: a GGUF's size
// only approximates its resident bytes when the file is already at roughly the quant this server
// loads at — `-quant` re-quantizes at load time regardless of the file's own native format
// (decoder/gguf.go's buildGGUFWeights), so pulling a Q8_0 file onto an int4 server resident-loads
// far smaller than this file's size suggests, and the reverse case overshoots. Parsing the
// file's own quant hint out of its name to correct for that would need a real quant-name table
// this coarse a check has no business building — see the scoping note in
// task-web-ui-2026-09.md's W33 part 2 for why that line was drawn here.
//
// Bands are deliberately conservative (fewer false "fits"): a resident load also needs KV cache
// and context on top of raw weight bytes, so "file size == everything free" already does not fit.
func fitEstimate(fileSize, free int64) string {
	switch {
	case fileSize >= free:
		return "wont_fit"
	case fileSize >= free*6/10:
		return "tight"
	default:
		return "fits"
	}
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
	free, freeOK := s.freeBytesForActiveBackend()
	out := make([]map[string]any, 0, len(files))
	for _, f := range files {
		row := map[string]any{"path": f.Path, "size": f.Size, "human": pull.HumanBytes(f.Size), "sha256": f.SHA256}
		if freeOK {
			row["fit"] = fitEstimate(f.Size, free)
		}
		out = append(out, row)
	}
	resp := map[string]any{"repo": ref.Repo, "files": out}
	if freeOK {
		backend := s.cfg.backend
		if backend == "" {
			backend = "cpu" // the flag's own default (main.go) — never shown blank to the page
		}
		resp["fit_backend"] = backend
		resp["free_bytes"] = free
		resp["free_human"] = pull.HumanBytes(free)
	}
	writeJSON(w, http.StatusOK, resp)
}

type webSearchReq struct {
	Query string `json:"query"`
	// Kind narrows the search (pull.Search's own searchKinds table). The page always sends
	// "gguf" today — the only kind this build's pull flow actually loads — but the field exists
	// now, not added later, so a future kind is a client change plus one table entry in pull.go,
	// never a new route or request shape.
	Kind string `json:"kind"`
}

// searchLimit bounds how many suggestions a search returns — a dropdown, not a full listing;
// pull.Search sends no limit= to HuggingFace at all when given 0, which is a request shape this
// route should never produce. Raised from 8 to 50 live during testing (2026-09-17): 8 was too
// narrow to surface a less-trending-but-still-relevant repo past HF's own trendingScore ordering
// (pull.Search's own doc comment — not downloads or likes) for anything but the most obvious query.
const searchLimit = 50

// handleWebSearch answers the repo box's search-as-you-type: candidates to PICK from, not a
// commitment to any of them — CheckAccess/List (handleWebList, above) still run, unchanged, once
// a person actually chooses one. A network or HF-side failure here is not fatal to typing a repo
// name by hand, so it is reported as an ordinary error the page can show or quietly drop, never a
// reason to block the box itself.
func (s *server) handleWebSearch(w http.ResponseWriter, r *http.Request) {
	if !s.webEnabled(w) {
		return
	}
	var req webSearchReq
	if !decodeJSON(w, r, &req) {
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	results, err := pull.Search(ctx, req.Query, req.Kind, searchLimit)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	out := make([]map[string]any, 0, len(results))
	for _, res := range results {
		out = append(out, map[string]any{"repo": res.Repo, "downloads": res.Downloads, "likes": res.Likes})
	}
	writeJSON(w, http.StatusOK, map[string]any{"repos": out})
}

// webPullRef resolves a pull request's repo box and the clicked selector (if any) into one Ref,
// WITHOUT string concatenation. req.Repo alone decides ref.Repo (`pull.ParseRef` cuts at the
// FIRST colon, so blindly appending a second selector after a box that already carries one —
// "owner/repo:q4_k_m" typed in, then a file clicked — produced "owner/repo:q4_k_m:file.gguf",
// re-cut into repo="owner/repo", selector="q4_k_m:file.gguf": a ".gguf"-suffixed string that
// LOOKS like a filename and is looked up as one, verbatim, in a repo that publishes no such name.
// Listing never showed this, because handleWebList parses req.Repo alone and only ever reads
// ref.Repo back out of it — so every Pull button failed while List worked, which read like a bad
// repo rather than a bad concatenation.
//
// A clicked file or quant REPLACES whatever selector the box already carried, rather than
// appending to it — the click is the more specific, more recent choice. With neither clicked, the
// box's own parse is returned as-is: re-parsing a resolved `demo:` tier would still yield the same
// Repo/File, but would drop Pin/Bytes (ParseRef only ever sets those for a literal "demo:tier"
// input, not for the repo/file pair a tier resolves to) — the digest a demo: pull is supposed to
// verify against.
func webPullRef(req webPullReq) (pull.Ref, error) {
	base, err := pull.ParseRef(req.Repo)
	if err != nil {
		return pull.Ref{}, err
	}
	switch {
	case req.File != "":
		return pull.Ref{Repo: base.Repo, File: req.File}, nil
	case req.Quant != "":
		return pull.Ref{Repo: base.Repo, Quant: req.Quant}, nil
	default:
		return base, nil
	}
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
	ref, err := webPullRef(req)
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

	// r.Context() dies when the browser tab closes, which cancels the transfer and lets
	// Download clean up its .part file — no orphaned multi-GB write after a closed tab. Wrapped
	// in our own cancel (N-23, docs/audit-2026-09-10.md) so a STALLED-but-open connection —
	// caught by sseWriter's write deadline below, not by r.Context() — stops the download the
	// same way.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	send := sseJSONSender(w, flusher, cancel)
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

type webLoadReq struct {
	Path string `json:"path"`
}

// webLoadPath decides whether the page may load p, and returns the resolved file to load.
//
// THIS IS THE WHOLE POLICY OF THE LOAD ROUTE (docs/tasks/task-web-ui-2026-09.md W5). The admin load
// takes any caller-named path and is gated behind -allow-admin for exactly that reason; the web UI
// is not widened into it. Instead the page can load only what the pull flow can put on disk: a
// regular .gguf file under pull.CacheRoot(). Both p and the root are symlink-resolved BEFORE the
// containment check, so neither a "../" in the request nor a symlink planted inside the cache can
// point the loader at a file outside it, and the resolved path — not the requested one — is what
// gets loaded. The suffix is checked on the resolved name too, which also refuses an in-progress
// download (Download writes "<name>.part" and renames only after the digest verifies).
//
// Not defended: someone who can already write into the user's cache directory can swap a file
// between this check and the load. That is the user's own account, which could run anything anyway.
func webLoadPath(p string) (string, error) {
	if p == "" {
		return "", errors.New("path is required")
	}
	if !filepath.IsAbs(p) {
		return "", errors.New("path must be the absolute path a pull returned")
	}
	root, err := pull.CacheRoot()
	if err != nil {
		return "", err
	}
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", fmt.Errorf("the pull cache %s is not readable: %w", root, err)
	}
	resolved, err := filepath.EvalSymlinks(filepath.Clean(p))
	if err != nil {
		return "", fmt.Errorf("no such model file: %s", p)
	}
	rel, err := filepath.Rel(realRoot, resolved)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", fmt.Errorf("the web UI loads only models pulled into %s; start the server with --model, or use -allow-admin, for anything else", root)
	}
	if !strings.EqualFold(filepath.Ext(resolved), ".gguf") {
		return "", errors.New("only a pulled .gguf file can be loaded from the web UI")
	}
	fi, err := os.Stat(resolved)
	if err != nil || !fi.Mode().IsRegular() {
		return "", fmt.Errorf("not a regular file: %s", p)
	}
	return resolved, nil
}

// handleWebLoad loads a model the pull flow downloaded and makes it routable, streaming SSE:
// start {name}, progress {elapsed} every couple of seconds, then done {id, elapsed} or error
// {message, status}. It shares loadDecoder and publishLoaded with the admin load, so a model loaded
// here is indistinguishable from one loaded any other way — including by unload.
//
// It uses the server's own settings (backend, quant, KV, sessions) with no per-request override:
// the page offers "load what you pulled", not a second admin API.
func (s *server) handleWebLoad(w http.ResponseWriter, r *http.Request) {
	if !s.webEnabled(w) {
		return
	}
	var req webLoadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	file, err := webLoadPath(req.Path)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	name := strings.TrimSuffix(filepath.Base(file), filepath.Ext(file))
	s.regMu.RLock()
	_, dup := s.models[name]
	s.regMu.RUnlock()
	if dup {
		writeErr(w, http.StatusConflict, fmt.Sprintf("model %q already loaded", name))
		return
	}
	// One load at a time, for the same reason as pulls: each one maps a multi-gigabyte file and may
	// claim most of the device's memory, and a double click must not start two.
	if !s.loads.acquire() {
		writeErr(w, http.StatusConflict, "a model load is already running")
		return
	}
	defer s.loads.release()

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
		if r.Context().Err() != nil {
			return // the tab went away; the load carries on regardless (below)
		}
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		flusher.Flush()
	}
	send("start", map[string]any{"name": name})

	// DETACHED FROM THE REQUEST, unlike the pull. A cancelled pull leaves nothing behind (Download
	// removes its .part), but a load cancelled part-way has already spent its minutes, and the user
	// asked for the model — so closing the tab does not undo the request. The load finishes, is
	// published, and the next /v1/models shows it. The handler still waits for it, so the
	// single-flight above stays held for the whole load rather than just for the request.
	type result struct {
		lm  *loadedModel
		err error
	}
	doneCh := make(chan result, 1)
	ctx := context.WithoutCancel(r.Context())
	go func() {
		lm, err := webLoadDecoder(ctx, modelSpec{name: name, path: file}, s.cfg)
		doneCh <- result{lm, err}
	}()
	start := time.Now()
	tick := time.NewTicker(webLoadHeartbeat)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			send("progress", map[string]any{"elapsed": time.Since(start).Round(time.Second).String()})
		case res := <-doneCh:
			switch {
			case res.err != nil:
				send("error", map[string]any{"message": res.err.Error() + s.unloadSuggestion(res.err), "status": http.StatusBadRequest})
			case !s.publishLoaded(res.lm):
				send("error", map[string]any{"message": fmt.Sprintf("model %q already loaded", name), "status": http.StatusConflict})
			default:
				send("done", map[string]any{"id": res.lm.name, "elapsed": time.Since(start).Round(time.Second).String()})
			}
			return
		}
	}
}

type webUnloadReq struct {
	Name string `json:"name"`
}

// unloadSuggestion turns a fit-guard refusal into an actionable one (W32, task-web-ui-2026-09.md
// bullet 6): the decoder package that raised it has no view of the registry, so it can only say the
// shortfall; the registry is what can name a way out. Returns "" for any other kind of error, or
// when nothing is resident to suggest freeing (a fresh server's first load is a real refusal with no
// fix on this page — the message should not imply one exists).
func (s *server) unloadSuggestion(err error) string {
	if !errors.Is(err, decoder.ErrWontFitResident) {
		return ""
	}
	s.regMu.RLock()
	var biggest string
	var biggestBytes int64
	for name, lm := range s.models {
		if lm.model == nil {
			continue
		}
		b := lm.model.ResidentWeightBytes() + lm.model.ExtraResidentBytes()
		if b > biggestBytes {
			biggest, biggestBytes = name, b
		}
	}
	s.regMu.RUnlock()
	if biggest == "" {
		return ""
	}
	// Prose, not a line break: the caller appends this straight onto res.err.Error(), which the page
	// renders as plain textContent with no white-space:pre-line — a literal "\n" would just collapse
	// to a space, so the sentence has to read correctly either way.
	return fmt.Sprintf(" Unload %q (%.1f GB) to make room.", biggest, float64(biggestBytes)/(1<<30))
}

// handleWebUnload is W32 (task-web-ui-2026-09.md): unload a model the page itself can already see.
//
// UNLIKE LOAD, THIS NEEDS NO PATH POLICY AT ALL. webLoadPath exists because a load names a
// filesystem path the admin route would otherwise trust unconditionally; unload names nothing but a
// registry key, and the only registry keys that exist are the ones GET /v1/models already publishes
// to every client. s.models[req.Name] under regMu — the same lookup unloadByName does internally —
// IS the whole policy: a name not currently loaded is a 404, exactly as if the page had asked to
// cancel a job id it never held (W27/W31's rule, applied here to a different registry). No new
// containment logic, no -allow-admin, no widening of the admin surface: unloadByName is the same
// function the admin route calls, so a model unloaded from the page drains exactly the way one
// unloaded through /admin does.
func (s *server) handleWebUnload(w http.ResponseWriter, r *http.Request) {
	if !s.webEnabled(w) {
		return
	}
	var req webUnloadReq
	if !decodeJSON(w, r, &req) {
		return
	}
	status, body, ok := s.unloadByName(req.Name, s.cfg.unloadDrainWait)
	if !ok {
		s.modelNotFound(w, req.Name)
		return
	}
	writeJSON(w, status, body)
}

// webLoadDecoder is loadDecoder, as a seam: the tests need a load that blocks until told to finish,
// to see the heartbeat, the single-flight and the tab-close behaviour while a load is in progress.
var webLoadDecoder = loadDecoder

// webLoadHeartbeat is how often a running load reports that it is still running. A load has no
// byte count to report, so this is liveness only — it tells "still loading" from "hung".
var webLoadHeartbeat = 2 * time.Second

// webEnabled mirrors requireAdmin's TCP-listener gate: the routes exist only when -web was
// passed, and say so in the same shape the rest of the API uses rather than 404-ing.
func (s *server) webEnabled(w http.ResponseWriter) bool {
	if !s.cfg.web {
		writeErr(w, http.StatusForbidden, "web UI is disabled; start the server with -web")
		return false
	}
	return true
}
