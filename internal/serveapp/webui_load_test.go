package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/pull"
)

// fakeCache points os.UserCacheDir at a temp directory (XDG_CACHE_HOME on Linux, HOME on darwin)
// and returns the pull cache root under it, created.
func fakeCache(t *testing.T) (tmp, root string) {
	t.Helper()
	tmp = t.TempDir()
	t.Setenv("XDG_CACHE_HOME", filepath.Join(tmp, "cache"))
	t.Setenv("HOME", filepath.Join(tmp, "home"))
	root, err := pull.CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	return tmp, root
}

func writeCacheFile(t *testing.T, p string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestWebLoadPath_confinement is W5's security property: the web UI loads what the pull flow can put
// on disk and NOTHING else. Every refusal here is a way a caller-named path could reach the loader
// if the route were only "the admin load without the admin gate".
func TestWebLoadPath_confinement(t *testing.T) {
	tmp, root := fakeCache(t)
	good := writeCacheFile(t, filepath.Join(root, "owner", "repo", "model-q4_k_m.gguf"))
	outside := writeCacheFile(t, filepath.Join(tmp, "outside.gguf"))
	sibling := writeCacheFile(t, filepath.Join(filepath.Dir(root), "models-evil", "x.gguf")) // shares root's string prefix
	writeCacheFile(t, filepath.Join(root, "owner", "repo", "half.gguf.part"))
	writeCacheFile(t, filepath.Join(root, "owner", "repo", "notes.txt"))
	if err := os.MkdirAll(filepath.Join(root, "owner", "repo", "dir.gguf"), 0o755); err != nil {
		t.Fatal(err)
	}
	escape := filepath.Join(root, "owner", "repo", "escape.gguf")
	if err := os.Symlink(outside, escape); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "owner", "repo", "alias.gguf")
	if err := os.Symlink(good, alias); err != nil {
		t.Fatal(err)
	}
	wantGood, _ := filepath.EvalSymlinks(good)

	for _, c := range []struct {
		name, path, want string // want "" = refused
	}{
		{"a pulled gguf", good, wantGood},
		{"a symlink to a pulled gguf, inside the cache (loads the target)", alias, wantGood},
		{"empty", "", ""},
		{"relative", filepath.Join("owner", "repo", "model-q4_k_m.gguf"), ""},
		{"dot-dot out of the cache", filepath.Join(root, "owner", "..", "..", "..", "outside.gguf"), ""},
		{"a file outside the cache", outside, ""},
		{"a sibling directory sharing the root's prefix", sibling, ""},
		{"a symlink inside the cache pointing out of it", escape, ""},
		{"an in-progress download", filepath.Join(root, "owner", "repo", "half.gguf.part"), ""},
		{"not a gguf", filepath.Join(root, "owner", "repo", "notes.txt"), ""},
		{"a directory named .gguf", filepath.Join(root, "owner", "repo", "dir.gguf"), ""},
		{"a missing file", filepath.Join(root, "owner", "repo", "gone.gguf"), ""},
		{"the cache root itself", root, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := webLoadPath(c.path)
			if c.want == "" {
				if err == nil {
					t.Fatalf("webLoadPath(%q) = %q, want a refusal", c.path, got)
				}
				return
			}
			if err != nil || got != c.want {
				t.Fatalf("webLoadPath(%q) = %q, %v; want %q", c.path, got, err, c.want)
			}
		})
	}
}

// TestWebLoadPath_cacheRootBehindSymlink: the containment check compares RESOLVED paths on both
// sides. A cache directory that is itself a symlink (a relocated ~/.cache, or macOS's /var →
// /private/var under every temp dir) must still accept its own files — resolving only the request
// side would refuse every legitimate load there.
func TestWebLoadPath_cacheRootBehindSymlink(t *testing.T) {
	tmp := t.TempDir()
	realCache := filepath.Join(tmp, "real-cache")
	if err := os.MkdirAll(realCache, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(tmp, "cache")
	if err := os.Symlink(realCache, link); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CACHE_HOME", link)
	t.Setenv("HOME", filepath.Join(tmp, "home"))
	root, err := pull.CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(root, link) {
		t.Skipf("os.UserCacheDir does not follow XDG_CACHE_HOME on this OS (root %s)", root)
	}
	p := writeCacheFile(t, filepath.Join(root, "o", "r", "m.gguf"))
	if _, err := webLoadPath(p); err != nil {
		t.Fatalf("a model in a symlinked cache was refused: %v", err)
	}
}

type webLoadEvent struct {
	name string
	data map[string]any
}

func readLoadEvents(t *testing.T, body io.Reader) []webLoadEvent {
	t.Helper()
	var out []webLoadEvent
	var ev webLoadEvent
	sc := bufio.NewScanner(body)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			ev.name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &ev.data); err != nil {
				t.Fatalf("bad SSE data %q: %v", line, err)
			}
		case line == "":
			if ev.name != "" {
				out = append(out, ev)
			}
			ev = webLoadEvent{}
		}
	}
	return out
}

func postLoad(s *server, ctx context.Context, path string) *httptest.ResponseRecorder {
	b, _ := json.Marshal(webLoadReq{Path: path})
	r := httptest.NewRequest(http.MethodPost, "/web/models/load", strings.NewReader(string(b))).WithContext(ctx)
	w := httptest.NewRecorder()
	s.handleWebLoad(w, r)
	return w
}

// pulledModel puts a file named like a pulled model into the fake pull cache and returns its path.
func pulledModel(t *testing.T, root, name string) string {
	t.Helper()
	return writeCacheFile(t, filepath.Join(root, "owner", "repo", name))
}

// fakeLoads swaps the loader for one that records what it was asked to load and returns a bare
// registry entry — enough to publish, route by name and list, with no weights behind it.
func fakeLoads(t *testing.T, gate func(ctx context.Context)) *[]modelSpec {
	t.Helper()
	var calls []modelSpec
	orig := webLoadDecoder
	t.Cleanup(func() { webLoadDecoder = orig })
	webLoadDecoder = func(ctx context.Context, spec modelSpec, cfg config) (*loadedModel, error) {
		calls = append(calls, spec)
		if gate != nil {
			gate(ctx)
		}
		if ctx.Err() != nil {
			return nil, errors.New("load was cancelled with the request")
		}
		return &loadedModel{name: spec.name}, nil
	}
	return &calls
}

// TestWebLoad_publishesAndStreams: a model in the pull cache is handed to the loader by its RESOLVED
// path (requested here through an in-cache symlink, so the two differ), streamed as start → done, and published — /v1/models lists it. A second load of the same
// name is a plain 409 before any loading starts, and a path outside the cache is a 400 that never
// reaches the loader at all.
func TestWebLoad_publishesAndStreams(t *testing.T) {
	tmp, root := fakeCache(t)
	p := pulledModel(t, root, "tiny-Q4_K_M.gguf")
	// Requested through a symlink inside the cache: the loader must get the TARGET, the path that
	// was checked, and the served name must come from it too.
	alias := filepath.Join(root, "owner", "repo", "alias.gguf")
	if err := os.Symlink(p, alias); err != nil {
		t.Fatal(err)
	}
	calls := fakeLoads(t, nil)
	s, err := newServer(config{web: true, backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}

	w := postLoad(s, context.Background(), alias)
	if w.Code != http.StatusOK || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("load: %d %s: %s", w.Code, w.Header().Get("Content-Type"), w.Body.String())
	}
	evs := readLoadEvents(t, w.Body)
	if len(evs) < 2 || evs[0].name != "start" || evs[len(evs)-1].name != "done" {
		t.Fatalf("events = %+v, want start … done", evs)
	}
	if id := evs[len(evs)-1].data["id"]; id != "tiny-Q4_K_M" {
		t.Fatalf("done id = %v, want tiny-Q4_K_M", id)
	}
	resolved, _ := filepath.EvalSymlinks(p)
	if len(*calls) != 1 || (*calls)[0].path != resolved || (*calls)[0].name != "tiny-Q4_K_M" {
		t.Fatalf("loader calls = %+v, want one call for %s as tiny-Q4_K_M", *calls, resolved)
	}
	mw := httptest.NewRecorder()
	s.handleModels(mw, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if !strings.Contains(mw.Body.String(), `"tiny-Q4_K_M"`) {
		t.Errorf("/v1/models does not list the loaded model: %s", mw.Body.String())
	}

	if w := postLoad(s, context.Background(), p); w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "already loaded") {
		t.Errorf("second load: %d %s, want 409 already loaded", w.Code, w.Body.String())
	}
	outside := writeCacheFile(t, filepath.Join(tmp, "elsewhere", "other.gguf"))
	if w := postLoad(s, context.Background(), outside); w.Code != http.StatusBadRequest {
		t.Errorf("load from outside the cache: %d, want 400: %s", w.Code, w.Body.String())
	}
	if len(*calls) != 1 {
		t.Errorf("the loader ran %d times, want 1 — a refused load must never reach it", len(*calls))
	}
}

// TestWebLoad_whileRunning covers what only shows while a load is in progress, using a loader that
// blocks until released: the heartbeat streams, a second load is refused as already running rather
// than started, and closing the tab does NOT cancel the load — it still finishes and is published.
func TestWebLoad_whileRunning(t *testing.T) {
	_, root := fakeCache(t)
	p := pulledModel(t, root, "first.gguf")
	other := pulledModel(t, root, "second.gguf")
	s, err := newServer(config{web: true, backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	origBeat := webLoadHeartbeat
	t.Cleanup(func() { webLoadHeartbeat = origBeat })
	webLoadHeartbeat = 5 * time.Millisecond

	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	fakeLoads(t, func(ctx context.Context) {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
	})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- postLoad(s, ctx, p) }()
	<-entered
	time.Sleep(50 * time.Millisecond) // several heartbeats

	// In a goroutine with a deadline: a second load that STARTS instead of being refused blocks in the
	// fake loader, and must fail here rather than hang the test until go test's timeout.
	second := make(chan *httptest.ResponseRecorder, 1)
	go func() { second <- postLoad(s, context.Background(), other) }()
	select {
	case w := <-second:
		if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "already running") {
			t.Errorf("concurrent load: %d %s, want 409 already running", w.Code, w.Body.String())
		}
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("a second load started while one was running — it must be refused (single-flight)")
	}

	cancel() // the tab closes mid-load
	close(release)
	w := <-done
	s.regMu.RLock()
	_, published := s.models["first"]
	s.regMu.RUnlock()
	if !published {
		t.Fatal("the tab closed mid-load and the model was not published — the load must run to completion")
	}

	evs := readLoadEvents(t, w.Body)
	beats := 0
	for _, e := range evs {
		if e.name == "progress" {
			if _, ok := e.data["elapsed"]; !ok {
				t.Errorf("progress event without elapsed: %+v", e)
			}
			beats++
		}
	}
	if len(evs) == 0 || evs[0].name != "start" || beats == 0 {
		t.Errorf("events before the tab closed = %+v, want start then progress heartbeats", evs)
	}

	// the single-flight is released once the first load is over
	orig := webLoadDecoder
	webLoadDecoder = func(ctx context.Context, spec modelSpec, cfg config) (*loadedModel, error) {
		return nil, errors.New("fake load failure")
	}
	t.Cleanup(func() { webLoadDecoder = orig })
	w = postLoad(s, context.Background(), other)
	evs = readLoadEvents(t, w.Body)
	if len(evs) == 0 || evs[len(evs)-1].name != "error" || !strings.Contains(evs[len(evs)-1].data["message"].(string), "fake load failure") {
		t.Errorf("load after the first finished: %d %+v, want it to run and report its error", w.Code, evs)
	}
}

// TestWebLoad_realModel is the end-to-end check the fakes cannot make: a real GGUF, placed in the pull
// cache the way a pull leaves it, loads through the web route with the real loader and then answers a
// chat request. No committed GGUF carries an embedded tokenizer, so this is gated on
// GOINFER_SERVE_MODEL — the same variable TestServe_admin uses.
func TestWebLoad_realModel(t *testing.T) {
	src := os.Getenv("GOINFER_SERVE_MODEL")
	if src == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf with a tokenizer> for the real web-load test")
	}
	src, err := filepath.EvalSymlinks(src)
	if err != nil {
		t.Fatal(err)
	}
	// A hard link when the temp dir shares the model's filesystem, a copy otherwise.
	tmp := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", filepath.Join(tmp, "cache"))
	t.Setenv("HOME", filepath.Join(tmp, "home"))
	root, err := pull.CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(root, "owner", "repo", filepath.Base(src))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(src, dst); err != nil {
		b, rerr := os.ReadFile(src)
		if rerr != nil {
			t.Fatal(rerr)
		}
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	s, err := newServer(config{web: true, backend: "cpu", quant: "int8int8", kvSessions: 2})
	if err != nil {
		t.Fatal(err)
	}
	w := postLoad(s, context.Background(), dst)
	evs := readLoadEvents(t, w.Body)
	if len(evs) == 0 || evs[len(evs)-1].name != "done" {
		t.Fatalf("real load: %d %+v", w.Code, evs)
	}
	id, _ := evs[len(evs)-1].data["id"].(string)
	s.regMu.RLock()
	lm := s.models[id]
	s.regMu.RUnlock()
	if lm == nil || lm.model == nil {
		t.Fatalf("done id %q is not a loaded model in the registry", id)
	}
	t.Cleanup(func() { lm.model.Close(); lm.closeEntryNatives() })

	body := `{"model":"` + id + `","max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`
	cw := httptest.NewRecorder()
	s.handleChat(cw, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if cw.Code != http.StatusOK {
		t.Fatalf("chat with the web-loaded model: %d %s", cw.Code, cw.Body.String())
	}
}
