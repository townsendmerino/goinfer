package serveapp

import (
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// ownerOnlyDir is a temp directory at mode 0700 whatever the umask, short enough to hold a Unix socket path
// (t.TempDir's path, built from the test's name, can pass the 104 bytes darwin allows).
func ownerOnlyDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "gs")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	if err := os.Chmod(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	return dir
}

// A path the page names gets one refusal whether it does not exist, exists outside the cache, or leaves the cache by a
// symlink: the wording must not tell a page which files this machine has.
func TestWebLoadPath_oneRefusalOutsideTheCache(t *testing.T) {
	tmp, root := fakeCache(t)
	outside := writeCacheFile(t, filepath.Join(tmp, "outside.gguf"))
	outsideDir := filepath.Join(tmp, "a-directory.gguf")
	if err := os.MkdirAll(outsideDir, 0o755); err != nil {
		t.Fatal(err)
	}
	notes := writeCacheFile(t, filepath.Join(tmp, "notes.txt"))
	escape := filepath.Join(root, "owner", "repo", "escape.gguf")
	writeCacheFile(t, filepath.Join(root, "owner", "repo", "model.gguf"))
	if err := os.Symlink(outside, escape); err != nil {
		t.Fatal(err)
	}
	cases := []struct{ name, path string }{
		{"a file that exists outside the cache", outside},
		{"a file that does not exist outside the cache", filepath.Join(tmp, "no-such-file.gguf")},
		{"a directory that exists outside the cache", outsideDir},
		{"a directory that does not exist", filepath.Join(tmp, "no-such-dir", "x.gguf")},
		{"a file that exists and is not a model", notes},
		{"dot-dot out of the cache to a file that exists", filepath.Join(root, "owner", "..", "..", "..", "outside.gguf")},
		{"dot-dot out of the cache to nothing", filepath.Join(root, "owner", "..", "..", "..", "nothing.gguf")},
		{"a symlink in the cache that points out of it", escape},
	}
	var first string
	for _, c := range cases {
		got, err := webLoadPath(c.path)
		if err == nil {
			t.Fatalf("%s: webLoadPath(%q) = %q, want a refusal", c.name, c.path, got)
		}
		msg := err.Error()
		if first == "" {
			first = msg
		}
		if msg != first {
			t.Errorf("%s: the refusal differs, so it tells the page something about the path\n  this:  %s\n  first: %s", c.name, msg, first)
		}
		for _, leak := range []string{c.path, filepath.Base(c.path), "no such", "not exist"} {
			if strings.Contains(msg, leak) {
				t.Errorf("%s: the refusal contains %q: %s", c.name, leak, msg)
			}
		}
	}
}

// requireAuth compares digests, so every near miss of the key must still be refused: a shorter one, a longer one, one
// of the same length, and the empty one.
func TestRequireAuth_nearMisses(t *testing.T) {
	const key = "s3cret-key-of-some-length"
	h := requireAuth(key, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	call := func(header, value string) int {
		r := httptest.NewRequest("POST", "/x", nil)
		if header != "" {
			r.Header.Set(header, value)
		}
		w := httptest.NewRecorder()
		h(w, r)
		return w.Code
	}
	for _, miss := range []string{"", key[:len(key)-1], key + "x", key[:len(key)-1] + "X", " " + key, key + " ", strings.ToUpper(key), strings.Repeat("a", len(key)), strings.Repeat("a", 1<<16)} {
		if c := call("x-api-key", miss); c != http.StatusUnauthorized && miss != "" {
			t.Errorf("x-api-key %q: got %d, want 401", miss, c)
		}
		if c := call("Authorization", "Bearer "+miss); c != http.StatusUnauthorized {
			t.Errorf("Authorization: Bearer %q: got %d, want 401", miss, c)
		}
	}
	if c := call("", ""); c != http.StatusUnauthorized {
		t.Errorf("no header: got %d, want 401", c)
	}
	if c := call("x-api-key", key); c != http.StatusOK {
		t.Errorf("the key in x-api-key: got %d, want 200", c)
	}
	if c := call("Authorization", "Bearer "+key); c != http.StatusOK {
		t.Errorf("the key as a bearer token: got %d, want 200", c)
	}
}

// The page is served with webUICSP, and the page's own files must load under it. The browser gates open the page from
// file://, where no policy applies, so a script or handler the policy blocks would pass them and break the served page;
// this test reads the files for what the policy does not allow. scripts/webui_csp_smoke.mjs checks a live server.
func TestWebUI_contentPolicyAllowsWhatThePageLoads(t *testing.T) {
	w := httptest.NewRecorder()
	(&server{}).handleWebUI(w, httptest.NewRequest("GET", "/", nil))
	got := w.Header().Get("Content-Security-Policy")
	if got != webUICSP {
		t.Fatalf("Content-Security-Policy = %q, want webUICSP", got)
	}
	for _, d := range []string{"default-src 'none'", "script-src 'self'", "connect-src 'self'", "frame-src data:", "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'"} {
		if !strings.Contains(got+";", d+";") {
			t.Errorf("the policy has no %q directive: %s", d, got)
		}
	}
	for _, loose := range []string{"'unsafe-eval'", "script-src 'self' 'unsafe-inline'", "*", "http:", "https:"} {
		if strings.Contains(got, loose) {
			t.Errorf("the policy contains %q: %s", loose, got)
		}
	}

	page, err := webUIFS.ReadFile("webui/index.html")
	if err != nil {
		t.Fatal(err)
	}
	html := string(page)
	local := regexp.MustCompile(`^[A-Za-z0-9_./-]+$`)
	scripts := regexp.MustCompile(`(?is)<script\b([^>]*)>(.*?)</script>`).FindAllStringSubmatch(html, -1)
	if len(scripts) == 0 {
		t.Fatal("index.html has no <script> element: this test is reading the wrong file")
	}
	for _, m := range scripts {
		src := regexp.MustCompile(`\bsrc="([^"]*)"`).FindStringSubmatch(m[1])
		if src == nil || !local.MatchString(src[1]) || strings.HasPrefix(src[1], "/") {
			t.Errorf("<script%s> does not load a file of this server: script-src 'self' blocks it", m[1])
		}
		if strings.TrimSpace(m[2]) != "" {
			t.Errorf("<script%s> has an inline body: script-src 'self' blocks it", m[1])
		}
	}
	if m := regexp.MustCompile(`(?i)<[a-z][^>]*\son[a-z]+\s*=`).FindString(html); m != "" {
		t.Errorf("index.html has an inline event handler, which script-src 'self' blocks: %s", m)
	}
	if regexp.MustCompile(`(?i)href\s*=\s*["']?\s*javascript:`).MatchString(html) {
		t.Error("index.html has a javascript: link, which script-src 'self' blocks")
	}
	for _, m := range regexp.MustCompile(`(?i)<link\b[^>]*\bhref="([^"]*)"`).FindAllStringSubmatch(html, -1) {
		if !local.MatchString(m[1]) || strings.HasPrefix(m[1], "/") {
			t.Errorf("<link href=%q> does not load a file of this server", m[1])
		}
	}
	if regexp.MustCompile(`(?i)<(img|iframe|object|embed|video|audio|source|form|base)\b`).MatchString(html) {
		t.Error("index.html has an element that loads or posts somewhere; check it against webUICSP and add it to this test")
	}

	// The scripts: nothing that needs 'unsafe-eval', and no request to another host.
	nFiles := 0
	err = fs.WalkDir(webUIFS, "webui/ui", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".js") {
			return err
		}
		b, err := webUIFS.ReadFile(p)
		if err != nil {
			return err
		}
		nFiles++
		js := string(b)
		for name, re := range map[string]*regexp.Regexp{
			"eval":                         regexp.MustCompile(`\beval\s*\(`),
			"new Function":                 regexp.MustCompile(`\bnew\s+Function\s*\(`),
			"a string passed to a timer":   regexp.MustCompile(`\bset(Timeout|Interval)\s*\(\s*["'` + "`" + `]`),
			"a worker, socket or beacon":   regexp.MustCompile(`\b(new\s+(Shared)?Worker|new\s+WebSocket|new\s+EventSource|sendBeacon|importScripts)\b`),
			"a request to another host":    regexp.MustCompile(`\bfetch\s*\(\s*["'` + "`" + `](?:[a-z]+:)?//`),
			"a request to a relative path": regexp.MustCompile(`\bfetch\s*\(\s*["'` + "`" + `][^/]`),
		} {
			if m := re.FindString(js); m != "" {
				t.Errorf("%s uses %s (%q), which webUICSP blocks or which leaves this server", p, name, m)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if nFiles < 2 {
		t.Fatalf("read %d script files under webui/ui, want the renderer and the page script", nFiles)
	}
}

// TestWebUI_contentPolicyInBrowser runs scripts/webui_csp_smoke.mjs: the page from its real handlers, so headless Chrome
// enforces webUICSP, with a stub vision model behind it. The page must do its work with no violation, and each thing the
// policy forbids must be refused.
func TestWebUI_contentPolicyInBrowser(t *testing.T) {
	s := &server{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleWebUI)
	mux.HandleFunc("GET /ui/{file}", s.handleWebAsset)
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[{"id":"gate-model","vision":true}]}`)
	})
	// A request with an image goes to this route, streamed; the page sends text-only ones as jobs, which the stub has none of.
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, piece := range []string{"Hello ", "from ", "the stub."} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", piece)
			w.(http.Flusher).Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
	})
	ts := httptest.NewServer(mux)
	defer ts.Close()
	runBrowserGate(t, "../../scripts/webui_csp_smoke.mjs", "web UI content policy", ts.URL+"/")
}

// The admin socket has no key: its file mode is the lock. It is refused in a directory where someone else could swap
// the file, and it is at mode 0600, with nothing else left beside it, in one where it starts.
func TestAdminSocket_directoryAndMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the directory check reads Unix mode bits")
	}
	srv, err := newServer(config{})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	for _, c := range []struct {
		name string
		mode os.FileMode
		ok   bool
	}{
		{"only the owner can write it", 0o700, true},
		{"others can read it", 0o755, true},
		{"its group can write it", 0o770, false},
		{"everyone can write it", 0o777, false},
		{"everyone can write it, and it is sticky", 0o777 | os.ModeSticky, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := ownerOnlyDir(t)
			if err := os.Chmod(dir, c.mode); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, "admin.sock")
			closeSock, err := startAdminSocket(srv, path, 1<<20)
			if !c.ok {
				if err == nil {
					closeSock()
					t.Fatalf("startAdminSocket in a %04o directory succeeded, want a refusal", c.mode.Perm())
				}
				if _, statErr := os.Lstat(path); statErr == nil {
					t.Error("a refused start left a socket behind")
				}
				return
			}
			if err != nil {
				t.Fatalf("startAdminSocket in a %v directory: %v", c.mode, err)
			}
			defer closeSock()
			fi, err := os.Lstat(path)
			if err != nil {
				t.Fatal(err)
			}
			if fi.Mode()&os.ModeSocket == 0 || fi.Mode().Perm() != 0o600 {
				t.Errorf("the socket is %v, want a socket at mode 0600", fi.Mode())
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 1 {
				var names []string
				for _, e := range entries {
					names = append(names, e.Name())
				}
				t.Errorf("the directory holds %v, want only the socket", names)
			}
			// The socket was renamed into place after it was bound: it must answer at its final path.
			conn, err := net.Dial("unix", path)
			if err != nil {
				t.Fatalf("dial the socket at its final path: %v", err)
			}
			conn.Close()
			resp, err := unixHTTPClient(path).Get("http://admin-socket/admin/status")
			if err != nil {
				t.Fatalf("GET /admin/status over the socket: %v", err)
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Errorf("GET /admin/status over the socket: %d, want 200", resp.StatusCode)
			}
		})
	}
	// A stale socket from a crashed process is replaced.
	dir := ownerOnlyDir(t)
	path := filepath.Join(dir, "admin.sock")
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	closeSock, err := startAdminSocket(srv, path, 1<<20)
	if err != nil {
		t.Fatalf("startAdminSocket over a stale file: %v", err)
	}
	closeSock()
	if _, err := os.Lstat(path); err == nil {
		t.Error("close left the socket file behind")
	}
}
