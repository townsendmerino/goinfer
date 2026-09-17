package serveapp

import (
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/pull"
)

// TestWebUI_disabledByDefault pins that the -web routes refuse when the flag is off. The
// page itself is inert, but handleWebPull starts a caller-named multi-gigabyte download and
// writes it to disk, so "off unless asked for" is the security property, not a preference.
func TestWebUI_disabledByDefault(t *testing.T) {
	s := &server{cfg: config{web: false}}
	for name, h := range map[string]http.HandlerFunc{
		"list":   s.handleWebList,
		"pull":   s.handleWebPull,
		"load":   s.handleWebLoad,
		"unload": s.handleWebUnload,
		"search": s.handleWebSearch,
	} {
		w := httptest.NewRecorder()
		body := `{"repo":"a/b","quant":"q4_k_m"}`
		if name == "search" {
			body = `{"query":"qwen","kind":"gguf"}`
		}
		h(w, httptest.NewRequest(http.MethodPost, "/web/models/"+name, strings.NewReader(body)))
		if w.Code != http.StatusForbidden {
			t.Errorf("%s with -web off: status %d, want 403", name, w.Code)
		}
		if !strings.Contains(w.Body.String(), "-web") {
			t.Errorf("%s: error should name the flag that enables it, got %s", name, w.Body.String())
		}
	}
}

// TestWebUI_rejectsBadRepo is the network boundary for the traversal fix in the pull package:
// the repo name now arrives from an HTTP body, which is exactly the untrusted source the
// allow-list exists for. No network is reached — ParseRef fails first, which is the point.
func TestWebUI_rejectsBadRepo(t *testing.T) {
	s := &server{cfg: config{web: true}}
	for _, repo := range []string{"../..", "a/../../etc", "./x", "a b/c", "notarepo", ""} {
		body, _ := json.Marshal(map[string]string{"repo": repo, "quant": "q4_k_m"})
		w := httptest.NewRecorder()
		s.handleWebList(w, httptest.NewRequest(http.MethodPost, "/web/models/list", strings.NewReader(string(body))))
		if w.Code != http.StatusBadRequest {
			t.Errorf("repo %q: status %d, want 400", repo, w.Code)
		}
	}
}

// TestWebUI_pullRefReplacesSelector pins the fix for a real bug hit from the page: typing
// "owner/repo:q4_k_m" into the repo box (List works fine — handleWebList parses req.Repo alone)
// and then clicking a file button used to send {repo: "owner/repo:q4_k_m", file: "x.gguf"},
// concatenated into "owner/repo:q4_k_m:x.gguf" and re-parsed. pull.ParseRef cuts at the FIRST
// colon, so that became repo="owner/repo", selector="q4_k_m:x.gguf" — a string that still ends in
// ".gguf" and is therefore looked up as a literal filename no repo publishes, in a repo that
// actually has "x.gguf" under a normal name. Every Pull button failed while List kept working,
// which reads like a bad repo rather than a bad concatenation — exactly what the user hit.
func TestWebUI_pullRefReplacesSelector(t *testing.T) {
	const repo = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
	cases := []struct {
		name string
		req  webPullReq
		want pull.Ref
	}{
		{"a clicked file REPLACES the box's own selector, not appends to it",
			webPullReq{Repo: repo + ":q4_k_m", File: "qwen2.5-coder-1.5b-instruct-q8_0.gguf"},
			pull.Ref{Repo: repo, File: "qwen2.5-coder-1.5b-instruct-q8_0.gguf"}},
		{"a clicked file with no selector in the box (the path everyone tested by hand)",
			webPullReq{Repo: repo, File: "qwen2.5-coder-1.5b-instruct-q8_0.gguf"},
			pull.Ref{Repo: repo, File: "qwen2.5-coder-1.5b-instruct-q8_0.gguf"}},
		{"a clicked quant REPLACES the box's own selector the same way",
			webPullReq{Repo: repo + ":q4_k_m", Quant: "q8_0"},
			pull.Ref{Repo: repo, Quant: "q8_0"}},
	}
	for _, c := range cases {
		got, err := webPullRef(c.req)
		if err != nil || got != c.want {
			t.Errorf("%s:\n  webPullRef(%+v) = %+v, %v\n  want %+v, <nil>", c.name, c.req, got, err, c.want)
		}
	}

	// neither file nor quant clicked: the box's own parse returns AS-IS, so a resolved demo: tier
	// keeps its Pin/Bytes (re-parsing "repo:file" would drop them — ParseRef only ever sets those
	// for a literal "demo:tier" input).
	names := pull.CuratedNames()
	if len(names) == 0 {
		t.Fatal("no curated demo tiers to test against")
	}
	tier := pull.Curated()[names[0]]
	want := pull.Ref{Repo: tier.Repo, File: tier.File, Pin: tier.SHA256, Bytes: tier.Bytes}
	got, err := webPullRef(webPullReq{Repo: "demo:" + names[0]})
	if err != nil || got != want {
		t.Errorf("nothing clicked, demo:%s: webPullRef = %+v, %v, want %+v, <nil> (Pin/Bytes must survive)", names[0], got, err, want)
	}

	// an invalid repo still fails exactly as ParseRef reports it
	if _, err := webPullRef(webPullReq{Repo: "../.."}); err == nil {
		t.Error("webPullRef(\"../..\") = <nil> error, want ParseRef's own rejection")
	}
}

// TestWebUI_pageIsSelfContained guards the offline property: the UI of an engine that runs
// offline must not need the network to RENDER. A CDN <script>/<link> would break that
// silently — the page would still look fine on the machine that added it.
//
// This does NOT forbid an <a href="https://…"> — an out-bound link the user may click (the
// AmbientCSS restyle, docs/completed/task-web-ui-ambient.md, added one to the published book)
// does not cost the page anything at render time; only an asset the page's own load depends on
// does.
// A blanket "no http(s):// substring anywhere" check would have banned that link too, which is
// a different property than the one this test is for.
func TestWebUI_pageIsSelfContained(t *testing.T) {
	// EVERY EMBEDDED FILE, not just index.html. The page is split across webui/ (§6.1 of
	// docs/tasks/task-web-ui-2026-09.md); a check that still read only index.html would pass while
	// a CDN import sat in ui/app.js — narrowing silently exactly when the page grew.
	var all strings.Builder
	nFiles := 0
	if err := fs.WalkDir(webUIFS, "webui", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := webUIFS.ReadFile(p)
		if err != nil {
			return err
		}
		nFiles++
		all.WriteString("\n/* ==== " + p + " ==== */\n")
		all.Write(b)
		return nil
	}); err != nil {
		t.Fatalf("walk embedded web UI: %v", err)
	}
	if nFiles < 3 {
		t.Fatalf("embedded web UI has %d file(s), want index.html + ui/app.css + ui/app.js at least", nFiles)
	}
	page := all.String()
	if len(webUIPage) == 0 {
		t.Fatal("embedded index.html is empty")
	}
	// A LOCAL stylesheet link (href="ui/...") is now expected; an external one is still banned.
	for _, bad := range []string{
		"<script src=\"http", "<script src='http", "<script src=\"//", "<script src='//",
		"<link rel=\"stylesheet\" href=\"http", "<link rel=\"stylesheet\" href=\"//",
		"@import", "//cdn", "integrity=",
	} {
		if strings.Contains(page, bad) {
			t.Errorf("embedded page references %q — it must be fully self-contained (no external assets)", bad)
		}
	}
	// The two allowed external references, neither a render-time asset: an out-bound link to
	// the book, and a plain-text attribution comment naming the vendored CSS's source (never
	// fetched — browsers strip CSS comments). Both pinned exactly rather than left as
	// "anything goes" — a DIFFERENT http(s) reference slipping in later (a tracking pixel, a
	// font @import, a fetch to an analytics host) is still exactly the kind of silent
	// offline-break this test exists to catch.
	bookLink := `<a class="book-link" href="https://townsendmerino.github.io/goinfer/" target="_blank" rel="noopener">`
	if !strings.Contains(page, bookLink) {
		t.Errorf("embedded page's book link is missing or no longer matches the pinned shape: want %q", bookLink)
	}
	attribution := "https://github.com/kikkupico/ambientcss"
	if !strings.Contains(page, attribution) {
		t.Errorf("embedded page's AmbientCSS attribution comment is missing: want %q", attribution)
	}
	// Every asset index.html loads must be embedded and of a type the asset route serves — a
	// reference to a file that is not there fails in the browser with no build or test error.
	for _, m := range regexp.MustCompile(`(?:src|href)="(ui/[^"]+)"`).FindAllStringSubmatch(string(webUIPage), -1) {
		ref := m[1]
		if _, err := webUIFS.ReadFile("webui/" + ref); err != nil {
			t.Errorf("index.html references %q, which is not embedded under webui/", ref)
		}
		ext := ref[strings.LastIndex(ref, "."):]
		if _, ok := webUIAssetTypes[ext]; !ok {
			t.Errorf("index.html references %q, whose type %q the /ui/ route does not serve", ref, ext)
		}
	}
	if n := strings.Count(page, "http://") + strings.Count(page, "https://"); n != 2 {
		t.Errorf("embedded page has %d http(s):// reference(s), want exactly 2 (the book link, the "+
			"AmbientCSS attribution comment) — a new one needs the SAME scrutiny those two already "+
			"got, not a free pass", n)
	}
	// And it must actually be the UI, so this test cannot pass on an empty/placeholder file.
	for _, want := range []string{"/v1/chat/completions", "/web/models/pull", "<title>goinfer</title>"} {
		if !strings.Contains(page, want) {
			t.Errorf("embedded page is missing %q", want)
		}
	}
	for _, want := range []string{`href="ui/app.css"`, `src="ui/app.js"`} {
		if !strings.Contains(string(webUIPage), want) {
			t.Errorf("index.html no longer loads %s — the page would render unstyled or inert", want)
		}
	}
}

// TestWebUI_assetRoute drives handleWebAsset through a real ServeMux using the exact pattern main.go
// registers, so path cleaning and {file} matching are the real ones. The page is served from the
// API's own origin: the property is that ONLY allow-listed embedded assets come back, with an
// explicit type and nosniff, and everything else is a 404 — never a sniffed or listed file.
func TestWebUI_assetRoute(t *testing.T) {
	s := &server{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ui/{file}", s.handleWebAsset)

	for _, c := range []struct{ file, ctype string }{
		{"app.css", "text/css; charset=utf-8"},
		{"app.js", "text/javascript; charset=utf-8"},
	} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/ui/"+c.file, nil))
		if w.Code != http.StatusOK {
			t.Errorf("/ui/%s: status %d, want 200", c.file, w.Code)
			continue
		}
		if got := w.Header().Get("Content-Type"); got != c.ctype {
			t.Errorf("/ui/%s: Content-Type %q, want %q", c.file, got, c.ctype)
		}
		if w.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Errorf("/ui/%s: missing X-Content-Type-Options: nosniff", c.file)
		}
		want, _ := webUIFS.ReadFile("webui/ui/" + c.file)
		if w.Body.String() != string(want) || len(want) == 0 {
			t.Errorf("/ui/%s: body is not the embedded file (%d bytes served, %d embedded)", c.file, w.Body.Len(), len(want))
		}
	}

	// Refused paths. Traversal is checked by what comes BACK, not by status alone: ServeMux cleans
	// "/ui/../x" into a redirect, and the property is that no Go source or index.html is served.
	for _, p := range []string{
		"/ui/missing.js",    // not embedded
		"/ui/app.txt",       // type not allow-listed
		"/ui/.hidden.js",    // dotfile
		"/ui/",              // directory: no listing
		"/ui/../webui.go",   // traversal out of ui/
		"/ui/..%2fwebui.go", // encoded traversal
		"/ui/%2e%2e%2findex.html",
	} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(http.MethodGet, p, nil))
		body := w.Body.String()
		if w.Code == http.StatusOK {
			t.Errorf("%s: status 200, want it refused", p)
		}
		if strings.Contains(body, "package serveapp") || strings.Contains(body, "<title>goinfer</title>") {
			t.Errorf("%s: served a file outside the asset allow-list (status %d)", p, w.Code)
		}
	}
}

// TestWebUI_noHTMLStringSinks makes §6.2 of docs/tasks/task-web-ui-2026-09.md a build failure rather
// than a convention: nothing in the embedded page may turn a string into markup or code. The page is
// served from the API's own origin with the user's key on it and renders text from models, some
// pulled from strangers' repos — so model output goes in through createTextNode and createElement
// (ui/markdown.js), and every one of these sinks is banned outright, even where a particular use
// looks harmless today.
//
// It matches CODE patterns (an assignment to .innerHTML, a call to insertAdjacentHTML), not bare
// words, so a comment explaining the rule does not trip it.
func TestWebUI_noHTMLStringSinks(t *testing.T) {
	sinks := []struct{ name, re string }{
		{"innerHTML assignment", `\.innerHTML\s*\+?=`},
		{"outerHTML assignment", `\.outerHTML\s*\+?=`},
		{"insertAdjacentHTML", `\.insertAdjacentHTML\s*\(`},
		{"document.write", `document\.write(ln)?\s*\(`},
		{"eval", `(^|[^.\w])eval\s*\(`},
		{"new Function", `new\s+Function\s*\(`},
		{"srcdoc assignment", `\.srcdoc\s*=`},
		{"createContextualFragment", `createContextualFragment\s*\(`},
		{"DOMParser.parseFromString", `parseFromString\s*\(`},
		{"string-form setTimeout/setInterval", `set(Timeout|Interval)\s*\(\s*["'\x60]`},
	}
	checked := 0
	err := fs.WalkDir(webUIFS, "webui", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !(strings.HasSuffix(p, ".js") || strings.HasSuffix(p, ".html")) {
			return err
		}
		b, err := webUIFS.ReadFile(p)
		if err != nil {
			return err
		}
		checked++
		for _, sink := range sinks {
			if loc := regexp.MustCompile("(?m)" + sink.re).FindIndex(b); loc != nil {
				line := 1 + strings.Count(string(b[:loc[0]]), "\n")
				t.Errorf("%s:%d uses %s — the web UI must never turn a string into markup or code; "+
					"build nodes with createElement/createTextNode (see ui/markdown.js)", p, line, sink.name)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk embedded web UI: %v", err)
	}
	if checked < 3 {
		t.Fatalf("checked only %d file(s) — this guard would be watching almost nothing", checked)
	}
	app, _ := webUIFS.ReadFile("webui/ui/app.js")
	if !strings.Contains(string(app), "renderReply(out, acc, live, entry)") || !strings.Contains(string(app), "Markdown.render(out, text)") {
		t.Error("ui/app.js no longer renders model output through Markdown.render — W1's renderer is not wired into the chat")
	}
}

// TestWebUI_markdownGateInBrowser runs scripts/webui_md_gate.mjs: the SHIPPED ui/markdown.js, in headless
// Chrome, against hostile input (every payload rendered into live DOM, a canary that must never fire),
// pathological input (bounded render time), and structural correctness. Only a real browser can show
// that a payload did not execute, which is why this is not a pure-Go test.
func TestWebUI_markdownGateInBrowser(t *testing.T) {
	runBrowserGate(t, "../../scripts/webui_md_gate.mjs", "W1 markdown renderer")
}

// TestWebUI_appGateInBrowser runs scripts/webui_app_gate.mjs: the SHIPPED index.html driven through its own
// send() with a fake SSE stream — W1's streamed rendering end to end, and W2's Copy on both clipboard
// paths (async API, and the textarea fallback an insecure http://<lan-ip> page needs), on a stopped
// answer, and after a re-render rebuilds every code-block button.
func TestWebUI_appGateInBrowser(t *testing.T) {
	runBrowserGate(t, "../../scripts/webui_app_gate.mjs", "web UI app")
}

// runBrowserGate runs one of the web UI's headless-Chrome gates. They need node and a Chrome/Chromium
// binary, which GitHub's ubuntu runners have. Where either is missing the test SKIPS, loudly — and a
// skip is not a pass: run the script by hand after changing the page on such a machine. Gate exit 2
// means the browser could not be driven at all (also a loud skip); exit 1 is a real failure.
func runBrowserGate(t *testing.T, script, what string) {
	t.Helper()
	if testing.Short() {
		t.Skipf("%s browser gate skipped in -short", what)
	}
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skipf("SKIPPED — no node on PATH; the %s browser gate did NOT run (node %s)", what, strings.TrimPrefix(script, "../../"))
	}
	haveChrome := false
	for _, b := range []string{"google-chrome", "google-chrome-stable", "chromium", "chromium-browser"} {
		if _, err := exec.LookPath(b); err == nil {
			haveChrome = true
			break
		}
	}
	if !haveChrome {
		t.Skipf("SKIPPED — no Chrome/Chromium on PATH; the %s browser gate did NOT run", what)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, node, script)
	out, err := cmd.CombinedOutput()
	switch code := cmd.ProcessState.ExitCode(); {
	case err == nil:
		t.Logf("%s", lastLine(out))
	case code == 2:
		t.Skipf("SKIPPED — the browser could not be driven here, so the %s gate did NOT run:\n%s", what, out)
	default:
		t.Fatalf("%s gate FAILED (exit %d):\n%s", what, code, out)
	}
}

func lastLine(b []byte) string {
	s := strings.TrimSpace(string(b))
	if i := strings.LastIndex(s, "\n"); i >= 0 {
		return s[i+1:]
	}
	return s
}

// TestPullState_singleFlight pins the one-at-a-time bound. Without it a few clicks queue
// unbounded concurrent multi-gigabyte downloads against the same disk.
func TestPullState_singleFlight(t *testing.T) {
	var p pullState
	if !p.acquire() {
		t.Fatal("first acquire must succeed")
	}
	if p.acquire() {
		t.Fatal("second acquire must fail while the first is held")
	}
	p.release()
	if !p.acquire() {
		t.Fatal("acquire must succeed after release")
	}
	p.release()

	// Concurrently, exactly one winner. Run under -race to mean anything.
	var wg sync.WaitGroup
	var mu sync.Mutex
	won := 0
	for range 32 {
		wg.Go(func() {
			if p.acquire() {
				mu.Lock()
				won++
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	if won != 1 {
		t.Errorf("concurrent acquire: %d winners, want exactly 1", won)
	}
}

// TestWebUI_rootRouteIsUnauthenticated guards V-02 (docs/review-2026-09-04.md): GET /{$} used to
// be wrapped in auth(...), so a browser's plain navigation -- which sends no Authorization header
// -- got the 401 JSON instead of the page, whenever -api-key was set (required off loopback). The
// page is the ONLY place a user could type the key in, so this was a deadlock: loading the page
// needed the key, and there was nowhere to enter the key without the page. auth stays on
// /web/models/list and /web/models/pull, which actually act.
//
// main()'s mux-building is inline, not a separately testable function (this is exactly why the
// bug went unguarded -- webui_test.go could exercise handleWebUI directly but never through the
// auth-wrapped mux registration), so this is asserted structurally: mux.HandleFunc("GET /{$}", ...)
// must NOT wrap its handler in the auth closure, while the /web/models/* registrations must.
func TestWebUI_rootRouteIsUnauthenticated(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var rootAuthed, listAuthed, pullAuthed, assetAuthed, loadAuthed bool
	var rootFound, listFound, pullFound, assetFound, loadFound bool
	ast.Inspect(af, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "HandleFunc" || len(call.Args) != 2 {
			return true
		}
		route, ok := call.Args[0].(*ast.BasicLit)
		if !ok {
			return true
		}
		// Searches the WHOLE wrapper chain, not just the outermost call — V-20
		// (docs/review-2026-09-04.md) nested list/pull one layer deeper as
		// sameOrigin(auth(maxBytes(...))), and a check anchored on the outermost call alone
		// would have silently stopped seeing auth(...) the moment that landed.
		wrapsInAuth := func(e ast.Expr) bool {
			found := false
			ast.Inspect(e, func(n ast.Node) bool {
				c, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				if id, ok := c.Fun.(*ast.Ident); ok && id.Name == "auth" {
					found = true
				}
				return true
			})
			return found
		}
		switch route.Value {
		case `"GET /{$}"`:
			rootFound = true
			rootAuthed = wrapsInAuth(call.Args[1])
		case `"GET /ui/{file}"`:
			assetFound = true
			assetAuthed = wrapsInAuth(call.Args[1])
		case `"POST /web/models/list"`:
			listFound = true
			listAuthed = wrapsInAuth(call.Args[1])
		case `"POST /web/models/pull"`:
			pullFound = true
			pullAuthed = wrapsInAuth(call.Args[1])
		case `"POST /web/models/load"`:
			loadFound = true
			loadAuthed = wrapsInAuth(call.Args[1])
		}
		return true
	})
	if !rootFound || !listFound || !pullFound || !assetFound || !loadFound {
		t.Fatalf("route(s) not found (root=%v list=%v pull=%v asset=%v load=%v) — this guard is watching nothing",
			rootFound, listFound, pullFound, assetFound, loadFound)
	}
	if !loadAuthed {
		t.Error("POST /web/models/load lost its auth(...) wrapping — this route loads a model into " +
			"the server's memory and must stay behind the API key")
	}
	if assetAuthed {
		t.Error("GET /ui/{file} is wrapped in auth(...) — the page's CSS and JS are plain subresource " +
			"loads that send no Authorization header, so with -api-key set the page would load " +
			"unstyled and inert, and the key field would never work (V-02)")
	}
	if rootAuthed {
		t.Error("GET /{$} is wrapped in auth(...) — a browser's plain navigation sends no " +
			"Authorization header, so the page (the only place to type the key in) 401s " +
			"whenever -api-key is set, and there is no way to ever load it (V-02)")
	}
	if !listAuthed {
		t.Error("POST /web/models/list lost its auth(...) wrapping — this route lists a repo " +
			"and should stay behind the API key")
	}
	if !pullAuthed {
		t.Error("POST /web/models/pull lost its auth(...) wrapping — this route starts a " +
			"caller-named multi-GB download and must stay behind the API key")
	}
}

// TestWebUI_pageRoutesExistOnlyUnderWebFlag pins that the page and its assets are registered INSIDE
// main.go's `if cfg.web { ... }` block. The page and /ui/ assets are harmless on their own, but
// "off unless -web" is the documented contract for the whole UI surface, and an asset route that
// drifted out of the block would serve UI files from every server. Found by AST, so it follows the
// code rather than a line number.
func TestWebUI_pageRoutesExistOnlyUnderWebFlag(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	inWeb := map[string]bool{}
	anywhere := map[string]int{}
	routeOf := func(n ast.Node) (string, bool) {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return "", false
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "HandleFunc" || len(call.Args) != 2 {
			return "", false
		}
		lit, ok := call.Args[0].(*ast.BasicLit)
		if !ok {
			return "", false
		}
		return lit.Value, true
	}
	ast.Inspect(af, func(n ast.Node) bool {
		if r, ok := routeOf(n); ok {
			anywhere[r]++
		}
		ifs, ok := n.(*ast.IfStmt)
		if !ok {
			return true
		}
		sel, ok := ifs.Cond.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "web" {
			return true
		}
		if x, ok := sel.X.(*ast.Ident); !ok || x.Name != "cfg" {
			return true
		}
		ast.Inspect(ifs.Body, func(m ast.Node) bool {
			if r, ok := routeOf(m); ok {
				inWeb[r] = true
			}
			return true
		})
		return true
	})
	for _, r := range []string{`"GET /{$}"`, `"GET /ui/{file}"`, `"POST /web/models/list"`, `"POST /web/models/pull"`, `"POST /web/models/load"`} {
		if anywhere[r] == 0 {
			t.Fatalf("route %s not registered anywhere — this guard is watching nothing", r)
		}
		if !inWeb[r] || anywhere[r] != 1 {
			t.Errorf("route %s must be registered exactly once, inside `if cfg.web { ... }` "+
				"(inside=%v, registrations=%d) — the UI surface is off unless -web", r, inWeb[r], anywhere[r])
		}
	}
}

// TestSameOrigin_refusesForeignOriginAllowsMatchingOrNone pins V-20 (docs/review-2026-09-04.md):
// on the key-free loopback default, auth() alone is a no-op (requireAuth returns h unchanged
// when key==""), so list/pull had NO protection against a cross-origin POST — any page open in
// the same browser could drive a caller-named multi-GB download onto the user's disk. Mirrors
// N-26's identical sameOrigin in demo/agent/cmd/agent-web/main.go.
func TestSameOrigin_refusesForeignOriginAllowsMatchingOrNone(t *testing.T) {
	called := false
	h := sameOrigin(func(w http.ResponseWriter, r *http.Request) { called = true })

	for _, tc := range []struct {
		name     string
		origin   string
		wantCode int
		wantCall bool
	}{
		{"no Origin header (curl, a same-origin form post outside a browser)", "", http.StatusOK, true},
		{"matching Origin", "http://127.0.0.1:8080", http.StatusOK, true},
		{"foreign Origin (the CSRF-style attack)", "https://evil.example", http.StatusForbidden, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called = false
			r := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:8080/web/models/pull", strings.NewReader("{}"))
			if tc.origin != "" {
				r.Header.Set("Origin", tc.origin)
			}
			w := httptest.NewRecorder()
			h(w, r)
			if w.Code != tc.wantCode {
				t.Errorf("status = %d, want %d", w.Code, tc.wantCode)
			}
			if called != tc.wantCall {
				t.Errorf("handler called = %v, want %v", called, tc.wantCall)
			}
		})
	}
}

// TestWebUI_listAndPullAreWrappedInSameOrigin is the wiring guard: the unit test above proves
// sameOrigin works in isolation, but that says nothing about whether the actual routes call it —
// the exact shape of gap this session's audit keeps finding (a helper with a test, and a call
// site nobody checked).
func TestWebUI_listAndPullAreWrappedInSameOrigin(t *testing.T) {
	fset := token.NewFileSet()
	af, err := parser.ParseFile(fset, "main.go", nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var listFound, pullFound, loadFound, unloadFound, searchFound, listSO, pullSO, loadSO, unloadSO, searchSO bool
	ast.Inspect(af, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "HandleFunc" || len(call.Args) != 2 {
			return true
		}
		route, ok := call.Args[0].(*ast.BasicLit)
		if !ok {
			return true
		}
		outer, ok := call.Args[1].(*ast.CallExpr)
		isSameOrigin := false
		if ok {
			if id, ok := outer.Fun.(*ast.Ident); ok && id.Name == "sameOrigin" {
				isSameOrigin = true
			}
		}
		switch route.Value {
		case `"POST /web/models/list"`:
			listFound = true
			listSO = isSameOrigin
		case `"POST /web/models/pull"`:
			pullFound = true
			pullSO = isSameOrigin
		case `"POST /web/models/load"`:
			loadFound = true
			loadSO = isSameOrigin
		case `"POST /web/models/unload"`:
			unloadFound = true
			unloadSO = isSameOrigin
		case `"POST /web/models/search"`:
			searchFound = true
			searchSO = isSameOrigin
		}
		return true
	})
	if !listFound || !pullFound || !loadFound || !unloadFound || !searchFound {
		t.Fatalf("route(s) not found (list=%v pull=%v load=%v unload=%v search=%v) — this guard is watching nothing", listFound, pullFound, loadFound, unloadFound, searchFound)
	}
	if !loadSO {
		t.Error("POST /web/models/load is not wrapped in sameOrigin(...) — a cross-origin POST could " +
			"make the server load a pulled model into memory on the key-free loopback default (V-20)")
	}
	if !listSO {
		t.Error("POST /web/models/list is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could list a repo's files on the key-free loopback default (V-20)")
	}
	if !pullSO {
		t.Error("POST /web/models/pull is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could start a caller-named multi-GB download on the key-free loopback default (V-20)")
	}
	if !unloadSO {
		t.Error("POST /web/models/unload is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could free a resident model on the key-free loopback default (V-20), W32")
	}
	if !searchSO {
		t.Error("POST /web/models/search is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could make this server proxy arbitrary HuggingFace searches on the key-free loopback default (V-20)")
	}
}
