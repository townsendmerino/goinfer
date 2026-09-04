package serveapp

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// TestWebUI_disabledByDefault pins that the -web routes refuse when the flag is off. The
// page itself is inert, but handleWebPull starts a caller-named multi-gigabyte download and
// writes it to disk, so "off unless asked for" is the security property, not a preference.
func TestWebUI_disabledByDefault(t *testing.T) {
	s := &server{cfg: config{web: false}}
	for name, h := range map[string]http.HandlerFunc{
		"list": s.handleWebList,
		"pull": s.handleWebPull,
	} {
		w := httptest.NewRecorder()
		h(w, httptest.NewRequest(http.MethodPost, "/web/models/"+name, strings.NewReader(`{"repo":"a/b","quant":"q4_k_m"}`)))
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

// TestWebUI_pageIsSelfContained guards the offline property: the UI of an engine that runs
// offline must not need the network to RENDER. A CDN <script>/<link> would break that
// silently — the page would still look fine on the machine that added it.
//
// This does NOT forbid an <a href="https://…"> — an out-bound link the user may click (the
// AmbientCSS restyle, docs/task-web-ui-ambient.md, added one to the published book) does not
// cost the page anything at render time; only an asset the page's own load depends on does.
// A blanket "no http(s):// substring anywhere" check would have banned that link too, which is
// a different property than the one this test is for.
func TestWebUI_pageIsSelfContained(t *testing.T) {
	page := string(webUIPage)
	if len(page) == 0 {
		t.Fatal("embedded page is empty")
	}
	for _, bad := range []string{
		"<script src=\"http", "<script src='http",
		"<link rel=\"stylesheet\" href", "//cdn", "integrity=",
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
		wg.Add(1)
		go func() {
			defer wg.Done()
			if p.acquire() {
				mu.Lock()
				won++
				mu.Unlock()
			}
		}()
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
	var rootAuthed, listAuthed, pullAuthed bool
	var rootFound, listFound, pullFound bool
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
		case `"POST /web/models/list"`:
			listFound = true
			listAuthed = wrapsInAuth(call.Args[1])
		case `"POST /web/models/pull"`:
			pullFound = true
			pullAuthed = wrapsInAuth(call.Args[1])
		}
		return true
	})
	if !rootFound || !listFound || !pullFound {
		t.Fatalf("route(s) not found (root=%v list=%v pull=%v) — this guard is watching nothing",
			rootFound, listFound, pullFound)
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
	var listFound, pullFound, listSO, pullSO bool
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
		}
		return true
	})
	if !listFound || !pullFound {
		t.Fatalf("route(s) not found (list=%v pull=%v) — this guard is watching nothing", listFound, pullFound)
	}
	if !listSO {
		t.Error("POST /web/models/list is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could list a repo's files on the key-free loopback default (V-20)")
	}
	if !pullSO {
		t.Error("POST /web/models/pull is not wrapped in sameOrigin(...) — a cross-origin POST " +
			"could start a caller-named multi-GB download on the key-free loopback default (V-20)")
	}
}
