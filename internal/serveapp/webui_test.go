package serveapp

import (
	"encoding/json"
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

// TestWebUI_rejectsBadRepo is the network boundary for the traversal fix in modelpull:
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
// offline must not need the network to render. A CDN <script>/<link> would break that
// silently — the page would still look fine on the machine that added it.
func TestWebUI_pageIsSelfContained(t *testing.T) {
	page := string(webUIPage)
	if len(page) == 0 {
		t.Fatal("embedded page is empty")
	}
	for _, bad := range []string{"http://", "https://", "//cdn", "<link rel=\"stylesheet\" href", "integrity="} {
		if strings.Contains(page, bad) {
			t.Errorf("embedded page references %q — it must be fully self-contained (no external assets)", bad)
		}
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
