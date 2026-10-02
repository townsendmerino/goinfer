package serveapp

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// R25 gate (docs/tasks/task-first-hour.md): with the flag on, a request through the real chat handler writes one line naming the route, the model, the status,
// the prompt and completion tokens the response itself reported, and a first-token time. The tiny committed fixture runs it with no asset.
func TestRequestLog_oneLinePerGenerationRequest(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "tiny-qwen2-moe") // a committed HF checkpoint dir WITH a tokenizer (the tiny GGUFs have none)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	srv, err := newServer(config{models: modelFlag{{name: "tiny", path: p}}, load: loadflags.Flags{Backend: "cpu", Quant: "int8int8"}})
	if err != nil {
		t.Fatalf("newServer on the tiny fixture: %v", err)
	}
	var logbuf bytes.Buffer
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", logRequests(&logbuf, srv.handleChat))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	post := func(body string) (*http.Response, []byte) {
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp, b
	}
	lines := func() []string { return strings.Split(strings.TrimRight(logbuf.String(), "\n"), "\n") }
	re := regexp.MustCompile(`^request: POST /v1/chat/completions model=(\S+) status=(\d+) prompt_tokens=(\S+) completion_tokens=(\S+) ttft=(\S+) total=(\d+)ms$`)

	// 1. A normal request: the line's counts are the response's own usage.
	resp, body := post(`{"model":"tiny","max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"Hi"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var out struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(body, &out)
	ls := lines()
	if len(ls) != 1 {
		t.Fatalf("one request wrote %d log lines: %q", len(ls), logbuf.String())
	}
	m := re.FindStringSubmatch(ls[0])
	if m == nil {
		t.Fatalf("the line does not have the documented shape: %q", ls[0])
	}
	if m[1] != "tiny" || m[2] != "200" {
		t.Errorf("model/status = %s/%s: %q", m[1], m[2], ls[0])
	}
	if m[3] != strconv.Itoa(out.Usage.PromptTokens) || m[4] != strconv.Itoa(out.Usage.CompletionTokens) {
		t.Errorf("tokens %s/%s, the response said %d/%d: %q", m[3], m[4], out.Usage.PromptTokens, out.Usage.CompletionTokens, ls[0])
	}
	if out.Usage.PromptTokens == 0 || out.Usage.CompletionTokens == 0 {
		t.Fatalf("the response reported no tokens — the comparison above proves nothing: %s", body)
	}
	if !strings.HasSuffix(m[5], "ms") {
		t.Errorf("no first-token time for a request that generated: %q", ls[0])
	}

	// 2. Streaming: the status recorder must still let the handler flush, and the line still comes once, at the end.
	logbuf.Reset()
	resp, body = post(`{"model":"tiny","max_tokens":3,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Hi"}]}`)
	if resp.StatusCode != 200 || !strings.Contains(string(body), "data: ") {
		t.Fatalf("stream broke under the log wrapper: %d %q", resp.StatusCode, body)
	}
	if ls = lines(); len(ls) != 1 || re.FindStringSubmatch(ls[0]) == nil || strings.Contains(ls[0], "prompt_tokens=-") {
		t.Errorf("streamed request line: %q", logbuf.String())
	}

	// 3. A request that fails before generating: its status, and "-" for what does not exist.
	logbuf.Reset()
	resp, _ = post(`{"model":"tiny","messages":`)
	if resp.StatusCode != 400 {
		t.Fatalf("a malformed body answered %d, want 400", resp.StatusCode)
	}
	ls = lines()
	if len(ls) != 1 || !strings.Contains(ls[0], "model=- status=400 prompt_tokens=- completion_tokens=- ttft=-") {
		t.Errorf("failed request line: %q", logbuf.String())
	}

	// 4. Off (nil writer) is the handler itself: it sees no trace in its context, its writer is not wrapped, and nothing is logged.
	logbuf.Reset()
	probed := false
	var rw http.ResponseWriter
	probe := func(w http.ResponseWriter, r *http.Request) {
		probed, rw = true, w
		if traceFrom(r.Context()) != nil {
			t.Error("the flag off still put a trace in the request context")
		}
	}
	rec := httptest.NewRecorder()
	logRequests(nil, probe)(rec, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if !probed || rw != http.ResponseWriter(rec) {
		t.Errorf("the flag off wrapped the response writer (%T) or never ran the handler", rw)
	}
	if logbuf.Len() != 0 {
		t.Errorf("wrote %q", logbuf.String())
	}
}

// The wiring half, as source: the flag exists and the four generation routes are wrapped with it, outermost. (The mux is built inside the serve entry point,
// so the middleware above is proven by calling it and this proves something calls it.)
func TestRequestLog_wiredIntoTheGenerationRoutes(t *testing.T) {
	src, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	if !strings.Contains(body, `"log-requests"`) || !strings.Contains(body, "cfg.logRequests") {
		t.Error("the -log-requests flag is not registered into cfg.logRequests")
	}
	for _, route := range []string{"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages"} {
		want := `mux.HandleFunc("POST ` + route + `", rl(auth(`
		if !strings.Contains(body, want) {
			t.Errorf("%s is not wrapped with the request log (want %q)", route, want)
		}
	}
}
