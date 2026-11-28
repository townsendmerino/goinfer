package serveapp

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// H2: /health carries each backend's startup self-test, so an operator sees that a backend declined or a kernel tier was stepped down without reading a log. The field is always an array, and a failing
// entry names the mismatch and what was stepped off. A model load runs the CPU test, so a served model always has a cpu entry.
func TestServe_healthReportsSelfTests(t *testing.T) {
	srv, _ := tinyServed(t) // loads a model, which runs the cpu self-test
	decoder.RecordSelfTest(decoder.SelfTestResult{Backend: "healthtest-gpu", Status: decoder.SelfTestDeclined,
		Mismatches: []string{"attention: observed 0.37, allowed 0.001"}, Disabled: []string{"avx2"}})
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", srv.handleHealth)
	ts := httptest.NewServer(mux)
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var h struct {
		Selftest []struct {
			Backend     string   `json:"backend"`
			Status      string   `json:"status"`
			Mismatches  []string `json:"mismatches"`
			SteppedDown []string `json:"stepped_down"`
		} `json:"selftest"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		t.Fatal(err)
	}
	byName := map[string]int{}
	for i, e := range h.Selftest {
		byName[e.Backend] = i
	}
	cpu, ok := byName["cpu"]
	if !ok {
		t.Fatalf("/health has no cpu self-test entry: %+v", h.Selftest)
	}
	if s := h.Selftest[cpu].Status; s != decoder.SelfTestPass && s != decoder.SelfTestSkipped {
		t.Errorf("a healthy host's cpu self-test is %q", s)
	}
	bad, ok := byName["healthtest-gpu"]
	if !ok {
		t.Fatalf("a declined backend is missing from /health: %+v", h.Selftest)
	}
	e := h.Selftest[bad]
	if e.Status != decoder.SelfTestDeclined || len(e.Mismatches) != 1 || len(e.SteppedDown) != 1 {
		t.Errorf("the declined entry must carry its status, mismatch and stepped-down tier: %+v", e)
	}
}

// /health's status follows the halt gate, through the production routes: "ok" while a generation route accepts a
// request and "halted" while it answers 503, for an admin halt and for a tripped swap guard, and the body says which.
// /health itself answers 200 in every state.
func TestServe_healthStatusFollowsTheHaltGate(t *testing.T) {
	cl, _, closer := wireServer(t)
	defer closer.Close()
	srv := closer.(*handlerCloser).srv
	srv.cfg.allowAdmin = true // an in-process handler takes no -allow-admin; requireAdmin reads the field per request

	do := func(method, path, body string) (int, string) {
		t.Helper()
		req, err := http.NewRequest(method, "http://goinfer"+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		resp, err := cl.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", method, path, err)
		}
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("%s %s: %v", method, path, err)
		}
		return resp.StatusCode, string(raw)
	}
	type report struct {
		Status           string  `json:"status"`
		Halted           *bool   `json:"halted"`
		HaltReason       *string `json:"halt_reason"`
		SwapGuardTripped *bool   `json:"swap_guard_tripped"`
	}
	read := func(path string) report {
		t.Helper()
		code, raw := do("GET", path, "")
		if code != http.StatusOK {
			t.Fatalf("GET %s answered %d %s, want 200 in every state", path, code, raw)
		}
		var r report
		if err := json.Unmarshal([]byte(raw), &r); err != nil {
			t.Fatalf("GET %s: %v in %s", path, err, raw)
		}
		if r.Halted == nil || r.SwapGuardTripped == nil {
			t.Fatalf("GET %s must always carry halted and swap_guard_tripped: %s", path, raw)
		}
		return r
	}
	const chat = `{"model":"tiny","max_tokens":1,"temperature":0,"logit_bias":{"66":100},"messages":[{"role":"user","content":"hi"}]}`
	check := func(when, wantStatus string, wantHalted, wantTripped bool, wantReason string) {
		t.Helper()
		h := read("/health")
		if h.Status != wantStatus || *h.Halted != wantHalted || *h.SwapGuardTripped != wantTripped {
			t.Errorf("%s: /health says status=%q halted=%v swap_guard_tripped=%v, want %q %v %v",
				when, h.Status, *h.Halted, *h.SwapGuardTripped, wantStatus, wantHalted, wantTripped)
		}
		if a := read("/admin/status"); *a.Halted != wantHalted || *a.SwapGuardTripped != wantTripped {
			t.Errorf("%s: /admin/status says halted=%v swap_guard_tripped=%v, want %v %v",
				when, *a.Halted, *a.SwapGuardTripped, wantHalted, wantTripped)
		}
		code, body := do("POST", "/v1/chat/completions", chat)
		refused := code == http.StatusServiceUnavailable
		if refused != (wantStatus == "halted") || (!refused && code != http.StatusOK) {
			t.Errorf("%s: a chat request answered %d %s", when, code, body)
		}
		if (h.Status == "ok") == refused {
			t.Errorf("%s: /health says status=%q and the chat route answered %d", when, h.Status, code)
		}
		if refused && !strings.Contains(body, wantReason) {
			t.Errorf("%s: the 503 does not give the reason %q: %s", when, wantReason, body)
		}
	}

	check("at start", "ok", false, false, "")
	if code, body := do("POST", "/admin/halt", `{"reason":"drill"}`); code != http.StatusOK {
		t.Fatalf("POST /admin/halt: %d %s", code, body)
	}
	check("after an admin halt", "halted", true, false, "drill")
	if r := read("/health").HaltReason; r == nil || *r != "drill" {
		t.Errorf("after an admin halt: halt_reason is %v, want drill", r)
	}
	if code, body := do("POST", "/admin/resume", ""); code != http.StatusOK {
		t.Fatalf("POST /admin/resume: %d %s", code, body)
	}
	check("after the resume", "ok", false, false, "")

	srv.swapGuardTripped.Store(true)
	check("with the swap guard tripped", "halted", false, true, "swap guard tripped")
	if r := read("/health").HaltReason; r != nil {
		t.Errorf("with the swap guard tripped: halt_reason is %q, want null (no admin halt is set)", *r)
	}
	srv.swapGuardTripped.Store(false)
	check("after the swap guard recovers", "ok", false, false, "")
}
