package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
