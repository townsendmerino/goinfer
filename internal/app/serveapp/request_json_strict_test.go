package serveapp

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Model config files accept Python's Infinity and NaN (decoder.loadConfig); request bodies do not. A body carrying either is malformed
// JSON and is refused with 400 by the decoder every handler uses, whatever field it sits in.
func TestRequestJSON_nonFiniteStaysRejected(t *testing.T) {
	h := maxBytes(1<<10, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model       string   `json:"model"`
			Temperature *float64 `json:"temperature"`
			Stop        []string `json:"stop"`
		}
		if !decodeJSON(w, r, &req) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"model": req.Model})
	})
	ts := httptest.NewServer(h)
	defer ts.Close()
	post := func(body string) int {
		resp, err := http.Post(ts.URL, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		defer resp.Body.Close()
		return resp.StatusCode
	}
	if code := post(`{"model":"m","temperature":0.5}`); code != http.StatusOK {
		t.Fatalf("VOID: a valid body got %d, so the handler no longer decodes what this test sends", code)
	}
	for _, body := range []string{
		`{"model":"m","temperature":Infinity}`,
		`{"model":"m","temperature":-Infinity}`,
		`{"model":"m","temperature":NaN}`,
		`{"model":Infinity}`,
	} {
		if code := post(body); code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400", body, code)
		}
	}
	// the same words inside a string are just text
	if code := post(`{"model":"Infinity","stop":["NaN","-Infinity"]}`); code != http.StatusOK {
		t.Errorf("tokens inside strings: got %d, want 200", code)
	}
}
