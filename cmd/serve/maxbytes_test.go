package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestMaxBytesAndDecode gates M3: a body over the cap is rejected 413 (not
// buffered whole), a malformed body is 400, and a valid body passes.
func TestMaxBytesAndDecode(t *testing.T) {
	h := maxBytes(1<<10, func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
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

	if code := post(`{"model":"m"}`); code != http.StatusOK {
		t.Errorf("valid body: status %d, want 200", code)
	}
	if code := post(`{"model":`); code != http.StatusBadRequest {
		t.Errorf("malformed body: status %d, want 400", code)
	}
	over := `{"model":"` + strings.Repeat("x", 4096) + `"}`
	if code := post(over); code != http.StatusRequestEntityTooLarge {
		t.Errorf("oversize body (%d bytes): status %d, want 413", len(over), code)
	}
}
