package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestRequireAuth gates B-14: no key → pass-through; key set → only the correct
// secret (via Bearer or x-api-key) passes, everything else is 401.
func TestRequireAuth(t *testing.T) {
	ok := func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }
	call := func(h http.HandlerFunc, set func(*http.Request)) int {
		r := httptest.NewRequest("POST", "/x", nil)
		if set != nil {
			set(r)
		}
		w := httptest.NewRecorder()
		h(w, r)
		return w.Code
	}
	// key == "" → auth disabled, everything passes.
	if c := call(requireAuth("", ok), nil); c != http.StatusOK {
		t.Errorf("no-key pass-through: got %d", c)
	}
	h := requireAuth("s3cret", ok)
	cases := []struct {
		name string
		set  func(*http.Request)
		want int
	}{
		{"no header", nil, http.StatusUnauthorized},
		{"wrong bearer", func(r *http.Request) { r.Header.Set("Authorization", "Bearer nope") }, http.StatusUnauthorized},
		{"wrong x-api-key", func(r *http.Request) { r.Header.Set("x-api-key", "nope") }, http.StatusUnauthorized},
		{"right bearer", func(r *http.Request) { r.Header.Set("Authorization", "Bearer s3cret") }, http.StatusOK},
		{"right x-api-key", func(r *http.Request) { r.Header.Set("x-api-key", "s3cret") }, http.StatusOK},
	}
	for _, tc := range cases {
		if c := call(h, tc.set); c != tc.want {
			t.Errorf("%s: got %d want %d", tc.name, c, tc.want)
		}
	}
}

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
