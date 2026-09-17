package pull

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// withMockHF points hfAPI at a local httptest server for the duration of the test, restoring the
// real endpoint afterward — hfAPI is a var (not a const) specifically for this.
func withMockHF(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	orig := hfAPI
	hfAPI = srv.URL
	t.Cleanup(func() { hfAPI = orig })
	return srv
}

func TestSearch_realResponseShape(t *testing.T) {
	var gotQuery string
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.RawQuery
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "downloads": 12345, "likes": 67, "private": false},
			{"id": "someone/private-repo", "downloads": 1, "likes": 0, "private": true},
		})
	})
	got, err := Search(context.Background(), "qwen", "gguf", 20)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].Repo != "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF" || got[0].Downloads != 12345 || got[0].Likes != 67 {
		t.Fatalf("Search returned %+v", got)
	}
	if !containsAll(gotQuery, "search=qwen", "filter=gguf", "limit=20") {
		t.Errorf("query params sent to HF = %q, missing an expected piece", gotQuery)
	}
}

func TestSearch_privateHitsAreDropped(t *testing.T) {
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{
			{"id": "a/b", "private": true},
			{"id": "c/d", "private": false},
		})
	})
	got, err := Search(context.Background(), "x", "gguf", 0)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].Repo != "c/d" {
		t.Fatalf("Search returned %+v, want only the non-private hit", got)
	}
}

func TestSearch_unknownKindNeverMakesARequest(t *testing.T) {
	called := false
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) { called = true })
	_, err := Search(context.Background(), "x", "safetensors", 0)
	if err == nil {
		t.Fatal("Search with an unknown kind returned no error")
	}
	if called {
		t.Error("Search with an unknown kind still made an HTTP request")
	}
}

func TestSearch_emptyQueryNeverMakesARequest(t *testing.T) {
	called := false
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) { called = true })
	got, err := Search(context.Background(), "", "gguf", 0)
	if err != nil || got != nil {
		t.Fatalf("Search(\"\", ...) = %v, %v, want nil, nil", got, err)
	}
	if called {
		t.Error("Search with an empty query still made an HTTP request")
	}
}

func TestSearch_hfErrorStatus(t *testing.T) {
	withMockHF(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusTooManyRequests) })
	_, err := Search(context.Background(), "x", "gguf", 0)
	if err == nil {
		t.Fatal("Search against a 429 returned no error")
	}
}

func containsAll(s string, parts ...string) bool {
	for _, p := range parts {
		if !strings.Contains(s, p) {
			return false
		}
	}
	return true
}
