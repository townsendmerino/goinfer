package serveapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// TestServe_admin is the Inc3 gate: --allow-admin off → 403; with it on, a
// load → generate → unload → 404 → reload cycle, and unload-while-busy → 409.
// Gated on GOINFER_SERVE_MODEL (a .gguf).
func TestServe_admin(t *testing.T) {
	path := os.Getenv("GOINFER_SERVE_MODEL")
	if path == "" {
		t.Skip("set GOINFER_SERVE_MODEL=<.gguf> for the admin test")
	}
	mkmux := func(srv *server) *httptest.Server {
		mux := http.NewServeMux()
		mux.HandleFunc("POST /v1/chat/completions", srv.handleChat)
		mux.HandleFunc("GET /v1/models", srv.handleModels)
		mux.HandleFunc("POST /admin/models/load", srv.handleAdminLoad)
		mux.HandleFunc("POST /admin/models/unload", srv.handleAdminUnload)
		return httptest.NewServer(mux)
	}
	post := func(ts *httptest.Server, p, body string) int {
		r, err := http.Post(ts.URL+p, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST %s: %v", p, err)
		}
		r.Body.Close()
		return r.StatusCode
	}

	// --allow-admin off → 403.
	noAdmin, _ := newServer(config{})
	tsNo := mkmux(noAdmin)
	if code := post(tsNo, "/admin/models/load", `{"path":"`+path+`"}`); code != http.StatusForbidden {
		t.Errorf("admin off: load status %d, want 403", code)
	}
	tsNo.Close()

	// Admin-enabled, starts empty.
	srv, err := newServer(config{allowAdmin: true, backend: "cpu", quant: "int8int8", kvSessions: 2})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	ts := mkmux(srv)
	defer ts.Close()
	chat := func() int {
		r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"t","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatalf("chat POST: %v", err)
		}
		r.Body.Close()
		return r.StatusCode
	}

	// load → /v1/models → generate.
	if code := post(ts, "/admin/models/load", `{"name":"t","path":"`+path+`"}`); code != 200 {
		t.Fatalf("load status %d, want 200", code)
	}
	r, _ := http.Get(ts.URL + "/v1/models")
	var ml struct {
		Data []struct{ ID string } `json:"data"`
	}
	json.NewDecoder(r.Body).Decode(&ml)
	r.Body.Close()
	if len(ml.Data) != 1 || ml.Data[0].ID != "t" {
		t.Errorf("/v1/models = %v, want [t]", ml.Data)
	}
	if code := chat(); code != 200 {
		t.Errorf("generate after load: %d, want 200", code)
	}

	// unload-while-busy → 409 (hold the model's mutex to simulate a generation).
	srv.regMu.RLock()
	lm := srv.models["t"]
	srv.regMu.RUnlock()
	lm.mu.Lock()
	if code := post(ts, "/admin/models/unload", `{"name":"t"}`); code != http.StatusConflict {
		t.Errorf("unload while busy: %d, want 409", code)
	}
	lm.mu.Unlock()

	// unload → 404 on a subsequent request → reload.
	if code := post(ts, "/admin/models/unload", `{"name":"t"}`); code != 200 {
		t.Errorf("unload status %d, want 200", code)
	}
	if code := chat(); code != http.StatusNotFound {
		t.Errorf("generate after unload: %d, want 404", code)
	}
	if code := post(ts, "/admin/models/load", `{"name":"t","path":"`+path+`"}`); code != 200 {
		t.Errorf("reload status %d, want 200", code)
	}
	if code := chat(); code != 200 {
		t.Errorf("generate after reload: %d, want 200", code)
	}
}
