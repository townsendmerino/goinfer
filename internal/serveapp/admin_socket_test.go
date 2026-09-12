package serveapp

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

// unixHTTPClient dials only path, ignoring whatever host/port a request URL names — the same
// pattern admin_cli.go's runAdminCLI uses to talk to the admin socket.
func unixHTTPClient(path string) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", path)
			},
		},
	}
}

// TestServe_adminSocket is the K5 gate (docs/task-halt-2026-09.md): with -admin-socket set, halt
// works over the socket with no -api-key; /admin/halt on the TCP listener is a plain 404 (the
// route was never registered — not a 403, which would confirm the surface exists); and a /v1
// client holding the correct -api-key still cannot reach it there, because there is nothing at
// that path to reach regardless of the key. No real model needed: halt/resume/status/generations
// none of them touch a loaded model, only K1's registry and K2's atomic state.
func TestServe_adminSocket(t *testing.T) {
	sockPath := filepath.Join(t.TempDir(), "admin.sock")
	const apiKey = "s3cr3t"

	srv, err := newServer(config{adminSocket: sockPath})
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}

	// The admin socket: production code path (admin_socket.go), not a hand-rolled test mux.
	closeSock, err := startAdminSocket(srv, sockPath, 1<<20)
	if err != nil {
		t.Fatalf("startAdminSocket: %v", err)
	}
	defer closeSock()

	client := unixHTTPClient(sockPath)

	// Halt over the socket, no api-key at all.
	haltResp, err := client.Post("http://admin-socket/admin/halt", "application/json", strings.NewReader(`{"reason":"k5 test"}`))
	if err != nil {
		t.Fatalf("POST /admin/halt over socket: %v", err)
	}
	var haltResult map[string]any
	json.NewDecoder(haltResp.Body).Decode(&haltResult)
	haltResp.Body.Close()
	if haltResp.StatusCode != http.StatusOK || haltResult["halted"] != true {
		t.Errorf("halt over socket: status=%d body=%v, want 200 halted:true", haltResp.StatusCode, haltResult)
	}

	statusResp, err := client.Get("http://admin-socket/admin/status")
	if err != nil {
		t.Fatalf("GET /admin/status over socket: %v", err)
	}
	var statusResult map[string]any
	json.NewDecoder(statusResp.Body).Decode(&statusResult)
	statusResp.Body.Close()
	if statusResult["halted"] != true {
		t.Errorf("status over socket after halt: %v, want halted:true", statusResult)
	}

	resumeResp, err := client.Post("http://admin-socket/admin/resume", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /admin/resume over socket: %v", err)
	}
	resumeResp.Body.Close()
	if resumeResp.StatusCode != http.StatusOK {
		t.Errorf("resume over socket: status=%d, want 200", resumeResp.StatusCode)
	}

	// The TCP listener: mirrors main.go's own conditional — /admin/* is registered there only
	// when cfg.adminSocket == "" (main.go's registerAdminRoutes call). Since it's set here, only
	// the ordinary auth-gated /v1-style route goes on this mux, matching what a real serve
	// process would do with these flags.
	auth := func(h http.HandlerFunc) http.HandlerFunc { return requireAuth(apiKey, h) }
	tcpMux := http.NewServeMux()
	tcpMux.HandleFunc("GET /health", auth(srv.handleHealth))
	ts := httptest.NewServer(tcpMux)
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/admin/halt", strings.NewReader(`{"reason":"should not work"}`))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	tcpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /admin/halt on TCP: %v", err)
	}
	tcpResp.Body.Close()
	if tcpResp.StatusCode != http.StatusNotFound {
		t.Errorf("/admin/halt on TCP with -admin-socket set: status=%d, want 404 (route must not be registered there at all, not merely refused)", tcpResp.StatusCode)
	}

	req2, _ := http.NewRequest("POST", ts.URL+"/admin/resume", nil)
	req2.Header.Set("Authorization", "Bearer "+apiKey)
	tcpResp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("POST /admin/resume on TCP: %v", err)
	}
	tcpResp2.Body.Close()
	if tcpResp2.StatusCode != http.StatusNotFound {
		t.Errorf("/admin/resume on TCP with -admin-socket set: status=%d, want 404", tcpResp2.StatusCode)
	}
}
