package serveapp

import (
	"context"
	"io"
	"net"
	"net/http"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestServe_generationRoutesExistWithoutAStartupModel: a server started with only --web (or
// --allow-admin / --admin-socket) loads its model later, but the mux is built once, and the
// generation routes used to be registered only when a model existed at startup. That server had no
// /v1/chat/completions and no /v1/jobs for its whole life — the web UI's own chat got the mux's bare
// "404 page not found" even after loading a model through the UI. Driven through the real binary,
// since the mux is built in Main: with no model loaded, a chat request must reach the handler and get
// its JSON "model not found", not the mux's plain-text 404.
func TestServe_generationRoutesExistWithoutAStartupModel(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and runs a binary")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "goinfer-serve")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/cmd/serve")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build cmd/serve here: %v\n%s", err, out)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "--web", "--addr", addr)
	cmd.Env = append(cmd.Environ(), "GOINFER_SWAP_GUARD=off")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { cancel(); _ = cmd.Wait() }()
	base := "http://" + addr
	for i := 0; ; i++ {
		if resp, err := http.Get(base + "/v1/models"); err == nil {
			resp.Body.Close()
			break
		}
		if i > 200 {
			t.Fatal("server never came up")
		}
		time.Sleep(50 * time.Millisecond)
	}
	for _, route := range []string{"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/jobs"} {
		resp, err := http.Post(base+route, "application/json",
			strings.NewReader(`{"model":"nothing-loaded","messages":[{"role":"user","content":"hi"}],"prompt":"hi","input":"hi","max_tokens":4}`))
		if err != nil {
			t.Fatalf("POST %s: %v", route, err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if strings.Contains(string(body), "404 page not found") {
			t.Errorf("POST %s: the route is not registered (mux 404) — a model loaded later could never be reached", route)
			continue
		}
		if resp.StatusCode/100 != 4 || !strings.Contains(string(body), "not found") {
			t.Errorf("POST %s: got %s %q, want the handler's JSON model-not-found", route, strconv.Itoa(resp.StatusCode), body)
		}
	}
}
