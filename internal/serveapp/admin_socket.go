package serveapp

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// K5 (docs/task-halt-2026-09.md): a second http.Server, on a Unix socket, serving ONLY /admin/*
// with no -api-key check — the socket's own file permissions (mode 0600, owned by whoever started
// serve) are the auth. Deliberately separate from the TCP listener rather than a second mux on the
// same *http.Server: a Unix socket has no host/port to bind wrong, and closing it independently at
// shutdown (main.go's closeAdminSock) can't accidentally also close the TCP listener or vice versa.

// defaultAdminSocketPath is the suggested -admin-socket path used in --help text and as the K5
// CLI's own default (`serve status`, etc.) when -admin-socket is not repeated on that command
// line. NOT used as the flag's own default value — -admin-socket is off (empty) unless a caller
// explicitly sets it; this only gives the two sides (the running server and the CLI that talks to
// it) a path to agree on without either one hardcoding the other's invocation.
func defaultAdminSocketPath() string {
	if runtime.GOOS == "darwin" {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, "Library", "Application Support", "goinfer", "admin.sock")
		}
	}
	return "/run/goinfer/admin.sock"
}

// startAdminSocket creates path fresh (unlinking anything already there — a stale socket from a
// prior crashed process, not a live one; nothing else would still be listening on it once this
// process owns it), listens on it at mode 0600, and serves every /admin/* route (registerAdminRoutes,
// admin.go) on it with no auth wrapper. The returned close func shuts the listener down and unlinks
// the socket file; main.go calls it unconditionally at shutdown (a no-op when -admin-socket was
// never set, since this function is then never called at all).
func startAdminSocket(s *server, path string, textCap int64) (close func(), err error) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("removing stale socket: %w", err)
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return nil, fmt.Errorf("creating socket dir: %w", err)
		}
	}
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, fmt.Errorf("listen: %w", err)
	}
	if err := os.Chmod(path, 0600); err != nil {
		ln.Close()
		return nil, fmt.Errorf("chmod 0600: %w", err)
	}
	mux := http.NewServeMux()
	// identity wrap: no auth(), no requireAdmin — the socket's file permissions are the whole
	// gate. See registerAdminRoutes' own doc comment (admin.go).
	registerAdminRoutes(mux, s, textCap, func(h http.HandlerFunc) http.HandlerFunc { return h })
	sockSrv := &http.Server{Handler: mux}
	go func() {
		if err := sockSrv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "admin socket %s: %v\n", path, err)
		}
	}()
	return func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = sockSrv.Shutdown(ctx)
		_ = os.Remove(path)
	}, nil
}
