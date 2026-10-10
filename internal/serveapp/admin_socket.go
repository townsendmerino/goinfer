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

// A second http.Server, on a Unix socket, serving only /admin/* with no -api-key check: the socket's file permissions
// (mode 0600, owned by whoever started serve) are the auth. It is separate from the TCP listener, not a second mux on
// the same *http.Server, so closing it at shutdown (main.go's closeAdminSock) cannot also close the TCP listener or
// vice versa.

// defaultAdminSocketPath is the suggested -admin-socket path for --help text and the admin CLI's default when
// -admin-socket is not repeated on its command line. It is not the flag's own default: -admin-socket is off (empty)
// unless set; this only gives the running server and the CLI a path to agree on.
func defaultAdminSocketPath() string {
	if runtime.GOOS == "darwin" {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, "Library", "Application Support", "goinfer", "admin.sock")
		}
	}
	return "/run/goinfer/admin.sock"
}

// startAdminSocket creates path fresh (unlinking whatever is there: a stale socket from a crashed process, since
// nothing live can still be listening once this process owns it), listens on it at mode 0600, and serves every
// /admin/* route (registerAdminRoutes, admin.go) on it with no auth wrapper. The returned close func shuts the
// listener down and unlinks the socket file; main.go calls it at shutdown.
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
