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

// adminSocketDirSafe refuses a directory in which someone else could replace the socket: one its group or everyone can
// write, unless it is sticky (/tmp), where only a file's owner can remove or rename it. Not checked on Windows, whose
// mode bits do not carry it.
func adminSocketDirSafe(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	fi, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("admin socket directory: %w", err)
	}
	if perm := fi.Mode().Perm(); perm&0o022 != 0 && fi.Mode()&os.ModeSticky == 0 {
		return fmt.Errorf("admin socket directory %s is writable by its group or by everyone (mode %04o): the socket has no key, its file permissions are the lock, and anyone who can write the directory can swap the file; put it in a directory only you can write (chmod 700, or the default path)", dir, perm)
	}
	return nil
}

// listenAdminSocket listens on path with the socket at mode 0600 from the moment it can be reached: it binds inside a
// new 0700 directory beside path, sets the mode, and renames the socket into place. Binding in place and then setting
// the mode leaves an instant at the process's umask, and a client that connects in it keeps its connection afterwards.
// Windows binds in place.
func listenAdminSocket(path string) (net.Listener, error) {
	if runtime.GOOS == "windows" {
		ln, err := net.Listen("unix", path)
		if err != nil {
			return nil, fmt.Errorf("listen: %w", err)
		}
		if err := os.Chmod(path, 0600); err != nil {
			ln.Close()
			return nil, fmt.Errorf("chmod 0600: %w", err)
		}
		return ln, nil
	}
	stage, err := os.MkdirTemp(filepath.Dir(path), ".as")
	if err != nil {
		return nil, fmt.Errorf("creating the socket's staging directory: %w", err)
	}
	defer os.Remove(stage)
	staged := filepath.Join(stage, "s")
	ln, err := net.Listen("unix", staged)
	if err != nil {
		return nil, fmt.Errorf("listen: %w", err)
	}
	// Close would unlink the name the listener was bound under; the socket no longer lives there.
	ln.(*net.UnixListener).SetUnlinkOnClose(false)
	if err = os.Chmod(staged, 0600); err == nil {
		err = os.Rename(staged, path)
	}
	if err != nil {
		ln.Close()
		os.Remove(staged)
		return nil, fmt.Errorf("placing the socket: %w", err)
	}
	return ln, nil
}

// startAdminSocket creates path fresh (unlinking whatever is there: a stale socket from a crashed process, since
// nothing live can still be listening once this process owns it), listens on it at mode 0600, and serves every
// /admin/* route (registerAdminRoutes, admin.go) on it with no auth wrapper. The returned close func shuts the
// listener down and unlinks the socket file; main.go calls it at shutdown.
//
// The file's permissions are the whole gate, so the directory must be one nobody else can swap the socket in
// (adminSocketDirSafe), and the socket is never reachable at a wider mode (listenAdminSocket).
func startAdminSocket(s *server, path string, textCap int64) (close func(), err error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("creating socket dir: %w", err)
	}
	if err := adminSocketDirSafe(dir); err != nil {
		return nil, err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("removing stale socket: %w", err)
	}
	ln, err := listenAdminSocket(path)
	if err != nil {
		return nil, err
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
