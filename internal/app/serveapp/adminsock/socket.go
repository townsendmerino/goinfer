package adminsock

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

// DefaultSocketPath returns the suggested -admin-socket path for help text and
// the admin CLI's default when -admin-socket is omitted on the subcommand line.
func DefaultSocketPath() string {
	if runtime.GOOS == "darwin" {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, "Library", "Application Support", "goinfer", "admin.sock")
		}
	}
	return "/run/goinfer/admin.sock"
}

// DirSafe refuses a directory writable by group or others unless sticky (/tmp),
// preventing another local user from swapping the unauthenticated socket file.
func DirSafe(dir string) error {
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

// Listen binds a Unix socket at path with mode 0600 from the instant it is
// reachable by staging inside a 0700 directory beside path and renaming into place.
func Listen(path string) (net.Listener, error) {
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

// Start creates path fresh at mode 0600 after checking DirSafe and serves handler
// on a dedicated http.Server. Calling the returned close function shuts down the
// server and unlinks path.
func Start(handler http.Handler, path string) (close func(), err error) {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("creating socket dir: %w", err)
	}
	if err := DirSafe(dir); err != nil {
		return nil, err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("removing stale socket: %w", err)
	}
	ln, err := Listen(path)
	if err != nil {
		return nil, err
	}
	sockSrv := &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
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
