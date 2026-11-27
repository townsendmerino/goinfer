package adminsock

import (
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestDirSafeAndStart(t *testing.T) {
	if DefaultSocketPath() == "" {
		t.Error("DefaultSocketPath returned empty")
	}
	if !IsCLICmd("halt") || IsCLICmd("serve") {
		t.Error("IsCLICmd mismatch")
	}
	// Unix socket paths on darwin/linux have a ~104-byte sun_path limit, so use /tmp rather than t.TempDir().
	base, err := os.MkdirTemp("/tmp", "gi-as-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(base) })
	if err := os.Chmod(base, 0700); err != nil {
		t.Fatal(err)
	}
	if err := DirSafe(base); err != nil {
		t.Fatalf("0700 dir rejected: %v", err)
	}
	if runtime.GOOS != "windows" {
		wide := filepath.Join(base, "wide")
		if err := os.Mkdir(wide, 0777); err != nil {
			t.Fatal(err)
		}
		_ = os.Chmod(wide, 0777)
		if err := DirSafe(wide); err == nil {
			t.Error("0777 non-sticky dir must be rejected")
		}
	}
	sock := filepath.Join(base, "a.sock")
	mux := http.NewServeMux()
	mux.HandleFunc("GET /admin/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"halted":false}`))
	})
	closeFn, err := Start(mux, sock)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	fi, err := os.Stat(sock)
	if err != nil {
		t.Fatalf("stat socket: %v", err)
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm() != 0600 {
		t.Errorf("socket mode = %04o, want 0600", fi.Mode().Perm())
	}
	resp, err := NewUnixHTTPClient(sock, 5*time.Second).Get("http://admin-socket/admin/status")
	if err != nil {
		t.Fatalf("GET over socket: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	closeFn()
	if _, err := os.Stat(sock); !os.IsNotExist(err) {
		t.Errorf("socket still exists after close: %v", err)
	}
}
