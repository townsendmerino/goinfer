package serveapp

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

// R28 through the callers: the real --help of both binaries (and `pull -h`) names the cache directory and XDG_CACHE_HOME. The usage text is
// assembled inside each main, so the only honest test is to run it; `go run` builds are cached after the first.
func TestHelp_namesTheModelCache(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("XDG_CACHE_HOME is what os.UserCacheDir reads on Linux only")
	}
	if testing.Short() {
		t.Skip("builds two binaries")
	}
	cache := t.TempDir()
	for _, c := range []struct {
		name string
		args []string
	}{
		{"goinfer-serve -h", []string{"run", "../../cmd/serve", "-h"}},
		{"goinfer-chat -h", []string{"run", "../../demo/chat", "-h"}},
		{"goinfer-serve pull -h", []string{"run", "../../cmd/serve", "pull", "-h"}},
		{"goinfer-chat pull -h", []string{"run", "../../demo/chat", "pull", "-h"}},
	} {
		cmd := exec.Command("go", c.args...)
		cmd.Env = append(os.Environ(), "XDG_CACHE_HOME="+cache)
		out, _ := cmd.CombinedOutput() // -h exits 0 or 2 depending on the front end; the text is what is asserted
		got := string(out)
		for _, want := range []string{cache + "/goinfer/models", "XDG_CACHE_HOME"} {
			if !strings.Contains(got, want) {
				t.Errorf("%s does not name %q:\n%.600s", c.name, want, got)
			}
		}
	}
}
