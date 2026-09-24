//go:build darwin

package decoder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// Load must map a .giw MAP_SHARED on darwin (giwmap_darwin.go): over a private mapping, Metal's no-copy
// buffers turn every GPU-read page into a wired anonymous copy and make fork() copy the whole mapping.
// Tested through Load, and by what the kernel reports for the region — vmmap's share mode is COW for a
// private file mapping and ALI for a shared one — not by trusting which helper Load names.
func TestLoad_mapsGIWSharedOnDarwin(t *testing.T) {
	vm, err := exec.LookPath("vmmap")
	if err != nil {
		t.Skip("vmmap not available")
	}
	raw, _, _, _ := tinyNormRopeGGUF("llama")
	w := loadTinyGGUFWeights(t, raw, "llama")
	blob, err := SerializeWeights(w, "giwmap")
	if err != nil {
		t.Fatal(err)
	}
	dir, err := filepath.EvalSymlinks(t.TempDir()) // vmmap prints the resolved path (/private/var/...)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "giwmap.giw")
	if err := os.WriteFile(path, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	_ = m.mmap[0] // make sure the region is present

	out, err := exec.Command(vm, "-interleaved", fmt.Sprint(os.Getpid())).Output()
	if err != nil {
		t.Skipf("vmmap: %v", err)
	}
	var region string
	for _, l := range strings.Split(string(out), "\n") {
		if strings.Contains(l, path) {
			region = strings.Join(strings.Fields(l), " ")
			break
		}
	}
	if region == "" {
		t.Fatalf("vmmap lists no region for %s", path)
	}
	t.Logf("%s", region)
	if strings.Contains(region, "SM=COW") || !strings.Contains(region, "SM=ALI") {
		t.Errorf("the .giw mapping is not shared (want SM=ALI): %s — a private mapping makes Metal's no-copy "+
			"pages anonymous copies and a later fork() copy the whole file", region)
	}
}
