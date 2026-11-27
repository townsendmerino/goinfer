//go:build linux

package decoder

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

// mappedRSSOf is this process's resident bytes in the mappings of path (/proc/self/smaps Rss), the pages a load holds.
func mappedRSSOf(t *testing.T, path string) int64 {
	t.Helper()
	f, err := os.Open("/proc/self/smaps")
	if err != nil {
		t.Skipf("no /proc/self/smaps: %v", err)
	}
	defer f.Close()
	var total int64
	in := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if fields := strings.Fields(line); len(fields) >= 5 && strings.Contains(fields[0], "-") && !strings.HasSuffix(fields[0], ":") {
			in = len(fields) >= 6 && fields[len(fields)-1] == path
			continue
		}
		if in && strings.HasPrefix(line, "Rss:") {
			var kb int64
			if _, err := fmt.Sscanf(strings.TrimPrefix(line, "Rss:"), "%d", &kb); err != nil {
				t.Fatalf("smaps %q: %v", line, err)
			}
			total += kb << 10
		}
	}
	return total
}

// TestGGUFDirectLoad_releasesTheSource holds a direct .gguf load to releasing its source as it converts: just before the
// mapping closes, the process holds under a quarter of the file resident (Linux; elsewhere the advice does not reclaim).
// Without the release the whole file stays resident beside the built weights, which on a 27.8 GB checkpoint put the peak
// past what the box holds.
func TestGGUFDirectLoad_releasesTheSource(t *testing.T) {
	path, err := filepath.Abs("../testdata/gemma2-hd-tiny.gguf")
	if err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	var held int64 = -1
	testgate.SetForTest(t, &ggufBuiltHookForTest, func(p string) {
		if p == path || filepath.Clean(p) == path {
			held = mappedRSSOf(t, path)
		}
	})
	m, err := Load(path, Options{Quant: "int8", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if held < 0 {
		t.Fatal("the build hook never ran: the load did not take the direct .gguf path")
	}
	t.Logf("source %d bytes; %d resident at the end of the build", st.Size(), held)
	if held*4 > st.Size() {
		t.Fatalf("%d of the source's %d bytes still resident at the end of the build: the load holds the source beside its weights", held, st.Size())
	}
}
