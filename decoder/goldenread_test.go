//go:build realckpt

package decoder

import (
	"compress/gzip"
	"io"
	"os"
	"strings"
)

// readGolden reads a test golden, gunzipping it when the name ends in .gz. Goldens over ~1 MB are
// committed compressed (CLAUDE.md — the convention for new goldens, and the backlog of 24 large ones
// was converted 2026-09-25), so a reader goes through this instead of os.ReadFile: the same call
// reads a small .json and a large .json.gz. A missing file is still an os.IsNotExist error, which is
// what the callers' "no golden — run the pin script" skips test for.
func readGolden(path string) ([]byte, error) {
	if !strings.HasSuffix(path, ".gz") {
		return os.ReadFile(path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	return io.ReadAll(gz)
}
