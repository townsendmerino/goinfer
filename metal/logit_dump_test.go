//go:build darwin

package metal

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDumpLogitsForBisect writes the resident GPU logits for a fixed 4-token sequence to
// $GOINFER_DUMP_LOGITS as raw little-endian float32. It exists to back the 9c Step-1
// "byte-identical" claim with an actual BITWISE diff (parent commit vs HEAD), not just a
// cosine-vs-CPU threshold: run it on the parent, run it on HEAD, `cmp` the two files. It uses
// only BuildResident + ForwardEmb + EmbedResidentForTest — API identical across the refactor —
// so the same source compiles and runs on both commits. Skips unless the env var is set.
func TestDumpLogitsForBisect(t *testing.T) {
	out := os.Getenv("GOINFER_DUMP_LOGITS")
	if out == "" {
		t.Skip("set GOINFER_DUMP_LOGITS=/path to dump resident logits for a bitwise bisect")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()

	ids := []int{7, 11, 42, 100}
	var all []float32
	for i, id := range ids {
		all = append(all, r.ForwardEmb(m.EmbedResidentForTest(id), i)...)
	}
	buf := make([]byte, 4*len(all))
	for i, v := range all {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	if err := os.WriteFile(out, buf, 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("dumped %d logits (%d bytes) to %s", len(all), len(buf), out)
}
