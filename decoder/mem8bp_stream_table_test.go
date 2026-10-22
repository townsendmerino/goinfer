package decoder

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestMEM8bp_streamedTablesAreTheWholeTableBytes is SQE-G1 (docs/tasks/task-memory-campaign-2026-10.md, MEM8(b′) gates): a
// streaming GGUF transcode that writes its embedding-shaped tables from their source rows (streamTablesOn) produces the same
// .giw bytes as one that builds them whole, for every tiny GGUF, quant, embed-int4 setting and target, and with the block
// size shrunk so a table spans many blocks (int4 at 4-row groups).
func TestMEM8bp_streamedTablesAreTheWholeTableBytes(t *testing.T) {
	e2b := filepath.Join(t.TempDir(), "gemma4-e2b.gguf")
	if err := os.WriteFile(e2b, tinyGemma4GGUF(g4Variant{name: "e2b", ple: true, sharedKV: 1, layers: 4, ffnGlobal: 256}), 0o644); err != nil {
		t.Fatal(err)
	}
	sources := []string{"../testdata/gemma1-tiny.gguf", "../testdata/gemma2-hd-tiny.gguf", "../testdata/glm-tiny.gguf", e2b}
	transcode := func(t *testing.T, path, quant string, embedInt4 bool, target GIWTarget, stream bool, block int) []byte {
		t.Helper()
		defer func(on bool, b int) { streamTablesOn, streamTableBlockBytes = on, b }(streamTablesOn, streamTableBlockBytes)
		streamTablesOn, streamTableBlockBytes = stream, block
		var buf bytes.Buffer
		if _, err := StreamTranscodeGGUF(context.Background(), path, &buf, quant, embedInt4, target, "mem8bp"); err != nil {
			t.Fatalf("StreamTranscodeGGUF(stream=%v): %v", stream, err)
		}
		return buf.Bytes()
	}
	cells := 0
	for _, src := range sources {
		if _, err := os.Stat(src); err != nil {
			t.Fatalf("no fixture %s: %v", src, err)
		}
		for _, quant := range []string{"int4", "int8int8", "int8", ""} {
			for _, embedInt4 := range []bool{false, true} {
				for _, target := range []GIWTarget{GIWTargetNone, GIWTargetMetal, GIWTargetCPUArm64, GIWTargetCUDA} {
					name := fmt.Sprintf("%s/%q/embed4=%v/%q", filepath.Base(src), quant, embedInt4, target)
					whole := transcode(t, src, quant, embedInt4, target, false, streamTableBlockBytes)
					for _, block := range []int{streamTableBlockBytes, 200} {
						got := transcode(t, src, quant, embedInt4, target, true, block)
						if !bytes.Equal(got, whole) {
							t.Errorf("%s, block %d B: the streamed tables' bundle (%d B) differs from the whole tables' (%d B)", name, block, len(got), len(whole))
						}
						cells++
					}
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[SQE-G1] %d cells: streamed-table bundles byte-identical to whole-table bundles\n", cells)
}
