package decoder

import (
	"context"
	"io"
	"os"
	"sync/atomic"
	"testing"
)

// TestStreamTranscode_buildsNoDecodeLayout: a transcode offers no tensor to the in-RAM row4 repack, from a safetensors
// directory or from a GGUF, with the logit tables at int4 too (the tables never skip row4 in a load that decodes). The
// resident load of the same checkpoint does offer its tensors, so the count is not zero by construction.
func TestStreamTranscode_buildsNoDecodeLayout(t *testing.T) {
	var calls atomic.Int64
	row4RepackCallsForTest = &calls
	defer func() { row4RepackCallsForTest = nil }()
	for _, c := range []struct {
		src    string
		stream func(src string) error
	}{
		{"../testdata/qwen3moe-tiny", func(src string) error {
			_, err := StreamTranscodeDir(context.Background(), src, io.Discard, "int4", true, GIWTargetCPUArm64, "t")
			return err
		}},
		{"../testdata/glm-tiny.gguf", func(src string) error {
			_, err := StreamTranscodeGGUF(context.Background(), src, io.Discard, "int4", true, GIWTargetCPUArm64, "t")
			return err
		}},
	} {
		if _, err := os.Stat(c.src); err != nil {
			t.Fatalf("%s is a committed fixture: %v", c.src, err)
		}
		calls.Store(0)
		m, err := Load(c.src, Options{Quant: "int4", EmbedInt4: true})
		if err != nil {
			t.Fatalf("%s: resident load: %v", c.src, err)
		}
		m.Close()
		if calls.Load() == 0 {
			t.Fatalf("%s: the resident load offered no tensor to the row4 repack: the counter sees nothing", c.src)
		}
		calls.Store(0)
		if err := c.stream(c.src); err != nil {
			t.Fatalf("%s: transcode: %v", c.src, err)
		}
		if n := calls.Load(); n != 0 {
			t.Errorf("%s: the transcode offered %d tensor(s) to the in-RAM row4 repack, want 0", c.src, n)
		}
	}
}

func TestRepackForDecode(t *testing.T) {
	if !repackForDecode(nil) {
		t.Error("a build with no sink is a load that decodes: it keeps its decode layout")
	}
	if repackForDecode(&giwWriter{}) {
		t.Error("a build for a sink decodes nothing: it keeps canonical bytes alone")
	}
}
