package embedders

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

func TestTruncateForContextAndClosed(t *testing.T) {
	ids := TruncateForContext(make([]int, 64), 16, -1)
	if len(ids) != 16 {
		t.Fatalf("truncated to %d, want 16", len(ids))
	}
	ids2 := append(TruncateForContext(make([]int, 64), 16, 7), 7)
	if len(ids2) != 16 || ids2[len(ids2)-1] != 7 {
		t.Fatalf("with appended token got len=%d last=%d", len(ids2), ids2[len(ids2)-1])
	}
	var de DecoderEmbedder
	if _, err := de.Encode("hello", false); !errors.Is(err, decoder.ErrClosed) {
		t.Fatalf("closed Encode err = %v, want decoder.ErrClosed", err)
	}
}

type accelStub struct {
	err  error
	used []string
}

func (s *accelStub) UseAccelerator(name string) (embeddinggemma2.Accelerator, error) {
	s.used = append(s.used, name)
	return nil, s.err
}

func TestIsEmbeddingGemma2AndChooseDevice(t *testing.T) {
	dir := t.TempDir()
	if IsEmbeddingGemma2(dir) {
		t.Error("empty dir recognized")
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"embedding_gemma2"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if !IsEmbeddingGemma2(dir) {
		t.Error("embedding_gemma2 dir not recognized")
	}
	ok := &accelStub{}
	if where, err := ChooseEG2Device(ok, "metal", false); err != nil || where != "Metal" || len(ok.used) != 1 {
		t.Errorf("metal: %q %v %v", where, err, ok.used)
	}
	bad := &accelStub{err: errors.New("no device")}
	if where, err := ChooseEG2Device(bad, "metal", false); err != nil || !strings.Contains(where, "Metal declined: no device") {
		t.Errorf("metal declined: %q %v", where, err)
	}
	if _, err := ChooseEG2Device(bad, "metal", true); err == nil {
		t.Error("metal declined under require must error")
	}
	if _, err := ChooseEG2Device(&accelStub{}, "cuda", true); err == nil {
		t.Error("cuda under require must error")
	}
}
