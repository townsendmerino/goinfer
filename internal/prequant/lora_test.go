package prequant

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const (
	tinyQwen35     = "../../decoder/testdata/qwen3_5-tiny"
	tinyQwen35LoRA = "../../decoder/testdata/qwen3_5-tiny-lora"
)

func promptHidden(t *testing.T, path string, o decoder.Options) []float32 {
	t.Helper()
	m, err := decoder.Load(path, o)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	defer m.Close()
	h, err := m.PromptHidden(context.Background(), []int{11, 48, 85, 122, 159, 196, 233})
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// TestTranscodeLoRA: a bundle built with -lora answers exactly as the base with the adapter merged at load does (f32,
// so the same arithmetic on the same weights), differs from the base, and carries a sidecar AdapterLoRA accepts for
// that adapter and refuses for another; a plain rebuild at the same path removes the sidecar; a GGUF input is refused.
func TestTranscodeLoRA(t *testing.T) {
	for _, p := range []string{tinyQwen35 + "/model.safetensors", tinyQwen35LoRA + "/adapter_model.safetensors"} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("no fixture %s", p)
		}
	}
	dir := t.TempDir()
	out := filepath.Join(dir, "merged.giw")
	if err := TranscodeLoRA(context.Background(), tinyQwen35, tinyQwen35LoRA, out, "", false, decoder.GIWTargetNone); err != nil {
		t.Fatal(err)
	}
	merged := promptHidden(t, tinyQwen35, decoder.Options{LoRA: tinyQwen35LoRA})
	bundled := promptHidden(t, out, decoder.Options{})
	base := promptHidden(t, tinyQwen35, decoder.Options{})
	same, moved := true, false
	for i := range merged {
		same = same && merged[i] == bundled[i]
		moved = moved || merged[i] != base[i]
	}
	if !same || !moved {
		t.Fatalf("bundle == merge-at-load: %v; the adapter moved the hidden state: %v", same, moved)
	}

	if a, err := AdapterLoRA(tinyQwen35LoRA, out); err != nil || a != "" {
		t.Fatalf("the bundle's own adapter: %q, %v (want \"\", nil)", a, err)
	}
	if a, err := AdapterLoRA(tinyQwen35LoRA, tinyQwen35); err != nil || a != tinyQwen35LoRA {
		t.Fatalf("a safetensors model: %q, %v (want the adapter itself)", a, err)
	}
	other := t.TempDir()
	b, _ := os.ReadFile(tinyQwen35LoRA + "/adapter_model.safetensors")
	if err := os.WriteFile(filepath.Join(other, "adapter_model.safetensors"), append(b, 0), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := AdapterLoRA(other, out); err == nil || !strings.Contains(err.Error(), "another adapter") {
		t.Fatalf("another adapter: %v", err)
	}

	if err := Transcode(context.Background(), tinyQwen35, out, "", false, decoder.GIWTargetNone); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out + ".lora.json"); !os.IsNotExist(err) {
		t.Fatalf("a plain rebuild left the sidecar: %v", err)
	}
	if _, err := AdapterLoRA(tinyQwen35LoRA, out); err == nil || !strings.Contains(err.Error(), "does not carry") {
		t.Fatalf("a bundle without the adapter: %v", err)
	}
	if err := TranscodeLoRA(context.Background(), filepath.Join(dir, "x.gguf"), tinyQwen35LoRA, out, "", false, decoder.GIWTargetNone); err == nil {
		t.Fatal("a GGUF input was accepted")
	}
}
