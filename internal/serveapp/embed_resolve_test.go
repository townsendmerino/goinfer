package serveapp

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEmbedModel_hfReferenceIsResolvedFirst is P7's serve half (task-checkpoint-fetch-2026-09.md): an hf: -embed-model goes
// through resolveEmbedModel before the encoder loads, and what loads is the path it resolved to, not the typed reference.
// A resolve failure fails startup naming the flag.
func TestEmbedModel_hfReferenceIsResolvedFirst(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"nomic_bert"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var asked []string
	orig := resolveEmbedModel
	t.Cleanup(func() { resolveEmbedModel = orig })
	resolveEmbedModel = func(_ context.Context, spec string) (string, error) {
		asked = append(asked, spec)
		if strings.Contains(spec, "gone") {
			return "", errors.New("repo not found")
		}
		return dir, nil
	}
	// The resolved directory holds only a minimal config.json, so the encoder fails PARSING it ("activation_function ...
	// unsupported"). That proves the load was handed what the resolver returned: the typed hf: string would have failed
	// with a missing file instead.
	_, err := newServer(config{embedPath: "hf:o/enc:safetensors", embedQuant: "f32"})
	if err == nil || !strings.Contains(err.Error(), "load embedding model") || !strings.Contains(err.Error(), "activation_function") || len(asked) != 1 || asked[0] != "hf:o/enc:safetensors" {
		t.Fatalf("newServer: %v (resolver asked %v); want the encoder to fail parsing the resolved dir's config", err, asked)
	}
	if _, err := newServer(config{embedPath: "hf:o/gone:safetensors"}); err == nil || !strings.Contains(err.Error(), "-embed-model hf:o/gone:safetensors") || !strings.Contains(err.Error(), "repo not found") {
		t.Fatalf("a failed resolve: %v, want it named with the flag", err)
	}
}

// TestEmbedEncoderLoads: the encoder checkpoint check accepts the NomicBert serve's encoder path loads, and refuses every
// other model_type, a generative one included, with a message that points a decoder-as-embedder at its GGUF form.
func TestEmbedEncoderLoads(t *testing.T) {
	if fam, err := embedEncoderLoads("nomic_bert"); err != nil || fam == "" {
		t.Errorf("nomic_bert: %q, %v; want accepted", fam, err)
	}
	for _, mt := range []string{"llama", "qwen3", "bert", ""} {
		if _, err := embedEncoderLoads(mt); err == nil || !strings.Contains(err.Error(), "GGUF") {
			t.Errorf("%q: %v, want refused with the GGUF pointer", mt, err)
		}
	}
}
