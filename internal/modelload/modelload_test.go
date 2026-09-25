package modelload

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// N-25, now in the one place all three apps read a .giw's tok half: when the bytes are neither a GGUF
// nor a tokenizer.json, the error names BOTH attempts. serve's copy used to return only the JSON one,
// so a corrupt GGUF-sourced bundle reported "invalid JSON" and pointed at the wrong half of the file.
func TestTokenizerFromTok_reportsBothErrors(t *testing.T) {
	_, err := TokenizerFromTok([]byte("neither a GGUF nor JSON"))
	if err == nil {
		t.Fatal("garbage tok half parsed")
	}
	msg := err.Error()
	if !strings.Contains(msg, "not a GGUF") || !strings.Contains(msg, "not tokenizer.json") {
		t.Errorf("error must name both the GGUF and the JSON attempt, got: %v", err)
	}
}

// A plain path is returned untouched — the property that lets fit, serve and chat all resolve every
// --model value without changing what an existing path means.
func TestResolve_plainPathUntouched(t *testing.T) {
	for _, p := range []string{"/models/x.gguf", "rel/dir", "x.giw", "hfdir/hf:not-a-prefix"} {
		got, err := Resolve(context.Background(), p)
		if err != nil || got != p {
			t.Errorf("Resolve(%q) = %q, %v; want the path unchanged", p, got, err)
		}
	}
}

// --lora with a .gguf (or .giw) is refused, and refused BEFORE a sidecar is built. A merged LoRA needs a
// safetensors base; the direct .gguf load always said so, but the sidecar default turned the .gguf into a
// .giw first, and the .giw path ignored LoRA — the adapter was dropped without a word.
func TestLoad_loraOnAQuantizedFileIsRefusedBeforeTranscoding(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "glm-tiny.gguf"))
	if err != nil {
		t.Skipf("tiny fixture: %v", err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(src, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, spec := range []string{src, filepath.Join(dir, "tiny.int4.canonical.giw")} {
		_, err := Load(context.Background(), Request{Spec: spec, Opts: decoder.Options{Quant: "int4", LoRA: "/some/adapter"}})
		if err == nil || !strings.Contains(err.Error(), "safetensors base") {
			t.Errorf("Load(%s, LoRA): err = %v, want a refusal naming the safetensors base", filepath.Base(spec), err)
		}
	}
	ents, _ := os.ReadDir(dir)
	for _, e := range ents {
		if strings.HasSuffix(e.Name(), ".giw") {
			t.Errorf("a sidecar %s was built for a load that was going to be refused", e.Name())
		}
	}
}
