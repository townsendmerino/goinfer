package modelload

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
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

// TestLoad_safetensorsDirGoesThroughSidecar is G-S18d's load half (docs/tasks/task-multimodal-support-2026-10.md, S18 on
// the Mac): where the sidecar default holds, a safetensors directory loads through a sidecar built beside it once; the
// second load reuses it without a rebuild; the tokenizer, chat template included, is still the directory's; and
// -direct-load keeps the directory. Red before S18: the directory loaded directly.
func TestLoad_safetensorsDirGoesThroughSidecar(t *testing.T) {
	if !prequant.DefaultToSidecar(false) {
		t.Skip("the sidecar default does not hold on this platform")
	}
	fix := filepath.Join("..", "..", "testdata", "tiny-qwen2-moe")
	ents, err := os.ReadDir(fix)
	if err != nil {
		t.Skipf("tiny fixture: %v", err)
	}
	src := filepath.Join(t.TempDir(), "tiny-qwen2.5-moe") // a dotted name: the sidecar must keep it whole
	if err := os.Mkdir(src, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.Type().IsRegular() {
			raw, err := os.ReadFile(filepath.Join(fix, e.Name()))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(src, e.Name()), raw, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	const tmpl = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
	if err := os.WriteFile(filepath.Join(src, "tokenizer_config.json"), []byte(`{"chat_template": "`+tmpl+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	opts := decoder.Options{Quant: "int4", EmbedInt4: true, Backend: "metal"}
	load := func(direct bool) *Result {
		t.Helper()
		res, err := Load(context.Background(), Request{Spec: src, Opts: opts, DirectLoad: direct})
		if err != nil {
			t.Fatalf("Load(direct=%v): %v", direct, err)
		}
		t.Cleanup(func() { res.Model.Close() })
		return res
	}
	first := load(false)
	want := src + ".int4.e4h.metal.giw"
	if first.LoadPath != want {
		t.Fatalf("LoadPath = %q, want the sidecar %q", first.LoadPath, want)
	}
	if first.Source != src {
		t.Errorf("Source = %q, want the directory %q (serve finds the vision tower and the name from it)", first.Source, src)
	}
	if got := first.Tokenizer.ChatTemplate(); got != tmpl {
		t.Errorf("chat template through the sidecar = %q, want the directory's %q", got, tmpl)
	}
	fi1, err := os.Stat(want)
	if err != nil {
		t.Fatal(err)
	}
	if second := load(false); second.LoadPath != want {
		t.Errorf("second LoadPath = %q, want %q", second.LoadPath, want)
	}
	if fi2, _ := os.Stat(want); !fi2.ModTime().Equal(fi1.ModTime()) {
		t.Error("the second load rebuilt a fresh sidecar")
	}
	if d := load(true); d.LoadPath != src {
		t.Errorf("-direct-load LoadPath = %q, want the directory", d.LoadPath)
	}
	for _, o := range []decoder.Options{{Quant: "int4", LoRA: "/an/adapter"}, {Quant: "q4k"}, {Quant: "int4", StreamWeights: true}} {
		if dirSidecarApplies(src, o) {
			t.Errorf("dirSidecarApplies(%+v) = true, want false", o)
		}
	}
}
