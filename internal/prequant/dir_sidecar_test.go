package prequant

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestDirSidecar_matchesDirectLoad is G-S18d's identity half (docs/tasks/task-multimodal-support-2026-10.md, S18 on the
// Mac): for every tiny safetensors fixture in testdata that decoder.Load reads as a generating model, the directory
// transcoded to a sidecar (as modelload now does by default on darwin and linux) must generate the same greedy stream as
// the directory loaded directly, at serve's defaults (int4, embed-int4, the Metal target). A config field the bundle
// drops is the silent kind of defect (LFM2's NormEps and AttnScale were both zero and legal-looking), and greedy
// decoding over a tiny random model is sensitive to any change in the logits. A fixture whose transcode fails is
// reported, not failed: modelload falls back to the direct load for it.
func TestDirSidecar_matchesDirectLoad(t *testing.T) {
	ents, err := os.ReadDir("../../testdata")
	if err != nil {
		t.Fatal(err)
	}
	const quant = "int4"
	target := decoder.GIWTargetForBackend("metal")
	prompts := [][]int{{1, 2, 3, 4, 5}, {3, 1, 4, 2}} // ids under every fixture's vocab (lfm2-tiny and smollm3-tiny have small ones)
	var compared, fellBack []string
	for _, e := range ents {
		dir := filepath.Join("../../testdata", e.Name())
		if !e.IsDir() || !hasSafetensors(dir) {
			continue
		}
		t.Run(e.Name(), func(t *testing.T) {
			direct, err := decoder.Load(dir, decoder.Options{Quant: quant, EmbedInt4: true})
			if err != nil {
				t.Skipf("not a model decoder.Load reads directly: %v", err)
			}
			defer direct.Close()
			ref, why := greedyOrSkip(direct, prompts)
			if why != "" {
				t.Skipf("the direct load does not generate (an encoder or tower): %s", why)
			}
			out := filepath.Join(t.TempDir(), "s.giw")
			if err := Transcode(context.Background(), dir, out, quant, true, target); err != nil {
				fellBack = append(fellBack, e.Name())
				t.Logf("transcode refused (modelload falls back to the direct load): %v", err)
				return
			}
			side, err := decoder.Load(out, decoder.Options{})
			if err != nil {
				t.Fatalf("the sidecar does not load: %v", err)
			}
			defer side.Close()
			got, why := greedyOrSkip(side, prompts)
			if why != "" {
				t.Fatalf("the direct load generates and the sidecar does not: %s", why)
			}
			for i := range ref {
				if !slices.Equal(ref[i], got[i]) {
					t.Errorf("prompt %d: direct %v, sidecar %v", i, ref[i], got[i])
				}
			}
			compared = append(compared, e.Name())
		})
	}
	t.Logf("compared %d fixtures: %s", len(compared), strings.Join(compared, " "))
	t.Logf("transcode refused, direct load kept: %v", fellBack)
	if len(compared) < 10 {
		t.Errorf("only %d fixtures compared; the sweep is not covering the families it claims", len(compared))
	}
}

// TestDirSidecar_cachePathAndFreshness is G-S18d's cache half: a directory's sidecar is named after the whole directory
// (a dotted name keeps its dots), it is fresh once written, and a file inside the directory rewritten in place, which
// leaves the directory's own mtime alone, makes it stale.
func TestDirSidecar_cachePathAndFreshness(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "qwen2.5-0.5b-instruct")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	w := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(w, make([]byte, 1<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	cache := streamCachePath(dir, "int4", true, decoder.GIWTargetMetal)
	if want := filepath.Join(root, "qwen2.5-0.5b-instruct.int4.e4h.metal.giw"); cache != want {
		t.Fatalf("streamCachePath(dir) = %q, want %q", cache, want)
	}
	time.Sleep(20 * time.Millisecond)
	if err := os.WriteFile(cache, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !cacheNewer(cache, dir) {
		t.Fatal("a sidecar written after every file in the directory must be newer")
	}
	dirInfo, _ := os.Stat(dir)
	time.Sleep(20 * time.Millisecond)
	if err := os.WriteFile(w, make([]byte, 1<<20+1), 0o644); err != nil { // in place: the directory's mtime does not move
		t.Fatal(err)
	}
	if after, _ := os.Stat(dir); !after.ModTime().Equal(dirInfo.ModTime()) {
		t.Logf("note: this filesystem moved the directory's mtime on an in-place write")
	}
	if cacheNewer(cache, dir) {
		t.Error("a safetensors file rewritten after the sidecar must make it stale")
	}
	if n, ok := projectedDirSidecarBytes(dir, "int4"); !ok || n <= 0 {
		t.Errorf("projectedDirSidecarBytes = %d, %v; want a positive projection", n, ok)
	}
}

func hasSafetensors(dir string) bool {
	m, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(m) > 0
}

// greedyOrSkip runs each prompt for 12 greedy tokens; why is non-empty when the model does not generate, and says why.
func greedyOrSkip(m *decoder.Model, prompts [][]int) (out [][]int, why string) {
	defer func() {
		if p := recover(); p != nil {
			why = fmt.Sprint("panic: ", p)
		}
	}()
	for _, p := range prompts {
		ch, gen := m.Generate(context.Background(), p, 12, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := gen.Err(); err != nil {
			return nil, err.Error()
		}
		out = append(out, ids) // an empty stream (EOS first) compares too: lfm2-tiny and smollm3-tiny stop at once on one prompt
	}
	if slices.IndexFunc(out, func(ids []int) bool { return len(ids) > 0 }) < 0 {
		return nil, "no prompt produced a token"
	}
	return out, ""
}

// TestDirSidecar_keepsMRopeSection: a Qwen-VL's m-RoPE section survives the sidecar. Real checkpoints carry it in
// rope_scaling, which the adapters read and clear, so the sidecar has to carry it itself: a .giw that drops it
// loads a model that runs plain RoPE on image positions. The tiny fixtures carry it in rope_parameters, a raw
// field that survives, which is why TestDirSidecar_matchesDirectLoad (text-only, where the three axes coincide
// anyway) cannot see the loss: each fixture here is copied with the section moved into rope_scaling, as the
// released checkpoints have it. A config with no section at all is refused, which is what makes a stale sidecar
// fail its self-check and rebuild.
func TestDirSidecar_keepsMRopeSection(t *testing.T) {
	for _, fx := range []string{"qwen25vl-tiny", "qwen3vl-tiny"} {
		t.Run(fx, func(t *testing.T) {
			src := filepath.Join("../../testdata", fx)
			if _, err := os.Stat(src); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			copyDir := func(cfgEdit func(map[string]any)) string {
				dst := filepath.Join(t.TempDir(), fx)
				if err := os.Mkdir(dst, 0o755); err != nil {
					t.Fatal(err)
				}
				ents, _ := os.ReadDir(src)
				for _, e := range ents {
					if !e.Type().IsRegular() {
						continue
					}
					raw, err := os.ReadFile(filepath.Join(src, e.Name()))
					if err != nil {
						t.Fatal(err)
					}
					if e.Name() == "config.json" {
						var c map[string]any
						if err := json.Unmarshal(raw, &c); err != nil {
							t.Fatal(err)
						}
						cfgEdit(c)
						raw, _ = json.Marshal(c)
					}
					if err := os.WriteFile(filepath.Join(dst, e.Name()), raw, 0o644); err != nil {
						t.Fatal(err)
					}
				}
				return dst
			}
			// textCfg is where the decoder reads the text config: the nested text_config when there is one.
			textCfg := func(c map[string]any) map[string]any {
				if tc, ok := c["text_config"].(map[string]any); ok {
					return tc
				}
				return c
			}
			var section []any
			released := copyDir(func(c map[string]any) {
				tc := textCfg(c)
				rp, _ := tc["rope_parameters"].(map[string]any)
				if rp == nil {
					rp, _ = c["rope_parameters"].(map[string]any)
				}
				section, _ = rp["mrope_section"].([]any)
				delete(rp, "mrope_section")
				tc["rope_scaling"] = map[string]any{"type": "mrope", "rope_type": "default", "mrope_section": section}
			})
			if len(section) != 3 {
				t.Fatalf("the fixture has no mrope_section in rope_parameters to move")
			}
			half := 0
			for _, v := range section {
				half += int(v.(float64))
			}
			direct, err := decoder.Load(released, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatal(err)
			}
			want := direct.MRopeAxisResident(half)
			direct.Close()
			if len(want) == 0 {
				t.Fatal("the direct load has no m-RoPE axis table: the rewritten config is not the released layout")
			}
			out := filepath.Join(t.TempDir(), "s.giw")
			if err := Transcode(context.Background(), released, out, "int4", false, decoder.GIWTargetForBackend("metal")); err != nil {
				t.Fatal(err)
			}
			side, err := decoder.Load(out, decoder.Options{})
			if err != nil {
				t.Fatal(err)
			}
			got := side.MRopeAxisResident(half)
			side.Close()
			if !slices.Equal(got, want) {
				t.Errorf("m-RoPE axis table through the sidecar %v, direct %v", got, want)
			}
			none := copyDir(func(c map[string]any) {
				tc := textCfg(c)
				for _, m := range []map[string]any{tc, c} {
					if rp, ok := m["rope_parameters"].(map[string]any); ok {
						delete(rp, "mrope_section")
					}
				}
			})
			if _, err := decoder.Load(none, decoder.Options{Quant: "int4"}); err == nil || !strings.Contains(err.Error(), "no m-RoPE section") {
				t.Errorf("a config with no m-RoPE section loaded (err %v); it must be refused", err)
			}
		})
	}
}
