package serveapp

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeAttach struct {
	int8       bool
	failAttach bool
	closed     *int
}

func (f *fakeAttach) EnableResident() error {
	if f.failAttach {
		return errors.New("cuda: device allocation failed: CUDA_ERROR_OUT_OF_MEMORY")
	}
	return nil
}
func (f *fakeAttach) Close() { *f.closed++ }

// TestAttachGemma3Tower pins the float32-default fallback: a float32 tower that was the DEFAULT and does not attach is released and replaced by the int8 device tower, with a line saying why; an
// explicit choice never falls back; the error paths are the old ones.
func TestAttachGemma3Tower(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		startInt8, fallback, require   bool
		f32Fails, int8Fails            bool
		wantInt8, wantResident, wantEr bool
		wantLoads                      []bool
		wantClosed                     int
		wantNote                       bool
	}{
		{"default float32 attaches", false, true, false, false, false, false, true, false, []bool{false}, 0, false},
		{"default float32 does not fit: falls back to the int8 device tower", false, true, false, true, false, true, true, false, []bool{false, true}, 1, true},
		{"default float32 does not fit and neither does int8: CPU tower, no error without -require-backend", false, true, false, true, true, true, false, false, []bool{false, true}, 1, true},
		{"default, both fail, under -require-backend: an error", false, true, true, true, true, true, false, true, []bool{false, true}, 1, true},
		{"explicit f32 does not fit: no fallback, a warning and the CPU tower", false, false, false, true, false, false, false, false, []bool{false}, 0, false},
		{"explicit f32 under -require-backend does not fit: an error, no fallback", false, false, true, true, false, false, false, true, []bool{false}, 0, false},
		{"explicit int8 attaches", true, false, false, false, false, true, true, false, []bool{true}, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var loads []bool
			closed := 0
			load := func(i8 bool) (*fakeAttach, error) {
				loads = append(loads, i8)
				return &fakeAttach{int8: i8, failAttach: (!i8 && tc.f32Fails) || (i8 && tc.int8Fails), closed: &closed}, nil
			}
			var warn bytes.Buffer
			enc, i8, resident, err := attachGemma3Tower(load, tc.startInt8, tc.fallback, "cuda", tc.require, &warn)
			if (err != nil) != tc.wantEr {
				t.Fatalf("error = %v, want error %v", err, tc.wantEr)
			}
			if err == nil && enc.int8 != tc.wantInt8 {
				t.Errorf("the returned encoder is int8=%v, want %v", enc.int8, tc.wantInt8)
			}
			if i8 != tc.wantInt8 || resident != tc.wantResident {
				t.Errorf("int8=%v resident=%v, want int8=%v resident=%v", i8, resident, tc.wantInt8, tc.wantResident)
			}
			if len(loads) != len(tc.wantLoads) {
				t.Fatalf("loads %v, want %v", loads, tc.wantLoads)
			}
			for i := range loads {
				if loads[i] != tc.wantLoads[i] {
					t.Errorf("loads %v, want %v", loads, tc.wantLoads)
				}
			}
			if closed != tc.wantClosed {
				t.Errorf("released %d encoders, want %d (the float32 one that did not attach, and no other)", closed, tc.wantClosed)
			}
			if note := strings.Contains(warn.String(), "did not fit on the device"); note != tc.wantNote {
				t.Errorf("fallback note printed = %v, want %v (output %q)", note, tc.wantNote, warn.String())
			}
		})
	}
}

func TestAttachGemma3Tower_loadErrorPropagates(t *testing.T) {
	closed := 0
	_, _, _, err := attachGemma3Tower(func(bool) (*fakeAttach, error) { return nil, errors.New("no weights") }, false, true, "cuda", false, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "no weights") {
		t.Errorf("error = %v, want the load error", err)
	}
	if closed != 0 {
		t.Errorf("closed %d", closed)
	}
}

// gemma3Checkpoint writes a Gemma 3 vision config and a sparse safetensors file of the given size, standing in for a checkpoint of that size.
func gemma3Checkpoint(t *testing.T, safetensorsBytes int64) string {
	t.Helper()
	dir := t.TempDir()
	cfg := `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, "model-00001-of-00001.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Truncate(safetensorsBytes); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestResolveGemma3VisionQuant pins the pre-load choice: the unset default stays float32 where the card holds the decoder, one floor-context KV slot and the float32 tower, and becomes int8, with a note,
// where it does not. The 4B (8.6 GB of safetensors) needs about 6.1 GB by the estimate (measured: 5.75 GB for one slot); the card in the tests is stood in for by cudaFreeBytes.
func TestResolveGemma3VisionQuant(t *testing.T) {
	old := cudaFreeBytes
	t.Cleanup(func() { cudaFreeBytes = old })
	dir4b := gemma3Checkpoint(t, 8_600_000_000)
	mk := func(dir, quant, backend string) config {
		c := config{visionQuant: quant, models: modelFlag{{path: dir}}}
		c.load.Backend = backend
		return c
	}
	for _, tc := range []struct {
		name     string
		cfg      config
		free     int64
		probe    bool
		want     string
		wantNote bool
	}{
		{"the 8 GB card idle (7.3 GB free): float32", mk(dir4b, "", "cuda"), 7_300_000_000, true, "", false},
		{"free just over the need: float32", mk(dir4b, "", "cuda"), 6_300_000_000, true, "", false},
		{"free just under the need: int8, with a note", mk(dir4b, "", "cuda"), 5_900_000_000, true, "int8", true},
		{"a 6 GB card (5.0 GB free): int8, with a note", mk(dir4b, "", "cuda"), 5_000_000_000, true, "int8", true},
		{"explicit f32 is never changed, however little is free", mk(dir4b, "f32", "cuda"), 3_000_000_000, true, "f32", false},
		{"explicit int8 is never changed", mk(dir4b, "int8", "cuda"), 20_000_000_000, true, "int8", false},
		{"not cuda: untouched", mk(dir4b, "", "webgpu"), 1_000_000_000, true, "", false},
		{"no free-VRAM probe: untouched (the attach fallback is the guard)", mk(dir4b, "", "cuda"), 0, false, "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cudaFreeBytes = func() (int64, bool) { return tc.free, tc.probe }
			var note bytes.Buffer
			got := resolveGemma3VisionQuant(tc.cfg, &note)
			if got.visionQuant != tc.want {
				t.Errorf("visionQuant = %q, want %q (note %q)", got.visionQuant, tc.want, note.String())
			}
			if has := strings.Contains(note.String(), "using the int8 device tower"); has != tc.wantNote {
				t.Errorf("note printed = %v, want %v (%q)", has, tc.wantNote, note.String())
			}
		})
	}
	// S18 on the Mac: the same choice on Metal, against its resident budget. The 4B's estimate: 8.6 GB of safetensors at 0.34 is 2.92 GB on the device,
	// twice that without a sidecar, plus one f16 KV slot at 2048 (0.40), the f16 tower (1.29) and the margin (0.40): about 8.5 GB from a directory with no
	// sidecar, 5.6 GB with one.
	oldM := metalFreeBytes
	t.Cleanup(func() { metalFreeBytes = oldM })
	for _, tc := range []struct {
		name     string
		free     int64
		probe    bool
		want     string
		wantNote bool
	}{
		{"a 64 GB Mac (44.8 GB budget): f16", 44_800_000_000, true, "", false},
		{"the 16 GB Mac (4.2-4.9 GB budget): int8, with a note", 4_600_000_000, true, "int8", true},
		{"no budget probe: untouched", 0, false, "", false},
	} {
		t.Run("metal: "+tc.name, func(t *testing.T) {
			metalFreeBytes = func() (int64, bool) { return tc.free, tc.probe }
			var note bytes.Buffer
			got := resolveGemma3VisionQuant(mk(dir4b, "", "metal"), &note)
			if got.visionQuant != tc.want {
				t.Errorf("visionQuant = %q, want %q (note %q)", got.visionQuant, tc.want, note.String())
			}
			if has := strings.Contains(note.String(), "using the int8 Metal tower"); has != tc.wantNote {
				t.Errorf("note printed = %v, want %v (%q)", has, tc.wantNote, note.String())
			}
		})
	}
	metalFreeBytes = func() (int64, bool) { return 1, true }
	if got := resolveGemma3VisionQuant(mk(dir4b, "f32", "metal"), io.Discard); got.visionQuant != "f32" {
		t.Errorf("an explicit f32 on Metal became %q", got.visionQuant)
	}
	// other models and several models: untouched
	cudaFreeBytes = func() (int64, bool) { return 100, true }
	other := t.TempDir()
	_ = os.WriteFile(filepath.Join(other, "config.json"), []byte(`{"model_type":"qwen2_5_vl"}`), 0o644)
	if got := resolveGemma3VisionQuant(mk(other, "", "cuda"), &bytes.Buffer{}); got.visionQuant != "" {
		t.Errorf("a non-Gemma-3 model got visionQuant %q", got.visionQuant)
	}
	two := mk(dir4b, "", "cuda")
	two.models = modelFlag{{path: dir4b}, {path: dir4b}}
	if got := resolveGemma3VisionQuant(two, &bytes.Buffer{}); got.visionQuant != "" {
		t.Errorf("two models got visionQuant %q", got.visionQuant)
	}
	// an unresolved reference (hf:...) and a GGUF: the first is left alone, the second is read at its size
	if _, ok := gemma3ResidentWeightsEstimate("hf:google/gemma-3-4b-it"); ok {
		t.Error("an hf: reference was estimated")
	}
	gg := filepath.Join(t.TempDir(), "g.gguf")
	f, _ := os.Create(gg)
	_ = f.Truncate(2_490_000_000)
	f.Close()
	if w, ok := gemma3ResidentWeightsEstimate(gg); !ok || w != 2_490_000_000 {
		t.Errorf("gguf estimate = %d, %v", w, ok)
	}
}
