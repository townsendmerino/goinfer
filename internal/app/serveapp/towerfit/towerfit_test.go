package towerfit

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/multimodal"
)

const gemma3Cfg = `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`

func TestTowerVRAMEstimate(t *testing.T) {
	cases := []struct {
		name, mt, cfg string
		maxPixels     int
		lo, hi        float64
	}{
		{"gemma4 E2B", "gemma4", `{"vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"patch_size":16,"position_embedding_size":10240,"default_output_length":280,"pooling_kernel_size":3}}`, 0, 0.80, 0.95},
		{"qwen3.5-0.8b", "qwen3_5", `{"vision_config":{"depth":12,"hidden_size":768,"in_channels":3,"intermediate_size":3072,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 0.60, 0.75},
		{"glm-ocr default ceiling", "glm_ocr", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 2.3, 2.7},
		{"qwen2.5-vl-3b", "qwen2_5_vl", `{"vision_config":{"depth":32,"hidden_size":1280,"in_channels":3,"intermediate_size":3420,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 3.40, 3.60},
		{"qwen3-vl-2b", "qwen3_vl", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}`, 0, 1.55, 1.75},
		{"glm-ocr 1 MP", "glm_ocr", `{"vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, 1_000_000, 1.8, 2.2},
	}
	for _, c := range cases {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(c.cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		d, ok := ReadTowerDims(dir)
		if !ok {
			t.Fatalf("%s: config did not parse into TowerDims", c.name)
		}
		got := float64(TowerVRAMEstimate(c.mt, d, c.maxPixels)) / 1e9
		if got < c.lo || got > c.hi {
			t.Errorf("%s: estimate %.2f, want %.2f-%.2f", c.name, got, c.lo, c.hi)
		}
	}
	if TowerVRAMEstimate("llama", TowerDims{Hidden: 1, Inter: 1, Layers: 1, Patch: 1}, 0) != 0 {
		t.Error("unknown model type must price at 0")
	}
}

func TestReserveAndWebGPUTowerFits(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(gemma3Cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	oldC, oldM, oldW, oldU, oldA := CUDATowerRegistered, MetalTowerRegistered, WebGPUTowerRegistered, WebGPUTowerUnified, WebGPUTowerAvailable
	t.Cleanup(func() {
		CUDATowerRegistered, MetalTowerRegistered, WebGPUTowerRegistered, WebGPUTowerUnified, WebGPUTowerAvailable = oldC, oldM, oldW, oldU, oldA
	})
	CUDATowerRegistered = func(string) bool { return true }
	MetalTowerRegistered = func(string) bool { return true }
	WebGPUTowerRegistered = func(string) bool { return true }
	WebGPUTowerUnified = func() bool { return true }
	WebGPUTowerAvailable = func() int64 { return 1 << 30 }

	cfg := Config{Backend: "webgpu", VisionPath: dir}
	need := Reserve(cfg, dir)
	if need <= 0 {
		t.Fatalf("expected positive WebGPU reserve, got %d", need)
	}
	err := WebGPUTowerFits(cfg, dir)
	if err == nil || !strings.Contains(err.Error(), "-vision-quant int8") || !strings.Contains(err.Error(), "not a fallback") {
		t.Errorf("WebGPUTowerFits: got %v, want refusal advising -vision-quant int8", err)
	}
	WebGPUTowerAvailable = func() int64 { return need }
	if err := WebGPUTowerFits(cfg, dir); err != nil {
		t.Errorf("exact fit refused: %v", err)
	}
	cfg.VisionDevice = "cpu"
	if got := Reserve(cfg, dir); got != 0 {
		t.Errorf("cpu device reserved %d, want 0", got)
	}
}

func TestTowerInt8AndRoute(t *testing.T) {
	backends := []string{"cpu", "auto", "metal", "webgpu", "cuda"}
	unasked := map[string]map[string]bool{
		"":            {"webgpu": true, "cuda": true},
		"gemma3":      {},
		"gemma4":      {},
		"qwen2_5_vl":  {},
		"qwen3_5":     {},
		"qwen3_5_moe": {},
		"qwen3_vl":    {},
		"glm_ocr":     {},
	}
	for fam, want := range unasked {
		for _, be := range backends {
			if got := TowerInt8(fam, "", be); got != want[be] {
				t.Errorf("family %q on %s: int8 = %v, want %v", fam, be, got, want[be])
			}
			if TowerInt8(fam, "f32", be) {
				t.Errorf("family %q on %s: f32 loaded int8", fam, be)
			}
			if !TowerInt8(fam, "int8", be) {
				t.Errorf("family %q on %s: int8 ignored", fam, be)
			}
		}
	}
	oldW, oldU := WebGPUTowerRegistered, WebGPUTowerUnified
	t.Cleanup(func() { WebGPUTowerRegistered, WebGPUTowerUnified = oldW, oldU })
	WebGPUTowerRegistered = func(string) bool { return true }
	WebGPUTowerUnified = func() bool { return false }
	r := (Config{Backend: "webgpu"}).RouteVisionTower("gemma4", false)
	if r.Backend != "cpu" || r.Note != DiscreteKeptNote {
		t.Errorf("discrete default route = %+v", r)
	}
	mm := (Config{Backend: "webgpu"}).RouteVisionTower("qwen3_5", true)
	if mm.Backend != "cpu" || mm.Note != MMProjKeptNote {
		t.Errorf("mmproj route = %+v", mm)
	}
}

type fakeOOMGrid struct{ closed bool }

func (f *fakeOOMGrid) Name() string { return "fake" }
func (f *fakeOOMGrid) Hidden([]float32, [][3]int) ([]float32, error) {
	return nil, errors.New("device allocation failed (fake)")
}
func (f *fakeOOMGrid) Close() error { f.closed = true; return nil }

func TestRecoverAndDeviceFallback(t *testing.T) {
	out, err := RecoverDeviceTower("Qwen2.5-VL", func() ([]float32, error) {
		panic("cuda: device allocation failed: CUDA_ERROR_OUT_OF_MEMORY")
	})
	if out != nil || err == nil || !strings.Contains(err.Error(), "Qwen2.5-VL tower failed") || !strings.Contains(err.Error(), "-vision-device cpu") {
		t.Errorf("RecoverDeviceTower: got %v, %v", out, err)
	}
	var fb DeviceFallback
	oom := &fakeOOMGrid{}
	var acc multimodal.GridTowerAccelerator = oom
	if _, err := fb.Run("Qwen3-VL", true, &acc, func(a multimodal.GridTowerAccelerator) ([]float32, error) {
		return a.Hidden(nil, nil)
	}); err == nil || acc == nil || oom.closed {
		t.Errorf("require=true must refuse fallback and keep accelerator: err=%v acc=%v closed=%v", err, acc, oom.closed)
	}
	got, err := fb.Run("Qwen3-VL", false, &acc, func(a multimodal.GridTowerAccelerator) ([]float32, error) {
		if a == nil {
			return []float32{42}, nil
		}
		return a.Hidden(nil, nil)
	})
	if err != nil || len(got) != 1 || got[0] != 42 || acc != nil || !oom.closed {
		t.Errorf("require=false must close accelerator and run CPU fallback: got=%v err=%v acc=%v closed=%v", got, err, acc, oom.closed)
	}
}
