package serveapp

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// The WebGPU towers are the default (S12-defaults), so the fit guard has to see them: towerReserve prices a WebGPU tower, and on a
// unified-memory machine webgpuTowerFits keeps it on the CPU, with the reason, when it exceeds the fit guard's budget.

const (
	gemma3Cfg = `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`
	gbF       = float64(1 << 30)
)

func webgpuCfg(t *testing.T, js, quant string) (config, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	c := config{visionPath: dir, visionQuant: quant}
	c.load.Backend = "webgpu"
	return c, dir
}

// stubWebGPU stands in for a gpu build's tower registrations and a machine: unified-memory or discrete, with avail bytes available.
func stubWebGPU(t *testing.T, unified bool, avail int64) {
	t.Helper()
	oldR, oldU, oldA := webgpuTowerRegistered, webgpuTowerUnified, webgpuTowerAvailable
	webgpuTowerRegistered = func(string) bool { return true }
	webgpuTowerUnified = func() bool { return unified }
	webgpuTowerAvailable = func() int64 { return avail }
	t.Cleanup(func() { webgpuTowerRegistered, webgpuTowerUnified, webgpuTowerAvailable = oldR, oldU, oldA })
}

func TestTowerReserve_webgpu(t *testing.T) {
	stubWebGPU(t, true, 0)
	c, dir := webgpuCfg(t, gemma3Cfg, "")
	// the default is the float32 tower: the same arithmetic the CUDA figure uses (about 2 GB for Gemma 3 4B)
	if got := float64(towerReserve(c, dir)) / 1e9; got < 1.9 || got > 2.5 {
		t.Errorf("Gemma 3 float32 WebGPU reserve %.2f GB, want 1.9-2.5", got)
	}
	// Gemma 3's explicit int8 is a WebGPU device tower too, at the int8 figure
	c.visionQuant = "int8"
	if mib := float64(towerReserve(c, dir)) / (1 << 20); mib < 520 || mib > 640 {
		t.Errorf("Gemma 3 int8 WebGPU reserve %.0f MiB, want 520-640", mib)
	}
	// -vision-device cpu, a binary with no WebGPU tower for the family, and another family's int8 tower (the CPU's) hold nothing
	c.visionQuant = ""
	cpu := c
	cpu.visionDevice = "cpu"
	if got := towerReserve(cpu, dir); got != 0 {
		t.Errorf("-vision-device cpu reserves %d, want 0", got)
	}
	webgpuTowerRegistered = func(string) bool { return false }
	if got := towerReserve(c, dir); got != 0 {
		t.Errorf("no WebGPU tower registered reserves %d, want 0", got)
	}
	webgpuTowerRegistered = func(string) bool { return true }
	q, qdir := webgpuCfg(t, `{"model_type":"qwen2_5_vl","vision_config":{"depth":32,"hidden_size":1280,"in_channels":3,"intermediate_size":3420,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, "int8")
	if got := towerReserve(q, qdir); got != 0 {
		t.Errorf("Qwen2.5-VL's int8 tower is the CPU's, reserves %d, want 0", got)
	}
	q.visionQuant = ""
	if got := float64(towerReserve(q, qdir)) / 1e9; got < 2 {
		t.Errorf("Qwen2.5-VL's float32 WebGPU tower reserves %.2f GB, want its weights and scratch (over 2 GB)", got)
	}
}

// A load that fits only without the tower: the tower stays on the CPU with the reason named, or the load is refused under
// -require-backend. Only a unified-memory machine has a figure to compare; a discrete GPU and an unreadable figure proceed.
func TestWebgpuTowerFits(t *testing.T) {
	c, dir := webgpuCfg(t, gemma3Cfg, "")
	need := towerReserve(func() config { stubWebGPU(t, true, 1<<40); return c }(), dir)
	if need <= 0 {
		t.Fatalf("test setup: the tower is priced at %d", need)
	}
	budget := func(avail int64) int64 { return int64(0.70 * float64(avail)) }
	cases := []struct {
		name    string
		unified bool
		avail   int64
		mod     func(*config)
		onCPU   bool // the tower is kept on the CPU
		err     bool
	}{
		{"unified, plenty available", true, 8 << 30, nil, false, false},
		{"unified, the budget just covers it", true, need*10/7 + 1<<20, nil, false, false},
		{"unified, the budget is one MiB short: the load fits only without the tower", true, need*10/7 - 1<<20, nil, true, false},
		{"unified, little available", true, 1 << 30, nil, true, false},
		{"unified, little available, -require-backend refuses", true, 1 << 30, func(c *config) { c.requireBE = true }, false, true},
		{"unified, the available memory cannot be read: proceeds", true, 0, nil, false, false},
		{"discrete GPU: no figure, proceeds", false, 1 << 20, nil, false, false},
		{"unified, already -vision-device cpu", true, 1 << 20, func(c *config) { c.visionDevice = "cpu" }, true, false},
		{"unified, int8 Gemma 3 (558 MiB) fits where float32 does not", true, 1 << 30, func(c *config) { c.visionQuant = "int8" }, false, false},
	}
	for _, tc := range cases {
		stubWebGPU(t, tc.unified, tc.avail)
		cfg := c
		if tc.mod != nil {
			tc.mod(&cfg)
		}
		var note bytes.Buffer
		got, err := webgpuTowerFits(cfg, dir, &note)
		switch {
		case tc.err:
			if err == nil || !strings.Contains(err.Error(), "-require-backend") || !strings.Contains(err.Error(), "WebGPU vision tower needs about") {
				t.Errorf("%s: err %v, want a -require-backend refusal naming the tower's size", tc.name, err)
			}
			continue
		case err != nil:
			t.Errorf("%s: unexpected error %v", tc.name, err)
			continue
		}
		if onCPU := got.towerBackend() == "cpu"; onCPU != tc.onCPU {
			t.Errorf("%s: tower on cpu %v, want %v (need %.2f GB, budget %.2f GB)", tc.name, onCPU, tc.onCPU, float64(need)/gbF, float64(budget(tc.avail))/gbF)
		}
		if fell := strings.Contains(note.String(), "the tower runs on the CPU"); fell != (tc.onCPU && cfg.visionDevice != "cpu") {
			t.Errorf("%s: note %q, want a fallback note exactly when the guard moved the tower", tc.name, note.String())
		}
	}
}

// loadVisionTower asks the guard: a Qwen2.5-VL tower that fits is attached on the device (its host blocks are released once), one
// that does not stays on the CPU and releases nothing.
func TestLoadVisionTower_webgpuTowerGuard(t *testing.T) {
	got := countReleases(t)
	vision.RegisterQwenResident(func(*vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return attachedQwenResident{}, nil
	})
	t.Cleanup(func() { vision.RegisterQwenResident(nil) })
	for _, c := range []struct {
		name  string
		avail int64
		want  int
	}{{"fits", 1 << 40, 1}, {"does not fit", 1 << 10, 0}} {
		stubWebGPU(t, true, c.avail)
		cfg := config{visionPath: "../../../testdata/qwen25vl-tiny", models: []modelSpec{{name: "m", path: "../../../testdata/qwen25vl-tiny"}}}
		cfg.load.Backend = "webgpu"
		s := &server{models: map[string]*loadedModel{}}
		before := got["Qwen2.5-VL"]
		if err := s.loadVisionTower(cfg); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if n := got["Qwen2.5-VL"] - before; n != c.want {
			t.Errorf("%s: %d tower releases, want %d (1 means the device tower attached)", c.name, n, c.want)
		}
	}
}
