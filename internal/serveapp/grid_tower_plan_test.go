package serveapp

import (
	"errors"
	"strings"
	"testing"
)

// TestPlanGridTower pins where a Qwen3.5+ or GLM-OCR tower runs (S2 of docs/tasks/task-multimodal-support-2026-10.md): the device tower only under --backend metal or cuda,
// float32 and registered for THAT backend; the CPU otherwise with the reason; and each CPU fallback a refusal under -require-backend.
func TestPlanGridTower(t *testing.T) {
	metal := []string{"metal"}
	cases := []struct {
		name       string
		registered []string
		int8       bool
		backend    string
		require    bool
		device     string
		where      string
		err        string
	}{
		{"cpu backend", metal, false, "cpu", false, "", "CPU", ""},
		{"cuda backend, only a Metal tower registered", metal, false, "cuda", false, "", "CPU (no CUDA tower in this binary)", ""},
		{"cuda backend, only a Metal tower registered, require", metal, false, "cuda", true, "", "", "no CUDA"},
		{"cuda", []string{"cuda"}, false, "cuda", false, "cuda", "CUDA", ""},
		{"cuda require", []string{"cuda"}, false, "cuda", true, "cuda", "CUDA", ""},
		{"cuda int8", []string{"cuda"}, true, "cuda", false, "", "CPU (-vision-quant int8; the CUDA tower is float32)", ""},
		{"cuda int8 require", []string{"cuda"}, true, "cuda", true, "", "", "CUDA tower is float32"},
		{"metal backend, only a CUDA tower registered", []string{"cuda"}, false, "metal", false, "", "CPU (no Metal tower in this binary)", ""},
		{"webgpu", []string{"metal", "cuda"}, false, "webgpu", true, "", "CPU", ""},
		{"metal", metal, false, "metal", false, "metal", "Metal", ""},
		{"metal int8", metal, true, "metal", false, "", "CPU (-vision-quant int8; the Metal tower is float32)", ""},
		{"metal int8 require", metal, true, "metal", true, "", "", "float32"},
		{"not registered", nil, false, "metal", false, "", "CPU (no Metal tower in this binary)", ""},
		{"not registered require", nil, false, "metal", true, "", "", "no Metal"},
	}
	for _, c := range cases {
		p, err := planGridTower("Qwen3.5", c.registered, c.int8, c.backend, c.require)
		if c.err != "" {
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: err %v, want one containing %q", c.name, err, c.err)
			}
			continue
		}
		if err != nil || p.device != c.device || p.where != c.where {
			t.Errorf("%s: got %+v, %v; want device %q where %q", c.name, p, err, c.device, c.where)
		}
	}

	// A device that declines at first use: the CPU with a note, or a refusal under -require-backend.
	p := gridTowerPlan{device: "metal", where: "Metal"}
	if acc, err := p.started("GLM-OCR", nil, errors.New("no device")); acc != nil || err != nil || p.where != "CPU (metal declined)" {
		t.Errorf("declined without -require-backend: acc %v err %v where %q", acc, err, p.where)
	}
	p = gridTowerPlan{device: "metal", where: "Metal", require: true}
	if _, err := p.started("GLM-OCR", nil, errors.New("no device")); err == nil || !strings.Contains(err.Error(), "-require-backend") {
		t.Errorf("declined under -require-backend: err %v", err)
	}
}

// TestTowerBackend: -vision-device cpu keeps a device tower on the CPU while the model keeps its backend; auto follows
// the backend.
func TestTowerBackend(t *testing.T) {
	var c config
	c.load.Backend = "metal"
	for dev, want := range map[string]string{"auto": "metal", "cpu": "cpu"} {
		c.visionDevice = dev
		if got := c.towerBackend(); got != want {
			t.Errorf("-vision-device %s under metal: tower backend %q, want %q", dev, got, want)
		}
	}
	c.visionDevice = "gpu"
	if err := (&server{}).loadVisionTower(c); err == nil || !strings.Contains(err.Error(), "-vision-device") {
		t.Errorf("an unknown -vision-device must be refused, got %v", err)
	}
}

// chooseGemma4Tower follows the same backend-generic rules (S4): the CPU with the reason for a backend that has no device tower, an int8 tower, or a binary
// that registers none for the backend, and a refusal for each under -require-backend. (A registered tower is not exercised here: the registry is global, and
// this package's tests must not leave a fake "cuda" tower behind; the served gates cover the attach.)
func TestChooseGemma4Tower_placement(t *testing.T) {
	cases := []struct {
		name    string
		int8    bool
		backend string
		require bool
		where   string
		errPart string
	}{
		{"cpu", false, "cpu", true, "CPU", ""},
		{"webgpu", false, "webgpu", true, "CPU", ""},
		{"cuda int8", true, "cuda", false, "CPU (-vision-quant int8; the CUDA tower is float32)", ""},
		{"cuda int8 require", true, "cuda", true, "", "Gemma 4 CUDA tower is float32"},
		{"cuda not in this binary", false, "cuda", false, "CPU (no CUDA tower in this binary)", ""},
		{"cuda not in this binary, require", false, "cuda", true, "", "no CUDA Gemma 4 tower"},
		{"metal int8", true, "metal", false, "CPU (-vision-quant int8; the Metal tower is float32)", ""},
	}
	for _, c := range cases {
		acc, where, err := chooseGemma4Tower(nil, c.int8, c.backend, c.require)
		if c.errPart != "" {
			if err == nil || !strings.Contains(err.Error(), c.errPart) {
				t.Errorf("%s: err %v, want one containing %q", c.name, err, c.errPart)
			}
			continue
		}
		if err != nil || acc != nil || where != c.where {
			t.Errorf("%s: got acc %v where %q err %v; want CPU where %q", c.name, acc, where, err, c.where)
		}
	}
}
