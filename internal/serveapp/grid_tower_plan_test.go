package serveapp

import (
	"errors"
	"strings"
	"testing"
)

// TestPlanGridTower pins where a Qwen3.5+ or GLM-OCR tower runs (S2 of docs/tasks/task-multimodal-support-2026-10.md):
// the Metal tower only under --backend metal, float32 and registered; the CPU otherwise with the reason; and each CPU
// fallback a refusal under -require-backend.
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
		{"cuda backend", metal, false, "cuda", true, "", "CPU", ""},
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
