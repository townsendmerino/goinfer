package serveapp

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// The WebGPU towers are the default (S12-defaults), so the fit guard has to see them: towerReserve prices a WebGPU tower, and on a
// unified-memory machine webgpuTowerFits refuses the load, with the reason, when the tower needs more than the memory available.

const (
	gibF      = float64(1 << 30)
	gemma3Cfg = `{"model_type":"gemma3","vision_config":{"hidden_size":1152,"intermediate_size":4304,"num_hidden_layers":27,"patch_size":14,"image_size":896,"num_channels":3}}`
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
	// the default is the float32 tower: the same arithmetic the CUDA figure uses (about 2 GiB for Gemma 3 4B)
	if got := float64(towerReserve(c, dir)) / gibF; got < 1.7 || got > 2.3 {
		t.Errorf("Gemma 3 float32 WebGPU reserve %.2f GiB, want 1.7-2.3", got)
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
	if got := float64(towerReserve(q, qdir)) / gibF; got < 2 {
		t.Errorf("Qwen2.5-VL's float32 WebGPU tower reserves %.2f GiB, want its weights and scratch (over 2 GiB)", got)
	}
}

// A tower that needs more than the memory available now is refused on a unified-memory machine (the CPU float32 tower holds the same
// RAM, so it is not a fallback); there is one budget, the backend's, and this is its physical backstop, so the boundary is the tower's
// own size and not a second fraction of what is left. A discrete GPU, an unreadable figure, -vision-device cpu and a tower that fits
// proceed.
func TestWebgpuTowerFits(t *testing.T) {
	c, dir := webgpuCfg(t, gemma3Cfg, "")
	stubWebGPU(t, true, 1<<40)
	need := towerReserve(c, dir)
	if need <= 0 {
		t.Fatalf("test setup: the tower is priced at %d", need)
	}
	cases := []struct {
		name    string
		unified bool
		avail   int64
		mod     func(*config)
		refuse  bool
	}{
		{"unified, plenty available", true, 8 << 30, nil, false},
		{"unified, exactly the tower's size is available: no second discount", true, need, nil, false},
		{"unified, one MiB short: the load fits only without the tower", true, need - 1<<20, nil, true},
		{"unified, little available", true, 1 << 30, nil, true},
		{"unified, little available, -require-backend refuses the same", true, 1 << 30, func(c *config) { c.requireBE = true }, true},
		{"unified, the available memory cannot be read: proceeds", true, 0, nil, false},
		{"discrete GPU: no figure, proceeds", false, 1 << 20, nil, false},
		{"unified, -vision-device cpu is the user's own choice: proceeds", true, 1 << 20, func(c *config) { c.visionDevice = "cpu" }, false},
		{"unified, int8 Gemma 3 (558 MiB) fits where float32 does not", true, 1 << 30, func(c *config) { c.visionQuant = "int8" }, false},
	}
	for _, tc := range cases {
		stubWebGPU(t, tc.unified, tc.avail)
		cfg := c
		if tc.mod != nil {
			tc.mod(&cfg)
		}
		err := webgpuTowerFits(cfg, dir)
		switch {
		case tc.refuse && (err == nil || !strings.Contains(err.Error(), "WebGPU vision tower needs about") || !strings.Contains(err.Error(), "not a fallback") ||
			!strings.Contains(err.Error(), "-vision-device cpu")):
			t.Errorf("%s: err %v, want a refusal that names the tower's size, says the CPU is not a fallback and names the way out", tc.name, err)
		case !tc.refuse && err != nil:
			t.Errorf("%s: unexpected error %v", tc.name, err)
		}
	}
}

// The refusal advises -vision-quant int8 only for Gemma 3, whose int8 form is a gated WebGPU device tower. For every other family it
// moves the tower to the CPU's lossy int8 path (ungated for GLM-OCR) and prices it at zero, so it is not offered.
func TestWebgpuTowerFits_int8AdviceIsGemma3Only(t *testing.T) {
	stubWebGPU(t, true, 1<<20)
	for _, tc := range []struct {
		name, js string
		advise   bool
	}{
		{"gemma3", gemma3Cfg, true},
		{"qwen2_5_vl", `{"model_type":"qwen2_5_vl","vision_config":{"depth":32,"hidden_size":1280,"in_channels":3,"intermediate_size":3420,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, false},
		{"glm_ocr", `{"model_type":"glm_ocr","vision_config":{"depth":24,"hidden_size":1024,"in_channels":3,"intermediate_size":4096,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}`, false},
		{"gemma4", `{"model_type":"gemma4","vision_config":{"hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"patch_size":16,"position_embedding_size":10240,"default_output_length":280,"pooling_kernel_size":3}}`, false},
	} {
		c, dir := webgpuCfg(t, tc.js, "")
		err := webgpuTowerFits(c, dir)
		if err == nil {
			t.Errorf("%s: no refusal with 1 MiB available", tc.name)
			continue
		}
		if got := strings.Contains(err.Error(), "-vision-quant int8"); got != tc.advise {
			t.Errorf("%s: advises int8 %v, want %v: %v", tc.name, got, tc.advise, err)
		}
		if !strings.Contains(err.Error(), "lower -ctx or -kv-sessions") || !strings.Contains(err.Error(), "-vision-device cpu") {
			t.Errorf("%s: the refusal does not name a smaller context and the CPU tower: %v", tc.name, err)
		}
	}
}

// loadVisionTower asks the guard: a Qwen2.5-VL tower that fits is attached on the device (its host blocks are released once), one
// that does not refuses the load and attaches nothing.
func TestLoadVisionTower_webgpuTowerGuard(t *testing.T) {
	got := countReleases(t)
	vision.RegisterQwenResident(func(*vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return attachedQwenResident{}, nil
	})
	t.Cleanup(func() { vision.RegisterQwenResident(nil) })
	for _, c := range []struct {
		name   string
		avail  int64
		want   int
		refuse bool
	}{{"fits", 1 << 40, 1, false}, {"does not fit", 1 << 10, 0, true}} {
		stubWebGPU(t, true, c.avail)
		cfg := config{visionPath: "../../../testdata/qwen25vl-tiny", models: []modelSpec{{name: "m", path: "../../../testdata/qwen25vl-tiny"}}}
		cfg.load.Backend = "webgpu"
		s := &server{models: map[string]*loadedModel{}}
		before := got["Qwen2.5-VL"]
		err := s.loadVisionTower(cfg)
		if c.refuse != (err != nil) {
			t.Fatalf("%s: loadVisionTower error %v, want an error: %v", c.name, err, c.refuse)
		}
		if n := got["Qwen2.5-VL"] - before; n != c.want {
			t.Errorf("%s: %d tower releases, want %d (1 means the device tower attached)", c.name, n, c.want)
		}
	}
}

// A Qwen3.5+ GGUF mmproj tower has no config.json, so towerReserve cannot price it: under webgpu it stays on the CPU unless
// -vision-device gpu asks for the device (a quant does not), on a unified-memory Mac and on a discrete GPU alike; the other
// backends and -vision-device cpu are untouched.
func TestMMProjTowerBackend(t *testing.T) {
	for _, unified := range []bool{true, false} {
		stubWebGPU(t, unified, 1<<40)
		for _, c := range []struct{ backend, device, quant, want string }{
			{"webgpu", "", "", "cpu"},
			{"webgpu", "auto", "", "cpu"},
			{"webgpu", "", "int8", "cpu"},
			{"webgpu", "", "f32", "cpu"},
			{"webgpu", "gpu", "", "webgpu"},
			{"webgpu", "cpu", "", "cpu"},
			{"metal", "", "", "metal"},
			{"cuda", "", "", "cuda"},
			{"cpu", "", "", "cpu"},
		} {
			cfg := config{visionDevice: c.device, visionQuant: c.quant}
			cfg.load.Backend = c.backend
			if got := mmprojTowerBackend(cfg); got != c.want {
				t.Errorf("unified %v, backend %q, -vision-device %q, -vision-quant %q: mmproj tower on %q, want %q", unified, c.backend, c.device, c.quant, got, c.want)
			}
		}
	}
}

// The default placement of a vision tower under webgpu reads the machine and the family: on a unified-memory Mac a tower with a
// WebGPU form is on the device; on a discrete GPU it stays on the CPU unless -vision-device gpu asks, or, for Gemma 3 only (the
// one family with a WebGPU int8 form), an explicit -vision-quant. A family with no WebGPU form, another backend, and an audio
// encoder (towerBackend) are never moved by the machine.
func TestVisionTowerBackend_webgpuDiscreteDefault(t *testing.T) {
	for _, c := range []struct {
		name, mt, device, quant string
		unified                 bool
		want                    string
	}{
		{"unified default", "gemma4", "", "", true, "webgpu"},
		{"discrete default", "gemma4", "", "", false, "cpu"},
		{"discrete auto", "gemma4", "auto", "", false, "cpu"},
		{"discrete, -vision-device gpu", "gemma4", "gpu", "", false, "webgpu"},
		{"discrete, -vision-device cpu", "gemma4", "cpu", "", false, "cpu"},
		{"discrete gemma3, -vision-quant int8 asks for the WebGPU int8 tower", "gemma3", "", "int8", false, "webgpu"},
		{"discrete gemma3, -vision-quant f32", "gemma3", "", "f32", false, "webgpu"},
		{"discrete gemma4, -vision-quant int8 is the CPU's int8 tower, not a request for the device", "gemma4", "", "int8", false, "cpu"},
		{"discrete qwen2_5_vl, -vision-quant int8", "qwen2_5_vl", "", "int8", false, "cpu"},
		{"discrete qwen3_5, -vision-quant f32", "qwen3_5", "", "f32", false, "cpu"},
		{"discrete glm_ocr, -vision-quant int8", "glm_ocr", "", "int8", false, "cpu"},
		{"discrete family with no WebGPU form (the stub registers only the families below)", "mistral3", "", "", false, "webgpu"},
	} {
		stubWebGPU(t, c.unified, 1<<40)
		webgpuTowerRegistered = func(mt string) bool { return mt != "mistral3" }
		cfg := config{visionDevice: c.device, visionQuant: c.quant}
		cfg.load.Backend = "webgpu"
		if got := cfg.visionTowerBackend(c.mt); got != c.want {
			t.Errorf("%s: vision tower backend %q, want %q", c.name, got, c.want)
		}
	}
	// the audio encoder and every other backend are never moved by the machine
	stubWebGPU(t, false, 0)
	cfg := config{}
	cfg.load.Backend = "webgpu"
	if got := cfg.towerBackend(); got != "webgpu" {
		t.Errorf("an audio encoder's tower backend on a discrete GPU is %q, want webgpu", got)
	}
	cfg.load.Backend = "cuda"
	if got := cfg.visionTowerBackend("gemma4"); got != "cuda" {
		t.Errorf("cuda vision tower backend %q, want cuda", got)
	}
}

// keptOnCPUNote is printed only when a WebGPU-capable vision tower exists and is kept on the CPU: not for a text-only load, a CPU-only
// vision tower (mistral3, lfm2_vl), an audio encoder, a tower on the device, or -vision-device cpu. The mmproj note prints for a GGUF
// mmproj tower kept on the CPU. loadVisionTower writes it, once, to stderr.
func TestKeptOnCPUNote(t *testing.T) {
	for _, c := range []struct {
		name, mt, device string
		backend          string
		unified          bool
		want             bool
	}{
		{"discrete, a WebGPU-capable tower kept on the CPU", "gemma4", "", "webgpu", false, true},
		{"discrete, -vision-device gpu: on the device", "gemma4", "gpu", "webgpu", false, false},
		{"discrete, -vision-device cpu is the user's own choice", "gemma4", "cpu", "webgpu", false, false},
		{"unified: on the device", "gemma4", "", "webgpu", true, false},
		{"discrete, a CPU-only vision tower (no WebGPU form)", "mistral3", "", "webgpu", false, false},
		{"discrete, an audio encoder", "voxtral", "", "webgpu", false, false},
		{"discrete, not a tower at all", "", "", "webgpu", false, false},
		{"another backend", "gemma4", "", "cuda", false, false},
		{"mmproj on a Mac, kept on the CPU", "gguf-mmproj", "", "webgpu", true, true},
		{"mmproj on a discrete GPU, kept on the CPU", "gguf-mmproj", "", "webgpu", false, true},
		{"mmproj, -vision-device gpu", "gguf-mmproj", "gpu", "webgpu", true, false},
		{"mmproj, -vision-device cpu", "gguf-mmproj", "cpu", "webgpu", true, false},
		{"mmproj under metal", "gguf-mmproj", "", "metal", true, false},
	} {
		stubWebGPU(t, c.unified, 1<<40)
		webgpuTowerRegistered = func(mt string) bool { return mt == "gemma4" || mt == "gemma3" }
		cfg := config{visionDevice: c.device}
		cfg.load.Backend = c.backend
		note := cfg.keptOnCPUNote(c.mt)
		if (note != "") != c.want || (c.want && !strings.Contains(note, "--vision-device gpu")) {
			t.Errorf("%s: note %q, want a note: %v naming --vision-device gpu", c.name, note, c.want)
		}
	}
}

// loadVisionTower is where the note is written: a text-only load prints nothing; a Gemma 3 checkpoint on a discrete GPU prints it once.
func TestLoadVisionTower_keptOnCPUNoteIsPrintedOnceAndOnlyForATower(t *testing.T) {
	capture := func(f func()) string {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatal(err)
		}
		old := os.Stderr
		os.Stderr = w
		f()
		os.Stderr = old
		_ = w.Close()
		b, _ := io.ReadAll(r)
		return string(b)
	}
	stubWebGPU(t, false, 0)
	webgpuTowerRegistered = func(mt string) bool { return mt == "gemma3" }
	cfg := config{}
	cfg.load.Backend = "webgpu"
	if out := capture(func() { _ = (&server{}).loadVisionTower(cfg) }); strings.Contains(out, "note:") {
		t.Errorf("a load with no vision tower printed %q", out)
	}
	cfg.visionPath = gemma3RealLayoutTiny(t)
	cfg.models = modelFlag{{path: "m"}}
	out := capture(func() { _ = (&server{}).loadVisionTower(cfg) })
	if n := strings.Count(out, "note: under --backend webgpu on a discrete GPU"); n != 1 {
		t.Errorf("a Gemma 3 tower kept on the CPU printed the note %d times: %q", n, out)
	}
}
