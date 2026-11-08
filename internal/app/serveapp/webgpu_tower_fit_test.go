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

// routeVisionTower is the one table for where a vision tower runs under --backend webgpu and what stderr says: family x machine x
// -vision-device x -vision-quant x whether this binary has a WebGPU tower for the family -> backend and note. "" for a note means none.
func TestRouteVisionTower(t *testing.T) {
	const (
		dNote = discreteKeptNote
		mNote = mmprojKeptNote
	)
	type row struct {
		name, family       string
		mmproj             bool
		backend            string // the config's --backend
		unified            bool
		device, quant      string
		registered         bool // this binary has a WebGPU tower for the family
		wantBackend, wantN string
	}
	rows := []row{
		// a unified-memory Mac: the device tower by default, no note
		{"unified default", "gemma4", false, "webgpu", true, "", "", true, "webgpu", ""},
		{"unified, auto", "qwen2_5_vl", false, "webgpu", true, "auto", "", true, "webgpu", ""},
		{"unified, -vision-device cpu", "gemma4", false, "webgpu", true, "cpu", "", true, "cpu", ""},
		// a discrete GPU: the CPU unless the device is asked for
		{"discrete default", "gemma4", false, "webgpu", false, "", "", true, "cpu", dNote},
		{"discrete, auto", "qwen3_vl", false, "webgpu", false, "auto", "", true, "cpu", dNote},
		{"discrete, -vision-device gpu", "gemma4", false, "webgpu", false, "gpu", "", true, "webgpu", ""},
		{"discrete, -vision-device cpu is the user's own choice", "gemma4", false, "webgpu", false, "cpu", "", true, "cpu", ""},
		{"discrete gemma3, -vision-quant int8 asks for the WebGPU int8 tower", "gemma3", false, "webgpu", false, "", "int8", true, "webgpu", ""},
		{"discrete gemma3, -vision-quant f32 asks for the device", "gemma3", false, "webgpu", false, "", "f32", true, "webgpu", ""},
		{"discrete gemma3, nothing asked", "gemma3", false, "webgpu", false, "", "", true, "cpu", dNote},
		{"discrete qwen3_5, -vision-quant f32 does not ask", "qwen3_5", false, "webgpu", false, "", "f32", true, "cpu", dNote},
		// an int8 tower other than Gemma 3's is the CPU's by the quant: the backend is left for the loader to refuse under
		// -require-backend, and no note says the device would move it
		{"unified gemma4 int8", "gemma4", false, "webgpu", true, "", "int8", true, "webgpu", ""},
		{"discrete gemma4 int8", "gemma4", false, "webgpu", false, "", "int8", true, "webgpu", ""},
		{"discrete qwen2_5_vl int8", "qwen2_5_vl", false, "webgpu", false, "", "int8", true, "webgpu", ""},
		{"discrete glm_ocr int8", "glm_ocr", false, "webgpu", false, "", "int8", true, "webgpu", ""},
		{"discrete qwen3_vl int8, -vision-device gpu", "qwen3_vl", false, "webgpu", false, "gpu", "int8", true, "webgpu", ""},
		// a build with no WebGPU tower for the family: the discrete default is still the CPU (so -require-backend passes), with no note
		{"discrete, family not registered in this binary", "gemma4", false, "webgpu", false, "", "", false, "cpu", ""},
		{"unified, family not registered in this binary", "gemma4", false, "webgpu", true, "", "", false, "webgpu", ""},
		// CPU-only vision towers and the unknown
		{"discrete mistral3", "mistral3", false, "webgpu", false, "", "", false, "cpu", ""},
		{"discrete lfm2_vl", "lfm2_vl", false, "webgpu", false, "", "", false, "cpu", ""},
		{"discrete, no family (text-only or unknown)", "", false, "webgpu", false, "", "", false, "webgpu", ""},
		// a Qwen3.5+ GGUF mmproj tower: the CPU unless -vision-device gpu, on either machine; a quant does not ask
		{"mmproj on a Mac", "qwen3_5", true, "webgpu", true, "", "", true, "cpu", mNote},
		{"mmproj on a discrete GPU", "qwen3_5", true, "webgpu", false, "", "", true, "cpu", mNote},
		{"mmproj, -vision-quant f32 does not ask", "qwen3_5", true, "webgpu", true, "", "f32", true, "cpu", mNote},
		{"mmproj, -vision-device gpu", "qwen3_5", true, "webgpu", true, "gpu", "", true, "webgpu", ""},
		{"mmproj, int8 is the CPU's by the quant", "qwen3_5", true, "webgpu", true, "", "int8", true, "webgpu", ""},
		{"mmproj, -vision-device cpu", "qwen3_5", true, "webgpu", true, "cpu", "", true, "cpu", ""},
		{"mmproj, no WebGPU tower in this binary: the CPU, no note", "qwen3_5", true, "webgpu", true, "", "", false, "cpu", ""},
		// other backends are never moved
		{"cuda", "gemma4", false, "cuda", false, "", "", true, "cuda", ""},
		{"metal int8", "gemma4", false, "metal", true, "", "int8", true, "metal", ""},
		{"cuda mmproj", "qwen3_5", true, "cuda", false, "", "", true, "cuda", ""},
		{"cpu", "gemma4", false, "cpu", false, "", "", true, "cpu", ""},
	}
	for _, r := range rows {
		stubWebGPU(t, r.unified, 1<<40)
		webgpuTowerRegistered = func(string) bool { return r.registered }
		cfg := config{visionDevice: r.device, visionQuant: r.quant}
		cfg.load.Backend = r.backend
		got := cfg.routeVisionTower(r.family, r.mmproj)
		if got.Backend != r.wantBackend || got.Note != r.wantN {
			t.Errorf("%s: route {%q, note %q}, want {%q, note %q}", r.name, got.Backend, got.Note, r.wantBackend, r.wantN)
		}
	}
	// an audio encoder never comes through the vision route: towerBackend is the plain backend on a discrete GPU
	stubWebGPU(t, false, 0)
	cfg := config{}
	cfg.load.Backend = "webgpu"
	if got := cfg.towerBackend(); got != "webgpu" {
		t.Errorf("an audio encoder's tower backend on a discrete GPU is %q, want webgpu", got)
	}
}

// loadVisionTower writes the note, once, and only for a tower this binary could have put on the device: a text-only load, a CPU-only
// vision tower (mistral3), an audio encoder (voxtral, qwen3_asr), a tower on the device (unified memory), an int8 tower that is
// the CPU's by the quant, and -vision-device cpu print nothing. The tower directories hold a config.json only: the family loaders
// fail on the missing weights, which is after the decision under test.
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
	tower := func(modelType string) string {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"`+modelType+`"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	registered := map[string]bool{"gemma3": true, "gemma4": true, "qwen2_5_vl": true}
	for _, c := range []struct {
		name, mt, device, quant string
		unified                 bool
		want                    int
	}{
		{"gemma4 on a discrete GPU", "gemma4", "", "", false, 1},
		{"gemma4 on a discrete GPU, -vision-device gpu", "gemma4", "gpu", "", false, 0},
		{"gemma4 on a Mac", "gemma4", "", "", true, 0},
		{"gemma4 int8 on a discrete GPU is the CPU's", "gemma4", "", "int8", false, 0},
		{"gemma4, -vision-device cpu", "gemma4", "cpu", "", false, 0},
		{"mistral3, a CPU-only tower", "mistral3", "", "", false, 0},
		{"voxtral, an audio encoder", "voxtral", "", "", false, 0},
		{"qwen3_asr, an audio encoder", "qwen3_asr", "", "", false, 0},
	} {
		stubWebGPU(t, c.unified, 1<<40)
		webgpuTowerRegistered = func(mt string) bool { return registered[mt] }
		cfg := config{visionPath: tower(c.mt), visionDevice: c.device, visionQuant: c.quant, models: modelFlag{{path: "m"}}}
		cfg.load.Backend = "webgpu"
		out := capture(func() { _ = (&server{}).loadVisionTower(cfg) })
		if n := strings.Count(out, "note: under --backend webgpu on a discrete GPU"); n != c.want {
			t.Errorf("%s: the note was printed %d times, want %d: %q", c.name, n, c.want, out)
		}
	}
	// a text-only load (no -vision, no tower to find) prints nothing
	stubWebGPU(t, false, 0)
	webgpuTowerRegistered = func(mt string) bool { return registered[mt] }
	cfg := config{}
	cfg.load.Backend = "webgpu"
	if out := capture(func() { _ = (&server{}).loadVisionTower(cfg) }); strings.Contains(out, "note:") {
		t.Errorf("a load with no vision tower printed %q", out)
	}
	// a model with a tower directory beside it prints the note once, found by auto-discovery as well as by -vision
	cfg.visionPath = gemma3RealLayoutTiny(t)
	cfg.models = modelFlag{{path: "m"}}
	if out := capture(func() { _ = (&server{}).loadVisionTower(cfg) }); strings.Count(out, "note: under --backend webgpu on a discrete GPU") != 1 {
		t.Errorf("a Gemma 3 tower kept on the CPU printed %q, want the note once", out)
	}
}

// The route feeds the family planners, which refuse under -require-backend a tower they cannot put on the device: an int8 tower
// other than Gemma 3's must still reach them as webgpu (so the refusal stays), and a discrete GPU's CPU default must reach them as
// cpu (so -require-backend does not refuse a choice the user did not make), whether or not this binary has a WebGPU tower.
func TestRouteVisionTower_requireBackendReachesTheFamilyPlanner(t *testing.T) {
	for _, c := range []struct {
		name       string
		unified    bool
		quant      string
		registered bool
		wantErr    bool
	}{
		{"discrete, int8: the planner refuses the CPU's int8 tower", false, "int8", true, true},
		{"unified, int8: the planner refuses", true, "int8", true, true},
		{"discrete, default: the CPU by default, nothing to refuse", false, "", true, false},
		{"discrete, default, no WebGPU tower in this binary: still the CPU, nothing to refuse", false, "", false, false},
		{"unified, default: the device tower is planned", true, "", true, false},
	} {
		stubWebGPU(t, c.unified, 1<<40)
		webgpuTowerRegistered = func(string) bool { return c.registered }
		cfg := config{visionQuant: c.quant, requireBE: true}
		cfg.load.Backend = "webgpu"
		r := cfg.routeVisionTower("qwen3_vl", false)
		_, err := planGridTower("Qwen3-VL", []string{"webgpu"}, towerInt8("qwen3_vl", cfg.visionQuant, r.Backend), r.Backend, cfg.requireBE)
		if (err != nil) != c.wantErr {
			t.Errorf("%s: planGridTower under -require-backend = %v, want an error: %v", c.name, err, c.wantErr)
		}
	}
}
