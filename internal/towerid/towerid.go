// Package towerid fingerprints every vision and audio tower's output on its committed tiny fixture, on the CPU or on a
// device backend reached through the registries production uses, so a change that must not move a tower's numerics
// (Track AX's import switch, docs/tasks/task-subsystems-2026-10.md §7.8) can be checked bit for bit.
//
// Contract. Run(root, backend) loads each family from root's fixtures with float32 weights, feeds it fixed synthetic
// inputs, and returns one Case per (family, input): the SHA-256 of the output's float32 bits and whether the device path
// attached. backend "cpu" runs every family on the CPU; a device name ("metal", "cuda", "webgpu") runs only the families
// that backend declares (multimodal.DeclaredTowers, and multimodal.WhisperDevices for the Whisper encoders), and a
// declared family whose device path does not attach is an error, never a CPU fallback.
//
// Check is the test every backend module calls: two runs must agree, and out (when not empty) receives the record. The
// callers pass GOINFER_TOWER_ID_OUT, read in their test files: this package reads no environment.
//
// Test it: go test ./internal/towerid/ (the CPU record, seconds).
package towerid

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Case is one tower output's fingerprint.
type Case struct {
	Family string `json:"family"`
	Input  string `json:"input"`
	Device bool   `json:"device"` // the device path attached; false on the CPU
	Values int    `json:"values"`
	SHA256 string `json:"sha256"`
}

// Record is one backend's fingerprints on one machine.
type Record struct {
	Schema  int    `json:"schema"`
	Backend string `json:"backend"`
	GOOS    string `json:"goos"`
	GOARCH  string `json:"goarch"`
	Cases   []Case `json:"cases"`
}

// The Whisper encoders have their own device registry (multimodal.UseWhisperDevice), outside the tower support table.
const (
	FamilyWhisper = "whisper"
	FamilyVoxtral = "voxtral"
	FamilyQwenASR = "qwen3-asr"
)

type runner struct {
	root, backend string
	cases         []Case
}

func (r *runner) device(family string) bool {
	if r.backend == "cpu" {
		return false
	}
	if family == FamilyWhisper || family == FamilyVoxtral {
		return slices.Contains(multimodal.WhisperDevices(), r.backend)
	}
	return multimodal.DeclaredTower(r.backend, family)
}

func (r *runner) add(family, input string, device bool, outs ...[]float32) {
	h, n := sha256.New(), 0
	var b [4]byte
	for _, o := range outs {
		for _, v := range o {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
		n += len(o)
	}
	r.cases = append(r.cases, Case{Family: family, Input: input, Device: device, Values: n, SHA256: hex.EncodeToString(h.Sum(nil))})
}

func normal(seed uint64, n int, scale, shift float64) []float32 {
	rng := rand.New(rand.NewPCG(seed, 1))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()*scale + shift)
	}
	return v
}

func uniform(seed uint64, n int) []float32 {
	rng := rand.New(rand.NewPCG(seed, 2))
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

func patches(grid [][3]int, dim int) int {
	n := 0
	for _, g := range grid {
		n += g[0] * g[1] * g[2]
	}
	return n * dim
}

// Run computes backend's record from the fixtures under root (the repository root).
func Run(root, backend string) (Record, error) {
	r := &runner{root: root, backend: backend}
	for _, f := range []func() error{r.siglip, r.qwen25, r.qwen3, r.qwen3Deepstack, r.glmOcr, r.gemma4Vision, r.pixtral,
		r.naflex, r.gemma4Audio, r.whisper, r.voxtral, r.qwenASR} {
		if err := f(); err != nil {
			return Record{}, err
		}
	}
	return Record{Schema: 1, Backend: backend, GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, Cases: r.cases}, nil
}

func (r *runner) path(rel string) string { return filepath.Join(r.root, rel) }

func (r *runner) siglip() error {
	dev := r.device(multimodal.TowerSigLIP)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadEncoder(r.path("testdata/siglip-tiny"), false)
	if err != nil {
		return fmt.Errorf("siglip: %w", err)
	}
	defer enc.Close()
	if dev {
		if err := enc.EnableResident(); err != nil {
			return fmt.Errorf("siglip on %s: %w", r.backend, err)
		}
	}
	c := enc.Cfg
	out, err := enc.Forward(normal(9, c.NumChannels*c.ImageSize*c.ImageSize, 1, 0))
	if err != nil {
		return fmt.Errorf("siglip: %w", err)
	}
	r.add(multimodal.TowerSigLIP, "normal-9", dev, out)
	return nil
}

func (r *runner) qwen25() error {
	dev := r.device(multimodal.TowerQwen25VL)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadQwenVisionEncoder(r.path("testdata/qwen25vl-tiny"), false)
	if err != nil {
		return fmt.Errorf("qwen2.5-vl: %w", err)
	}
	defer enc.Close()
	if dev {
		if err := enc.EnableResident(); err != nil {
			return fmt.Errorf("qwen2.5-vl on %s: %w", r.backend, err)
		}
		if !enc.ResidentEnabled() {
			return fmt.Errorf("qwen2.5-vl on %s: EnableResident returned no error but the tower is not resident", r.backend)
		}
	}
	c := enc.Cfg
	dim := c.InChans * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	for i, g := range [][][3]int{{{1, 8, 8}}, {{1, 12, 16}}} {
		out, err := enc.Forward(normal(10+uint64(i), patches(g, dim), 1, 0), g)
		if err != nil {
			return fmt.Errorf("qwen2.5-vl %v: %w", g, err)
		}
		r.add(multimodal.TowerQwen25VL, fmt.Sprint(g), dev, out)
	}
	return nil
}

// gridGrids reach one and two images, square and not, and the window sizes the tiny towers have.
var gridGrids = [][][3]int{{{1, 4, 4}}, {{1, 6, 8}}, {{1, 10, 6}}, {{1, 4, 4}, {1, 6, 4}}}

func (r *runner) qwen3() error {
	dev := r.device(multimodal.TowerQwen3)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadQwen3VisionEncoder(r.path("testdata/qwen35vl-tiny"), false)
	if err != nil {
		return fmt.Errorf("qwen3: %w", err)
	}
	defer enc.Close()
	var acc multimodal.GridTowerAccelerator
	if dev {
		if acc, err = multimodal.NewQwen3Tower(r.backend, enc); err != nil {
			return fmt.Errorf("qwen3 on %s: %w", r.backend, err)
		}
		defer acc.Close()
	}
	c := enc.Cfg
	dim := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	for i, g := range gridGrids {
		out, err := multimodal.Qwen3TowerFeatures(enc, acc, normal(20+uint64(i), patches(g, dim), 1, 0), g)
		if err != nil {
			return fmt.Errorf("qwen3 %v: %w", g, err)
		}
		r.add(multimodal.TowerQwen3, fmt.Sprint(g), dev, out)
	}
	return nil
}

func (r *runner) qwen3Deepstack() error {
	dev := r.device(multimodal.TowerQwen3)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadQwen3VisionEncoder(r.path("testdata/qwen3vl-vision-tiny"), false)
	if err != nil {
		return fmt.Errorf("qwen3 deepstack: %w", err)
	}
	defer enc.Close()
	var acc multimodal.GridTowerAccelerator
	if dev {
		if acc, err = multimodal.NewQwen3Tower(r.backend, enc); err != nil {
			return fmt.Errorf("qwen3 deepstack on %s: %w", r.backend, err)
		}
		defer acc.Close()
	}
	c := enc.Cfg
	dim := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	for i, g := range [][][3]int{{{1, 4, 6}}, {{1, 6, 4}, {1, 4, 4}}} {
		merged, deep, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, normal(30+uint64(i), patches(g, dim), 1, 0), g)
		if err != nil {
			return fmt.Errorf("qwen3 deepstack %v: %w", g, err)
		}
		if len(deep) == 0 {
			return fmt.Errorf("qwen3 deepstack %v: no DeepStack outputs (the fixture is not a DeepStack tower)", g)
		}
		r.add(multimodal.TowerQwen3+"-deepstack", fmt.Sprint(g), dev, append([][]float32{merged}, deep...)...)
	}
	return nil
}

func (r *runner) glmOcr() error {
	dev := r.device(multimodal.TowerGlmOcr)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadGlmOcrVisionEncoder(r.path("testdata/glm-ocr-vision-tiny"), false)
	if err != nil {
		return fmt.Errorf("glm-ocr: %w", err)
	}
	defer enc.Close()
	var acc multimodal.GridTowerAccelerator
	if dev {
		if acc, err = multimodal.NewGlmOcrTower(r.backend, enc); err != nil {
			return fmt.Errorf("glm-ocr on %s: %w", r.backend, err)
		}
		defer acc.Close()
	}
	c := enc.Cfg
	dim := c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize
	for i, g := range gridGrids {
		out, err := multimodal.GlmOcrTowerFeatures(enc, acc, normal(40+uint64(i), patches(g, dim), 1, 0), g)
		if err != nil {
			return fmt.Errorf("glm-ocr %v: %w", g, err)
		}
		r.add(multimodal.TowerGlmOcr, fmt.Sprint(g), dev, out)
	}
	return nil
}

func (r *runner) gemma4Vision() error {
	dev := r.device(multimodal.TowerGemma4)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := vision.LoadGemma4Encoder(r.path("testdata/gemma4-vision-tiny"), false)
	if err != nil {
		return fmt.Errorf("gemma4-vision: %w", err)
	}
	defer enc.Close()
	var acc multimodal.Gemma4TowerAccelerator
	if dev {
		if acc, err = multimodal.NewGemma4Tower(r.backend, enc); err != nil {
			return fmt.Errorf("gemma4-vision on %s: %w", r.backend, err)
		}
		defer acc.Close()
	}
	k, dim := enc.Cfg.PoolingKernelSize, 3*enc.Cfg.PatchSize*enc.Cfg.PatchSize
	for i, wh := range [][2]int{{2, 3}, {3, 2}} {
		var pos [][2]int
		for y := range wh[1] * k {
			for x := range wh[0] * k {
				pos = append(pos, [2]int{x, y})
			}
		}
		out, err := multimodal.Gemma4TowerFeatures(enc, acc, uniform(50+uint64(i), len(pos)*dim), pos)
		if err != nil {
			return fmt.Errorf("gemma4-vision %v: %w", wh, err)
		}
		r.add(multimodal.TowerGemma4, fmt.Sprintf("%dx%d pooled", wh[0], wh[1]), dev, out)
	}
	return nil
}

func (r *runner) pixtral() error {
	if r.backend != "cpu" {
		return nil // no backend declares it
	}
	enc, err := vision.LoadPixtralVisionEncoder(r.path("internal/vision/testdata/pixtral-vision-tiny"), false)
	if err != nil {
		return fmt.Errorf("pixtral: %w", err)
	}
	defer enc.Close()
	p := enc.Cfg.PatchSize
	h, w := 3*p, 2*p
	px, grid, err := vision.PixtralPatchify(normal(60, 3*h*w, 1, 0), 3, h, w, p)
	if err != nil {
		return fmt.Errorf("pixtral: %w", err)
	}
	out, err := enc.Forward(px, [][2]int{grid})
	if err != nil {
		return fmt.Errorf("pixtral: %w", err)
	}
	r.add(multimodal.TowerPixtral, fmt.Sprintf("%dx%d", h, w), false, out)
	return nil
}

func (r *runner) naflex() error {
	if r.backend != "cpu" {
		return nil // no backend declares it
	}
	enc, err := vision.LoadSiglip2NaFlexEncoder(r.path("internal/vision/testdata/siglip2-naflex-tiny"), false)
	if err != nil {
		return fmt.Errorf("siglip2-naflex: %w", err)
	}
	defer enc.Close()
	p := enc.Cfg.PatchSize
	h, w := 2*p, 3*p
	px, grid, err := vision.PatchifyNaFlex(normal(61, 3*h*w, 1, 0), 3, h, w, p)
	if err != nil {
		return fmt.Errorf("siglip2-naflex: %w", err)
	}
	out, err := enc.Forward(px, grid)
	if err != nil {
		return fmt.Errorf("siglip2-naflex: %w", err)
	}
	r.add(multimodal.TowerSiglip2NaFlex, fmt.Sprintf("%dx%d", h, w), false, out)
	return nil
}

func (r *runner) gemma4Audio() error {
	dev := r.device(multimodal.TowerGemma4Audio)
	if r.backend != "cpu" && !dev {
		return nil
	}
	enc, err := audio.LoadGemma4AudioEncoder(r.path("testdata/gemma4-audio-tiny"))
	if err != nil {
		return fmt.Errorf("gemma4-audio: %w", err)
	}
	var acc embeddinggemma2.AudioAccelerator
	if dev {
		if acc, err = embeddinggemma2.NewAudioAccelerator(r.backend, enc); err != nil {
			return fmt.Errorf("gemma4-audio on %s: %w", r.backend, err)
		}
		defer acc.Close()
	}
	const mels = 128 // Gemma 4's log-mel width ([T, 128])
	for i, T := range []int{37, 64} {
		feats := normal(70+uint64(i), T*mels, 1, 0)
		var out []float32
		if acc == nil {
			out, err = enc.Forward(feats, T)
		} else {
			var h, hb []float32
			var n int
			if h, n, err = enc.Subsample(feats, T); err == nil {
				if hb, err = acc.Blocks(h, n); err == nil {
					out, err = enc.FinishBlocks(hb, n)
				}
			}
		}
		if err != nil {
			return fmt.Errorf("gemma4-audio T=%d: %w", T, err)
		}
		r.add(multimodal.TowerGemma4Audio, fmt.Sprintf("T=%d", T), dev, out)
	}
	return nil
}

// whisperForward runs a Whisper-shaped encoder on the CPU, or on backend's Whisper device.
func (r *runner) whisperForward(family string, enc *audio.WhisperEncoder, seed uint64) error {
	dev := r.device(family)
	if dev {
		d, err := multimodal.UseWhisperDevice(r.backend, enc)
		if err != nil {
			return fmt.Errorf("%s on %s: %w", family, r.backend, err)
		}
		defer d.Close()
	}
	out, err := enc.Forward(normal(seed, enc.Cfg.MelBins*audio.WhisperFrames, 0.6, -0.3))
	if err != nil {
		return fmt.Errorf("%s: %w", family, err)
	}
	r.add(family, fmt.Sprintf("normal-%d", seed), dev, out)
	return nil
}

func (r *runner) whisper() error {
	if r.backend != "cpu" && !r.device(FamilyWhisper) {
		return nil
	}
	enc, err := audio.LoadWhisperEncoder(r.path("testdata/whisper-tiny-rand"))
	if err != nil {
		return fmt.Errorf("whisper: %w", err)
	}
	defer enc.Close()
	return r.whisperForward(FamilyWhisper, enc, 80)
}

func (r *runner) voxtral() error {
	if r.backend != "cpu" && !r.device(FamilyVoxtral) {
		return nil
	}
	vox, err := audio.LoadVoxtralAudio(r.path("testdata/voxtral-tiny"))
	if err != nil {
		return fmt.Errorf("voxtral: %w", err)
	}
	defer vox.Enc.Close()
	return r.whisperForward(FamilyVoxtral, vox.Enc, 81)
}

func (r *runner) qwenASR() error {
	if r.backend != "cpu" {
		return nil // no device path
	}
	enc, err := audio.LoadQwenASREncoder(r.path("testdata/qwen3asr-tiny"))
	if err != nil {
		return fmt.Errorf("qwen3-asr: %w", err)
	}
	defer enc.Close()
	const T, valid = 200, 150 // feats [128][T], T a multiple of 100
	out, n, err := enc.Forward(normal(90, 128*T, 1, 0), T, valid)
	if err != nil {
		return fmt.Errorf("qwen3-asr: %w", err)
	}
	r.add(FamilyQwenASR, fmt.Sprintf("T=%d valid=%d n=%d", T, valid, n), false, out)
	return nil
}

// Check runs backend's record twice and fails unless both runs agree case by case; a non-empty out receives the record
// as JSON. A device backend's caller skips first when the device is absent.
func Check(t testing.TB, root, backend, out string) Record {
	t.Helper()
	a, err := Run(root, backend)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(root, backend)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Cases) == 0 {
		t.Fatalf("%s: no tower ran", backend)
	}
	if !slices.Equal(a.Cases, b.Cases) {
		for i := range min(len(a.Cases), len(b.Cases)) {
			if a.Cases[i] != b.Cases[i] {
				t.Errorf("%s %s %s: run 1 %s, run 2 %s", backend, a.Cases[i].Family, a.Cases[i].Input, a.Cases[i].SHA256, b.Cases[i].SHA256)
			}
		}
		t.Fatalf("%s: two runs disagree, so the record cannot grade identity", backend)
	}
	for _, c := range a.Cases {
		t.Logf("%-18s %-28s device=%-5v values=%-7d %s", c.Family, c.Input, c.Device, c.Values, c.SHA256)
	}
	if out != "" {
		j, err := json.MarshalIndent(a, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(out, append(j, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return a
}
