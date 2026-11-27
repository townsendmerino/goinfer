package towers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

type blockReleaser interface{ ReleaseBlocks() error }

// ReleaseTowerBlocks drops a device-attached tower's host blocks so host RAM is freed while the GPU tower serves images.
var ReleaseTowerBlocks = func(family string, enc any) {
	r, ok := enc.(blockReleaser)
	if !ok {
		return
	}
	if err := r.ReleaseBlocks(); err != nil {
		fmt.Fprintf(os.Stderr, "vision: the %s tower keeps its host blocks: %v\n", family, err)
	}
}

// ResidentTower is the device-attach contract used by QwenDeviceForward.
type ResidentTower interface {
	ResidentEnabled() bool
	Close()
}

// QwenDeviceForward runs forward under panic recovery and falls back to CPU on device OOM unless require is set.
func QwenDeviceForward(t ResidentTower, require bool, forward func() ([]float32, error)) ([]float32, error) {
	out, err := towerfit.RecoverDeviceTower("Qwen2.5-VL", forward)
	if !t.ResidentEnabled() || !mem.DeviceOutOfMemory(err) {
		return out, err
	}
	if require {
		return nil, towerfit.RefuseCPUFallback("Qwen2.5-VL", err)
	}
	fmt.Fprintf(os.Stderr, "vision: the Qwen2.5-VL tower ran out of device memory (%v); it runs on the CPU from now on\n", err)
	t.Close()
	return towerfit.RecoverDeviceTower("Qwen2.5-VL", forward)
}

// GridTowerPlan records the target accelerator device and banner description for a lazy grid vision tower.
type GridTowerPlan struct {
	Device, Where string
	Require       bool
}

// DeviceTowerName returns the display name for a backend that can host a device vision tower ("Metal", "CUDA").
func DeviceTowerName(backend string) (string, bool) {
	switch backend {
	case "metal":
		return "Metal", true
	case "cuda":
		return "CUDA", true
	}
	return "", false
}

// PlanGridTower decides at startup where a Qwen3.5+ or GLM-OCR grid tower will run.
func PlanGridTower(family string, registered []string, int8Tower bool, backend string, require bool) (GridTowerPlan, error) {
	name, ok := DeviceTowerName(backend)
	if backend == "webgpu" {
		name, ok = "WebGPU", true
	}
	if !ok {
		return GridTowerPlan{Where: "CPU"}, nil
	}
	if int8Tower {
		if require {
			return GridTowerPlan{}, fmt.Errorf("-require-backend: the %s %s tower is float32; -vision-quant int8 keeps it on the CPU", family, name)
		}
		return GridTowerPlan{Where: "CPU (-vision-quant int8; the " + name + " tower is float32)"}, nil
	}
	if !slices.Contains(registered, backend) {
		if require {
			return GridTowerPlan{}, fmt.Errorf("-require-backend: this binary has no %s %s tower", name, family)
		}
		return GridTowerPlan{Where: "CPU (no " + name + " tower in this binary)"}, nil
	}
	return GridTowerPlan{Device: backend, Where: name, Require: require}, nil
}

// Started settles p once the tower's weights load, falling back to CPU unless Require is set.
func (p *GridTowerPlan) Started(family string, acc multimodal.GridTowerAccelerator, err error) (multimodal.GridTowerAccelerator, error) {
	if err == nil {
		return acc, nil
	}
	if p.Require {
		return nil, fmt.Errorf("-require-backend: the %s tower could not start on %s: %w", family, p.Device, err)
	}
	fmt.Fprintf(os.Stderr, "vision: the %s tower runs on the CPU: %s declined it: %v\n", family, p.Device, err)
	p.Where = "CPU (" + p.Device + " declined)"
	return nil, nil
}

// ChooseGemma4Tower selects and starts Gemma 4's vision tower accelerator or falls back to CPU.
func ChooseGemma4Tower(enc *vision.Gemma4Encoder, int8Tower bool, backend string, require bool) (multimodal.Gemma4TowerAccelerator, string, error) {
	name, ok := DeviceTowerName(backend)
	if backend == "webgpu" {
		name, ok = "WebGPU", true
	}
	if !ok {
		return nil, "CPU", nil
	}
	if int8Tower {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 %s tower is float32; -vision-quant int8 keeps it on the CPU", name)
		}
		return nil, "CPU (-vision-quant int8; the " + name + " tower is float32)", nil
	}
	if !slices.Contains(multimodal.Gemma4Towers(), backend) {
		if require {
			return nil, "", fmt.Errorf("-require-backend: this binary has no %s Gemma 4 tower", name)
		}
		return nil, "CPU (no " + name + " tower in this binary)", nil
	}
	t, err := multimodal.NewGemma4Tower(backend, enc)
	if err != nil {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 tower could not start on %s: %w", name, err)
		}
		return nil, "CPU (" + name + " declined: " + err.Error() + ")", nil
	}
	return t, name, nil
}

// QwenTowerPlacement places Qwen2.5-VL's vision tower on CUDA/WebGPU or falls back to CPU.
func QwenTowerPlacement(backend string, int8Tower, require bool, attach func() error, warn io.Writer) (string, error) {
	name := "CUDA"
	switch backend {
	case "cuda":
	case "webgpu":
		name = "WebGPU"
	default:
		return "CPU", nil
	}
	if int8Tower {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL %s tower is float32; -vision-quant int8 keeps it on the CPU", name)
		}
		return "CPU (-vision-quant int8; the " + name + " tower is float32)", nil
	}
	if err := attach(); err != nil {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL tower could not start on %s: %w", backend, err)
		}
		fmt.Fprintf(warn, "vision: the Qwen2.5-VL tower runs on the CPU: %s declined it: %v\n", backend, err)
		return "CPU (" + backend + " declined)", nil
	}
	return name, nil
}

// AttachableTower is the encoder contract required by AttachGemma3Tower.
type AttachableTower interface {
	EnableResident() error
	Close()
}

// AttachGemma3Tower loads Gemma 3's SigLIP encoder and attaches its device-resident form, falling back from default f32 to int8 when needed.
func AttachGemma3Tower[E AttachableTower](load func(int8 bool) (E, error), startInt8, fallback bool, backend string, require bool, warn io.Writer) (enc E, int8, resident bool, err error) {
	const label = "Gemma 3 SigLIP"
	enc, err = load(startInt8)
	if err != nil {
		return enc, startInt8, false, err
	}
	if !fallback {
		resident, err = towerfit.AttachResidentTower(label, enc, backend, require, warn)
		return enc, startInt8, resident, err
	}
	var why bytes.Buffer
	if ok, _ := towerfit.AttachResidentTower(label, enc, backend, false, &why); ok {
		return enc, startInt8, true, nil
	}
	enc.Close()
	fmt.Fprintf(warn, "note: the float32 Gemma 3 vision tower did not fit on the device (%s); using the int8 device tower instead (-vision-quant f32 asks for float32 and does not fall back)\n",
		strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(why.String()), "warning:")))
	enc, err = load(true)
	if err != nil {
		return enc, true, false, err
	}
	resident, err = towerfit.AttachResidentTower(label, enc, backend, require, warn)
	return enc, true, resident, err
}

// SiglipLoader returns a precision-parameterized Gemma 3 SigLIP loader for dir.
func SiglipLoader(dir string, head bool) func(int8 bool) (*vision.Encoder, error) {
	return func(i8 bool) (*vision.Encoder, error) {
		load := vision.LoadEncoder
		if head {
			load = vision.LoadEncoderHead
		}
		e, err := load(dir, i8)
		if err != nil {
			return nil, fmt.Errorf("load vision encoder (%s): %w", dir, err)
		}
		return e, nil
	}
}

// Gemma3ResidentWeightsEstimate estimates CUDA resident decoder weight bytes from checkpoint files on disk.
func Gemma3ResidentWeightsEstimate(path string) (int64, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	if !fi.IsDir() {
		low := strings.ToLower(path)
		if strings.HasSuffix(low, ".gguf") || strings.HasSuffix(low, ".giw") {
			return fi.Size(), fi.Size() > 0
		}
		return 0, false
	}
	files, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	var total int64
	for _, f := range files {
		if st, err := os.Stat(f); err == nil {
			total += st.Size()
		}
	}
	return total * 27 / 100, total > 0
}

// HasAudioConfig reports whether dir/config.json declares an audio_config.
func HasAudioConfig(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Audio json.RawMessage `json:"audio_config"`
	}
	return json.Unmarshal(raw, &c) == nil && len(c.Audio) > 0 && string(c.Audio) != "null"
}

// VisionPathError rejects non-GGUF file paths passed to -vision.
func VisionPathError(dir string) error {
	fi, err := os.Stat(dir)
	if err != nil || fi.IsDir() {
		return nil
	}
	if strings.HasSuffix(strings.ToLower(dir), ".gguf") {
		return nil
	}
	return fmt.Errorf("-vision %s is a file; -vision takes a directory with a vision tower (config.json and safetensors)", dir)
}
