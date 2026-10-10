package serveapp

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/vision"
)

// fakeSiglipResident records what a resident SigLIP factory was handed.
type fakeSiglipResident struct{}

func (fakeSiglipResident) ForwardPatches([]float32) ([]float32, error) {
	return nil, fmt.Errorf("fake")
}
func (fakeSiglipResident) Close() {}

// TestLoadVisionTower_siglipHonoursVisionDevice: -vision-device cpu keeps Gemma 3's SigLIP tower on the CPU, f32, under a device backend. It must not
// follow --backend alone: under cuda (and webgpu) the tower would load int8 and the device tower attach whatever the flag said, so G-S3c's "tower on the
// CPU in every arm" would not hold on CUDA.
func TestLoadVisionTower_siglipHonoursVisionDevice(t *testing.T) {
	var calls, int8Calls int
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		calls++
		if _, err := e.GPUWeights(); err == nil {
			int8Calls++
		}
		return fakeSiglipResident{}, nil
	})
	t.Cleanup(func() { vision.RegisterResident(nil) })
	dir := gemma3RealLayoutTiny(t)
	for _, tc := range []struct {
		device, quant     string
		wantCalls, wantI8 int
	}{{"cpu", "", 0, 0}, {"auto", "", 1, 0}, {"auto", "int8", 1, 1}} { // the default tower under cuda is float32 since 2026-10-08; an explicit int8 is the int8 one
		calls, int8Calls = 0, 0
		cfg := config{visionPath: dir, visionDevice: tc.device, visionQuant: tc.quant, models: modelFlag{{path: "m"}}}
		cfg.load.Backend = "cuda"
		if err := (&server{}).loadVisionTower(cfg); err != nil {
			t.Fatalf("-vision-device %s: %v", tc.device, err)
		}
		if calls != tc.wantCalls || int8Calls != tc.wantI8 {
			t.Errorf("-vision-device %s -vision-quant %q under cuda: resident factory called %d times (%d with an int8 tower), want %d (%d)",
				tc.device, tc.quant, calls, int8Calls, tc.wantCalls, tc.wantI8)
		}
	}
}

// gemma3RealLayoutTiny is testdata/gemma3-vl-tiny with its tower's tensors under the real checkpoint's
// "vision_tower.vision_model." prefix (the fixture uses the shorter "vision_tower.", which aikit's LoadEncoder does not
// look for). Only the safetensors header changes: the offsets are relative to the data, which is copied as is.
func gemma3RealLayoutTiny(t *testing.T) string {
	t.Helper()
	src := "../../testdata/gemma3-vl-tiny"
	raw, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Skipf("no %s: %v", src, err)
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
		t.Fatal(err)
	}
	out := map[string]json.RawMessage{}
	for k, v := range hdr {
		if rest, ok := strings.CutPrefix(k, "vision_tower."); ok {
			k = "vision_tower.vision_model." + rest
		}
		out[k] = v
	}
	h, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	buf := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	buf = append(append(buf, h...), raw[8+n:]...)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buf, 0o644); err != nil {
		t.Fatal(err)
	}
	cfgJSON, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgJSON, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// fallingSiglipResident stands in for a card that cannot hold the float32 tower: the factory refuses a float32 encoder (GPUWeights errs for one) and accepts the int8 one.
func TestLoadVisionTower_gemma3FloatDefaultFallsBackToInt8(t *testing.T) {
	var f32Tries, int8Tries int
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		if _, err := e.GPUWeights(); err == nil {
			int8Tries++
			return fakeSiglipResident{}, nil
		}
		f32Tries++
		return nil, errors.New("cuda: device allocation failed: CUDA_ERROR_OUT_OF_MEMORY")
	})
	t.Cleanup(func() { vision.RegisterResident(nil) })
	dir := gemma3RealLayoutTiny(t)
	for _, tc := range []struct {
		quant                   string
		wantF32Tries, wantInt8s int
	}{{"", 1, 1}, {"f32", 1, 0}} { // the default falls back to int8; an explicit f32 does not
		f32Tries, int8Tries = 0, 0
		cfg := config{visionPath: dir, visionQuant: tc.quant, models: modelFlag{{path: "m"}}}
		cfg.load.Backend = "cuda"
		if err := (&server{}).loadVisionTower(cfg); err != nil {
			t.Fatalf("-vision-quant %q: %v", tc.quant, err)
		}
		if f32Tries != tc.wantF32Tries || int8Tries != tc.wantInt8s {
			t.Errorf("-vision-quant %q under cuda with no room for float32: float32 attach tried %d times, int8 %d, want %d and %d", tc.quant, f32Tries, int8Tries, tc.wantF32Tries, tc.wantInt8s)
		}
	}
}
