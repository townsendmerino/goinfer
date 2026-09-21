//go:build cuda && goinfer_testhooks

package cuda

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
)

// TestVisionTowerTiming is R8's committed timing driver (docs/tasks/red-october.md R8; the P6 driver was a throwaway).
// Real gemma-3-4b-it SigLIP tower, 896^2 = 4096 patches, 27 layers, int8 both arms. It times the RESIDENT tower's Forward
// (host im2col + upload + 27 layers + download), one discarded warm-up then GOINFER_VISION_REPS (default 5) measured runs,
// heartbeat per run, and reports min/median/max seconds. It can also
//
//   - GOINFER_VISION_SAVE=path : write the resident output (f32 LE) — the "before" arm a later kernel is scored against;
//
//   - GOINFER_VISION_CPU_REF=path : read the CPU int8 reference from path, computing (~40 s) and writing it if absent;
//     and it prints cosine(resident, CPU ref) [the P6 gate's own 0.91-0.96 range] and, if GOINFER_VISION_BASE=path is set,
//     cosine(resident, that saved baseline) with the max |diff|, which is how a change to the kernels is localised.
//
//     GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVisionTowerTiming -v -timeout 30m
func TestVisionTowerTiming(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GEMMA3_4B")
	if dir == "" {
		dir = home + "/models/gemma-3-4b-it"
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", dir, err)
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	reps := 5
	if v := os.Getenv("GOINFER_VISION_REPS"); v != "" {
		fmt.Sscanf(v, "%d", &reps)
	}
	e, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	defer e.Close()
	cfg := e.Cfg
	pixels := make([]float32, cfg.NumChannels*cfg.ImageSize*cfg.ImageSize)
	fa, fb := 0.0091, 0.0037 // pattern A: the P6 gate's own; GOINFER_VISION_PATTERN=b is the second, different-frequency pattern (R8 gate 3)
	if os.Getenv("GOINFER_VISION_PATTERN") == "b" {
		fa, fb = 0.0173, 0.0071
	}
	for i := range pixels {
		pixels[i] = float32(math.Sin(float64(i)*fa))*0.5 + float32(math.Cos(float64(i)*fb))*0.3
	}
	var cpuRef []float32
	if p := os.Getenv("GOINFER_VISION_CPU_REF"); p != "" {
		if b, err := os.ReadFile(p); err == nil {
			cpuRef = f32sFromBytes(b)
		} else {
			t0 := time.Now()
			if cpuRef, err = e.Forward(pixels); err != nil {
				t.Fatalf("CPU Forward: %v", err)
			}
			os.WriteFile(p, f32sToBytes(cpuRef), 0o644)
			fmt.Printf("[vision] CPU int8 reference computed in %s and cached at %s\n", time.Since(t0).Round(time.Second), p)
		}
	}
	if err := e.EnableResident(); err != nil {
		t.Fatalf("EnableResident: %v", err)
	}
	var out []float32
	var secs []float64
	t0 := time.Now()
	for i := 0; i <= reps; i++ {
		s := time.Now()
		if out, err = e.Forward(pixels); err != nil {
			t.Fatalf("resident Forward: %v", err)
		}
		d := time.Since(s).Seconds()
		tag := "measured"
		if i == 0 {
			tag = "warm-up (discarded)"
		} else {
			secs = append(secs, d)
		}
		fmt.Printf("[vision] run %d/%d %s: %.3f s   (elapsed %s)\n", i, reps, tag, d, time.Since(t0).Round(time.Second))
	}
	if len(secs) == 0 { // GOINFER_VISION_REPS=0: one forward only (for ncu)
		return
	}
	sort.Float64s(secs)
	fmt.Printf("=== VISION TOWER (resident CUDA, %d patches, %d layers): min %.3f  median %.3f  max %.3f s over %d runs ===\n",
		cfg.ImageSize/cfg.PatchSize*cfg.ImageSize/cfg.PatchSize, cfg.NumHiddenLayers, secs[0], secs[len(secs)/2], secs[len(secs)-1], len(secs))
	if p := os.Getenv("GOINFER_VISION_SAVE"); p != "" {
		os.WriteFile(p, f32sToBytes(out), 0o644)
	}
	if cpuRef != nil {
		fmt.Printf("cosine(resident, CPU int8 reference) = %.6f\n", cosineF32(cpuRef, out))
	}
	if p := os.Getenv("GOINFER_VISION_BASE"); p != "" {
		if b, err := os.ReadFile(p); err == nil {
			base := f32sFromBytes(b)
			var md float64
			for i := range base {
				md = math.Max(md, math.Abs(float64(base[i]-out[i])))
			}
			fmt.Printf("cosine(resident, saved baseline) = %.8f  max|diff| = %.5f\n", cosineF32(base, out), md)
		}
	}
}

func f32sToBytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func f32sFromBytes(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}
