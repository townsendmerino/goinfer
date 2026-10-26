//go:build darwin

package visionmetal_test

// crossover_test.go is the ViT throughput slice of docs/BENCH-gpu.md: a real-sized SigLIP
// tower run on the CPU vs the Metal resident encoder, emitting bench.Record rows the same
// report generator renders. Unlike ANN (an N×batch crossover), a ViT is one image at a fixed
// patch count, so the measurement is throughput-at-a-real-size — the tiny parity fixture
// (hidden 32) says nothing about throughput, so this needs a real tower.
//
// It uses a RANDOM real-sized checkpoint (scripts/oracle/gen_siglip_bench.py): throughput does not
// depend on weight values, and parity is gated GPU-vs-CPU on the SAME tower (int8), so random
// weights are exactly right and need no download. Gated on AIKIT_GPU_BENCH + the checkpoint +
// a device, so a normal `go test` stays green.

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "github.com/townsendmerino/goinfer/metal/internal/visionmetal" // registers the Metal ResidentEncoder

	"github.com/townsendmerino/aikit/bench"
	"github.com/townsendmerino/goinfer/internal/vision"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// TestMetalViTThroughput times real SigLIP forwards (one or more tower sizes), CPU vs Metal
// resident, and records each — the size sweep shows the GPU win growing as bigger ops amortize
// the per-op dispatch floor.
func TestMetalViTThroughput(t *testing.T) {
	if os.Getenv("AIKIT_GPU_BENCH") == "" {
		t.Skip("periodic GPU pass — set AIKIT_GPU_BENCH=1 to run (docs/BENCH-gpu.md)")
	}
	models := strings.Split(envStr("AIKIT_BENCH_MODELS", "../../../internal/vision/testdata/siglip-bench,../../testdata/siglip-bench-l"), ",")
	recordsPath := envStr("AIKIT_BENCH_RECORDS", "../../docs/bench-records/vit-metal.jsonl")
	_ = os.MkdirAll(filepath.Dir(recordsPath), 0o755)
	_ = os.Remove(recordsPath) // fresh; both sizes append below
	ran := 0
	for _, modelDir := range models {
		modelDir = strings.TrimSpace(modelDir)
		if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
			t.Logf("skip %s (run scripts/oracle/gen_siglip_bench.py)", modelDir)
			continue
		}
		if oneViT(t, modelDir, recordsPath) {
			ran++
		}
	}
	if ran == 0 {
		t.Skip("no SigLIP bench towers present — run scripts/oracle/gen_siglip_bench.py")
	}
	t.Logf("records → %s", recordsPath)
}

// oneViT benchmarks a single tower and appends its cpu/metal records. Returns false if the
// device is unavailable (so the caller can skip cleanly).
func oneViT(t *testing.T, modelDir, recordsPath string) bool {
	// int8: the resident path needs int8 matmul weights, and gating GPU against a CPU tower
	// loaded the SAME way makes the device path the only variable.
	enc, err := vision.LoadEncoder(modelDir, true)
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	defer enc.Close()
	c := enc.Cfg
	np := (c.ImageSize / c.PatchSize) * (c.ImageSize / c.PatchSize)

	// deterministic synthetic image [C*H*W]
	px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
	var s uint32 = 7
	for i := range px {
		s = s*1664525 + 1013904223
		px[i] = float32(int32(s>>8)%2000-1000) / 1000.0
	}

	timeForward := func() (out []float32, best time.Duration) {
		for range 2 { // warm
			o, err := enc.Forward(px)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			out = o
		}
		best = time.Hour
		for range 8 {
			t0 := time.Now()
			o, err := enc.Forward(px)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			out = o
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return out, best
	}

	cpuOut, cpuT := timeForward()
	if err := enc.EnableResident(); err != nil {
		t.Logf("EnableResident: %v (no Metal device?)", err)
		return false
	}
	gpuOut, gpuT := timeForward()

	if len(cpuOut) != len(gpuOut) {
		t.Fatalf("output len %d != %d", len(gpuOut), len(cpuOut))
	}
	// Metal's layernorm/softmax reduce in f32 (MSL has no double, where the CPU tower uses
	// float64), so a DEEP tower accumulates a little per-layer drift the tiny 2-layer parity
	// fixture (cosine 1.0) never shows — ~1e-4 over 12 layers here, retrieval-identical. The
	// gate rejects a real divergence (a bug drops cosine far more) while admitting that drift.
	cos := cosine(cpuOut, gpuOut)
	parityOK := cos >= 1-1e-3
	if !parityOK {
		t.Errorf("PARITY: SigLIP Metal≢CPU cosine %.6f below 1-1e-3 — the resident tower diverged", cos)
	}

	cpuImgS := 1.0 / cpuT.Seconds()
	gpuImgS := 1.0 / gpuT.Seconds()
	sp := gpuImgS / cpuImgS

	machine := envStr("AIKIT_BENCH_MACHINE", "apple")
	gpuName := "Apple GPU (Metal)"
	if dev, e := gpu.CreateSystemDefaultDevice(); e == nil {
		gpuName = dev.Name()
		dev.ReleaseObjects()
	}
	meta := bench.CaptureMeta()
	if meta.AikitCommit == "" {
		meta.AikitCommit = envStr("AIKIT_BENCH_COMMIT", "unknown")
	}
	meta.Iters = 8
	cpuDev, gpuDev := bench.HostDevice(), bench.HostDevice()
	cpuDev.Machine, cpuDev.Chip = machine, envStr("AIKIT_BENCH_CHIP", "")
	gpuDev.Machine, gpuDev.GPU, gpuDev.SMorFamily = machine, gpuName, "Apple"
	shape := bench.Shape{Patches: np, Dim: c.HiddenSize}
	cr := cos // record cosine as the quality (1.0 = matches CPU)

	err = bench.AppendRecords(recordsPath,
		bench.Record{
			Workload: "vision.SigLIP.Forward", Backend: "cpu-simd", Precision: "int8",
			Device: cpuDev, Shape: shape,
			Timing:     bench.Timing{Compute: float64(cpuT.Nanoseconds()) / 1e6, Wall: float64(cpuT.Nanoseconds()) / 1e6},
			Throughput: cpuImgS, ThroughputUnit: "images/s",
			Quality: bench.Quality{ParityOK: true}, Meta: meta,
		},
		bench.Record{
			Workload: "vision.SigLIP.Forward", Backend: "metal", Precision: "int8",
			Device: gpuDev, Shape: shape,
			Timing:     bench.Timing{Compute: float64(gpuT.Nanoseconds()) / 1e6, Wall: float64(gpuT.Nanoseconds()) / 1e6},
			Throughput: gpuImgS, ThroughputUnit: "images/s",
			Quality:      bench.Quality{RecallAtK: &cr, ParityOK: parityOK, MaxDeltaVsCPU: 1 - cos},
			SpeedupVsCPU: &sp, Meta: meta,
		},
	)
	if err != nil {
		t.Fatalf("append records: %v", err)
	}
	t.Logf("SigLIP hidden=%d layers=%d patches=%d: CPU %v (%.1f img/s), Metal %v (%.1f img/s), %.2f×, cosine %.6f",
		c.HiddenSize, c.NumHiddenLayers, np, cpuT, cpuImgS, gpuT, gpuImgS, sp, cos)
	return true
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
}
