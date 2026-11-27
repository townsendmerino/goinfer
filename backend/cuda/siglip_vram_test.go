//go:build cuda

package cuda

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

var sigVRAMImages = []string{"gemma3_preprocess_image.png"}

// TestSiglipCUDA_int8VRAM measures what the shipped default's int8 SigLIP tower claims on the device (S18's G-S18c needs the figure to price it): free VRAM before the encoder attaches, after it attaches, and after
// one forward (the peak, since scratch may be allocated lazily). Heavy; a record.
func TestSiglipCUDA_int8VRAM(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	skipIfNoCUDADevice(t)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	q, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	free := func() float64 {
		b, ok := decoder.FreeBytesFor("cuda")
		if !ok {
			t.Skip("no free-VRAM reading")
		}
		return float64(b) / (1 << 20)
	}
	f0 := free()
	if err := q.EnableResident(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(q.Close)
	f1 := free()
	data, err := os.ReadFile("../../testdata/" + sigVRAMImages[0])
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(data, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Forward(pv.Data); err != nil {
		t.Fatal(err)
	}
	f2 := free()
	fmt.Fprintf(os.Stderr, "[S18 int8 tower VRAM] free before %.0f MiB, after attach %.0f MiB (tower holds %.0f MiB), after one forward %.0f MiB (%.0f MiB held in all)\n", f0, f1, f0-f1, f2, f0-f2)
}

func skipIfNoCUDADevice(t *testing.T) {
	t.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	dev.ReleaseObjects()
}

func caseCount(t *testing.T, env string, def, hi int) int {
	t.Helper()
	v := os.Getenv(env)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 || n > hi {
		t.Fatalf("%s=%q: want 1 to %d", env, v, hi)
	}
	return n
}

func audioRandFeats(mels int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	f := make([]float32, mels*audio.WhisperFrames)
	for i := range f {
		f[i] = float32(rng.NormFloat64()*0.6 - 0.3)
	}
	return f
}
