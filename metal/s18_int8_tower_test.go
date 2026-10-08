//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
)

// G-S18f (docs/tasks/task-multimodal-support-2026-10.md, "S18 on the Mac, the tower", registered before the code): the
// Metal int8 SigLIP tower (tower_gemm_w8, weights in groups of 32, activations f32) is exact on its own weights. The
// reference is aikit's CPU float32 tower running the same weights rounded the way the tower rounds them
// (dequantG32(quantG32(w))), so the tower bar holds: every token at cosine >= 0.9999. What the rounding itself costs
// against the true float32 tower is G-S18g's question (the served reply), recorded here as relative L2, with a
// regression guard at 0.06.

// s18RoundWeights rounds every projection of a float32 encoder in place to the int8 tower's weights.
func s18RoundWeights(t *testing.T, enc *vision.Encoder) {
	t.Helper()
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range w.Blocks {
		for _, p := range []vision.VisionProj{b.Q, b.K, b.V, b.O, b.FC1, b.FC2} {
			q, sc := quantG32(p.W, p.Out, p.In)
			copy(p.W, dequantG32(q, sc, p.Out, p.In))
		}
	}
}

func TestS18Int8Tower_tiny(t *testing.T) {
	enc, err := vision.LoadEncoder("../testdata/siglip-tiny", false)
	if err != nil {
		t.Skipf("siglip-tiny: %v", err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	// G-S3a's sharpening: the tiny tower's norms are ones and its biases zero.
	rng := rand.New(rand.NewSource(9))
	for _, b := range w.Blocks {
		gvRandomise(rng, b.LN1W, b.LN1B, b.LN2W, b.LN2B)
	}
	s3Sharpen(rng, w.PatchB)
	for _, b := range w.Blocks {
		s3Sharpen(rng, b.Q.B, b.K.B)
		s3Scale(6, b.Q.W, b.K.W)
	}
	siglipForceInt8 = true
	r, err := newSiglipVResident(enc)
	siglipForceInt8 = false
	if err != nil {
		t.Fatal(err)
	}
	if !r.a.int8W || r.a.blocks[0].fc1.w8 == (Buffer{}) || r.a.blocks[0].fc1.wh != (Buffer{}) {
		t.Fatal("the int8 tower was not built (no int8 weights, or an f16 copy beside them)")
	}
	s18RoundWeights(t, enc) // after the upload: the CPU reference now runs the tower's own weights
	c := enc.Cfg
	px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
	for i := range px {
		px[i] = float32(rng.NormFloat64())
	}
	want, err := enc.Forward(px)
	if err != nil {
		t.Fatal(err)
	}
	patches, err := enc.GridPatches(px)
	if err != nil {
		t.Fatal(err)
	}
	worst := func() float64 {
		got, err := r.ForwardPatches(patches)
		if err != nil {
			t.Fatal(err)
		}
		return gvWorst(t, got, want, c.HiddenSize)
	}
	wc := worst()
	fmt.Fprintf(os.Stderr, "[S18 G-S18f] siglip-tiny int8: worst token cosine %.9f against the CPU on the same weights\n", wc)
	if wc < 0.9999 {
		t.Errorf("siglip-tiny int8: worst token cosine %.9f under 0.9999", wc)
	}
	for _, d := range append(append([]s3Defect(nil), s3SiglipDefects...), s3Defect{7, "each group's scale from the next group (tower_gemm_w8 dbg 4)", nil}) {
		r.a.planted = gvDefect{}
		if d.set != nil {
			d.set(&r.a.planted)
		} else {
			r.a.gemmDbg = 4
		}
		wd := worst()
		r.a.planted, r.a.gemmDbg = gvDefect{}, 0
		fmt.Fprintf(os.Stderr, "[S18 G-S18f] siglip-tiny int8 planted (%d) %s: worst %.6f\n", d.n, d.name, wd)
		if wd >= 0.9999 {
			t.Errorf("planted defect (%d) %s left the bar green (worst %.9f)", d.n, d.name, wd)
		}
	}
}

// TestS18Int8Tower_real is G-S18f at real size: Gemma 3's tower as serve builds it on Metal for -vision-quant int8 (a
// head-only encoder, the blocks streamed, EnableResident), on the four F2a images, against the CPU float32 tower on the
// rounded weights (bar 0.9999 every token), with the rounding's cost against the true float32 tower recorded (relative
// L2, red above 0.06).
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -timeout 30m -run '^TestS18Int8Tower_real$' -v ./metal/
func TestS18Int8Tower_real(t *testing.T) {
	dir := s3Real(t, "gemma-3-4b-it")
	t0 := time.Now()
	logf := func(f string, a ...any) {
		fmt.Fprintf(os.Stderr, "[S18 G-S18f %5.0fs] %s\n", time.Since(t0).Seconds(), fmt.Sprintf(f, a...))
	}
	dev, err := vision.LoadEncoderHead(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil {
		t.Fatalf("the Metal int8 SigLIP tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	if dev.HasBlocks() {
		t.Error("attaching the Metal tower loaded the encoder's host blocks")
	}
	logf("int8 tower attached from a head-only encoder")
	cpu, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	pix := make([][]float32, len(s3Images))
	exact := make([][]float32, len(s3Images))
	for i, img := range s3Images {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		pix[i] = pv.Data
		if exact[i], err = cpu.Forward(pix[i]); err != nil {
			t.Fatal(err)
		}
		logf("%s: the float32 CPU tower", img)
	}
	s18RoundWeights(t, cpu)
	for i, img := range s3Images {
		want, err := cpu.Forward(pix[i])
		if err != nil {
			t.Fatal(err)
		}
		got, err := dev.Forward(pix[i])
		if err != nil {
			t.Fatal(err)
		}
		wc := gvWorst(t, got, want, cpu.Cfg.HiddenSize)
		var num, den float64
		for j := range got {
			d := float64(got[j]) - float64(exact[i][j])
			num, den = num+d*d, den+float64(exact[i][j])*float64(exact[i][j])
		}
		rel := math.Sqrt(num / den)
		logf("%s: worst token cosine %.9f against the CPU on the same weights; relative L2 %.4f against the true float32 tower", img, wc, rel)
		if wc < 0.9999 {
			t.Errorf("%s: worst token cosine %.9f under 0.9999", img, wc)
		}
		if rel > 0.06 {
			t.Errorf("%s: relative L2 %.4f against the float32 tower, over the 0.06 guard", img, rel)
		}
	}
}

// TestS18TowerHostMemory is G-S18h (registered before the code): Gemma 3's tower attached the way serve now attaches it
// on Metal (a head-only encoder, the blocks streamed), at both forms. The host never holds the tower: the Go heap's peak
// during the attach (sampled every 5 ms) stays under 10% of the f32 blocks (2.17 GB), and the encoder holds no blocks
// after it. The process's phys_footprint grows by the device weights + scratch + head within 10% after one image (the
// figures are printed for towerReserve's calibration, G-S18c-Mac). Then the CPU fallback: the device tower detached, the
// head-only encoder's CPU forward loads its blocks and matches LoadEncoder's float32 tower exactly.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -timeout 30m -run '^TestS18TowerHostMemory$' -v ./metal/
func TestS18TowerHostMemory(t *testing.T) {
	dir := s3Real(t, "gemma-3-4b-it")
	data, err := os.ReadFile("../testdata/glm_ocr/table.png")
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(data, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	for _, int8 := range []bool{false, true} {
		name := map[bool]string{false: "f16", true: "int8"}[int8]
		t.Run(name, func(t *testing.T) { s18TowerHostMemory(t, dir, pv.Data, int8, name) })
	}
}

// s18TowerHostMemory is one form of G-S18h; run each in its own process (-run 'TestS18TowerHostMemory/int8'), or the
// second's footprint reads net of what the first released.
func s18TowerHostMemory(t *testing.T, dir string, px []float32, int8 bool, name string) {
	{
		runtime.GC()
		base := s18Footprint(t)
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		heap0 := ms.HeapAlloc
		stop, peak := make(chan struct{}), make(chan uint64)
		go func() {
			var mx uint64
			var m runtime.MemStats
			tk := time.NewTicker(5 * time.Millisecond)
			defer tk.Stop()
			for {
				select {
				case <-stop:
					peak <- mx
					return
				case <-tk.C:
					runtime.ReadMemStats(&m)
					mx = max(mx, m.HeapAlloc)
				}
			}
		}()
		enc, err := vision.LoadEncoderHead(dir, int8)
		if err != nil {
			t.Fatal(err)
		}
		if err := enc.EnableResident(); err != nil {
			t.Fatalf("%s: the Metal tower did not attach: %v", name, err)
		}
		close(stop)
		heapPeak := <-peak - heap0
		if _, err := enc.Forward(px); err != nil {
			t.Fatal(err)
		}
		runtime.GC()
		grew := s18Footprint(t) - base
		fmt.Fprintf(os.Stderr, "[S18 G-S18h] %s tower: Go heap peak during the attach +%.0f MB; phys_footprint +%.0f MB after one image; host blocks held: %v\n",
			name, float64(heapPeak)/1e6, float64(grew)/1e6, enc.HasBlocks())
		if heapPeak > 217e6 {
			t.Errorf("%s: the attach held %.0f MB of Go heap at its peak, over 10%% of the f32 blocks", name, float64(heapPeak)/1e6)
		}
		if enc.HasBlocks() {
			t.Errorf("%s: the encoder holds its blocks after the attach", name)
		}
		if int8 {
			enc.Close()
			return
		}
		// The CPU fallback: detached, the head-only encoder runs the float32 CPU tower.
		enc.Close()
		got, err := enc.Forward(px)
		if err != nil {
			t.Fatal(err)
		}
		full, err := vision.LoadEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		want, err := full.Forward(px)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("the CPU fallback differs from LoadEncoder's tower at %d", i)
			}
		}
		fmt.Fprintln(os.Stderr, "[S18 G-S18h] the CPU fallback of a head-only encoder matches LoadEncoder's float32 tower exactly")
		runtime.KeepAlive(full)
	}
}

// s18Footprint is the process's phys_footprint in bytes (footprint(1)), what Metal's unified memory charges.
func s18Footprint(t *testing.T) int64 {
	t.Helper()
	out, err := exec.Command("footprint", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Fatalf("footprint: %v", err)
	}
	for _, l := range strings.Split(string(out), "\n") {
		if i := strings.Index(l, "phys_footprint:"); i >= 0 {
			f := strings.Fields(l[i+len("phys_footprint:"):])
			if len(f) >= 2 {
				v, _ := strconv.ParseFloat(f[0], 64)
				switch f[1] {
				case "KB":
					return int64(v * 1e3)
				case "MB":
					return int64(v * 1e6)
				case "GB":
					return int64(v * 1e9)
				}
			}
		}
	}
	t.Fatalf("no phys_footprint in %q", out)
	return 0
}
