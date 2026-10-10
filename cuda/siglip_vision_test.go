//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
)

// G-S3a on CUDA for the float32 SigLIP tower (the S4 addendum, docs/tasks/task-multimodal-support-2026-10.md): the CUDA tower against aikit's CPU tower, both float32,
// every output token at cosine >= 0.9999 (0.999-0.9999 ambiguous, parked). The Metal rebuild's test shape, bars and planted defects: (1) the attention scale dropped,
// (5) the position table dropped, (6) the patch-embed bias dropped, each alone required to turn the tiny check red. The tiny tower is sharpened first (norms
// randomised; random patch and q/k biases; q/k weights scaled up) because its all-ones norms, zero biases and init-scale q/k hid 5 of S3's 7 defects. The real tower
// (~/models/gemma-3-4b-it, never the archive) runs under GOINFER_HEAVY_TESTS=1 on the four F2a images, attached the way serve attaches it, and the test asserts the
// attached type so another package's registration cannot satisfy it. head_dim is 72 at real size, which no other tower on this base has.

var sigImages = []string{"gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"}

func sigSharpen(rng *rand.Rand, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] = float32(0.5 * rng.NormFloat64())
		}
	}
}

func sigScale(f float32, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] *= f
		}
	}
}

func sigGrade(t *testing.T, what string, got, want []float32, width int) float64 {
	t.Helper()
	w := gridWorst(t, got, want, width)
	fmt.Fprintf(os.Stderr, "[S4 G-S3a siglip] %s: worst token cosine %.9f\n", what, w)
	switch {
	case w < 0.999:
		t.Errorf("%s: worst token cosine %.9f under 0.999 (G-S3a FAIL)", what, w)
	case w < 0.9999:
		t.Errorf("%s: worst token cosine %.9f in 0.999-0.9999 (G-S3a ambiguous, parked)", what, w)
	}
	return w
}

// siglipAttachedTower returns the *siglipTower attached to enc, failing if another registration won.
func siglipAttachedTower(t *testing.T, enc *vision.Encoder) *siglipTower {
	t.Helper()
	rv := reflect.ValueOf(enc).Elem().FieldByName("resident")
	if rv.IsNil() {
		t.Fatal("no resident tower attached")
	}
	// resident is unexported: read it through its address.
	r, ok := reflect.NewAt(rv.Type(), unsafe.Pointer(rv.UnsafeAddr())).Elem().Interface().(vision.ResidentEncoder)
	if !ok {
		t.Fatal("the attached value is not a ResidentEncoder")
	}
	st, ok := r.(*siglipTower)
	if !ok {
		t.Fatalf("the attached tower is %T, not the float32 *cuda.siglipTower", r)
	}
	return st
}

func TestSiglipCUDA_tiny(t *testing.T) {
	newTestTower(t, 64) // skips without a CUDA device
	enc, err := vision.LoadEncoder("../testdata/siglip-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(9))
	for _, b := range w.Blocks {
		gridRandomise(rng, b.LN1W, b.LN1B, b.LN2W, b.LN2B)
	}
	sigSharpen(rng, w.PatchB)
	for _, b := range w.Blocks {
		sigSharpen(rng, b.Q.B, b.K.B)
		sigScale(6, b.Q.W, b.K.W)
	}
	if err := enc.EnableResident(); err != nil {
		t.Fatalf("the float32 encoder did not attach a tower: %v", err)
	}
	t.Cleanup(enc.Close)
	r := siglipAttachedTower(t, enc)
	c := enc.Cfg
	px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
	for i := range px {
		px[i] = float32(rng.NormFloat64())
	}
	// The reference: aikit's CPU tower, on a second encoder with the SAME (sharpened, aliased) weights and no resident attached.
	cpu, err := vision.LoadEncoder("../testdata/siglip-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	cw, _ := cpu.Weights()
	for i, b := range w.Blocks { // copy the sharpened weights across (the slices alias each encoder's own)
		cb := cw.Blocks[i]
		copy(cb.LN1W, b.LN1W)
		copy(cb.LN1B, b.LN1B)
		copy(cb.LN2W, b.LN2W)
		copy(cb.LN2B, b.LN2B)
		copy(cb.Q.W, b.Q.W)
		copy(cb.Q.B, b.Q.B)
		copy(cb.K.W, b.K.W)
		copy(cb.K.B, b.K.B)
	}
	copy(cw.PatchB, w.PatchB)
	want, err := cpu.Forward(px)
	if err != nil {
		t.Fatal(err)
	}
	patches, err := enc.GridPatches(px)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.ForwardPatches(patches)
	if err != nil {
		t.Fatal(err)
	}
	sigGrade(t, "siglip-tiny", got, want, c.HiddenSize)
	defects := []struct {
		n    int
		name string
		set  func(d *gridDefect)
	}{
		{1, "attention scale dropped", func(d *gridDefect) { d.noScale = true }},
		{5, "position table dropped", func(d *gridDefect) { d.noPosEmbed = true }},
		{6, "patch-embed bias dropped", func(d *gridDefect) { d.noPatchBias = true }},
	}
	for _, d := range defects {
		r.g.planted = gridDefect{}
		d.set(&r.g.planted)
		g2, err := r.ForwardPatches(patches)
		r.g.planted = gridDefect{}
		if err != nil {
			t.Fatal(err)
		}
		wd := gridWorst(t, g2, want, c.HiddenSize)
		fmt.Fprintf(os.Stderr, "[S4 G-S3a siglip] siglip-tiny planted (%d) %s: worst %.6f\n", d.n, d.name, wd)
		if wd >= 0.9999 {
			t.Errorf("siglip-tiny: planted defect (%d) %s left the bar green (worst %.9f): the fixture cannot see it", d.n, d.name, wd)
		}
	}
}

// TestSiglipCUDA_int8StaysTheInt8Tower: an int8-loaded encoder still gets the W8A8 tower (the dispatching registration must not change the default).
func TestSiglipCUDA_int8StaysTheInt8Tower(t *testing.T) {
	newTestTower(t, 64)
	enc, err := vision.LoadEncoder("../testdata/siglip-tiny", true)
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.EnableResident(); err != nil {
		t.Skipf("the int8 tiny tower does not attach on this build: %v", err)
	}
	t.Cleanup(enc.Close)
	typ := reflect.ValueOf(enc).Elem().FieldByName("resident").Elem().Type().String()
	if !strings.Contains(typ, "VisionEncoder") {
		t.Errorf("an int8 encoder attached %s, want the W8A8 *cuda.VisionEncoder", typ)
	}
}

func TestSiglipCUDA_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower)")
	}
	newTestTower(t, 64)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if strings.HasPrefix(dir, "/srv/models") || strings.HasPrefix(dir, "/Volumes/") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	cpu, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil {
		t.Fatalf("the CUDA float32 SigLIP tower did not attach: %v", err)
	}
	t.Cleanup(dev.Close)
	siglipAttachedTower(t, dev)
	for _, img := range sigImages {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		want, err := cpu.Forward(pv.Data)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		got, err := dev.Forward(pv.Data)
		td := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "[S4 G-S3a siglip real] %-30s tower CUDA %s, CPU %s (exploratory)\n", img, td.Round(time.Millisecond), tc.Round(time.Millisecond))
		sigGrade(t, "gemma-3-4b-it "+img, got, want, cpu.Cfg.HiddenSize)
	}
}

// TestSiglipCUDA_int8VsFloat32 is G-S3d's feature half (the S4 addendum): the shipped default's tower, the W8A8
// *cuda.VisionEncoder on an int8-loaded encoder, against the float32 CPU tower on the four F2a images: relative L2 and worst
// / mean per-token cosine, to set beside docs/measurements/siglip-int8-fidelity-2026-10-07.md. A record, not a gate: it
// asserts only that the int8 tower is what attached. Heavy.
func TestSiglipCUDA_int8VsFloat32(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1 (loads the real tower twice)")
	}
	newTestTower(t, 64)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	ref, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	q, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := q.EnableResident(); err != nil {
		t.Fatalf("the int8 CUDA tower did not attach: %v", err)
	}
	t.Cleanup(q.Close)
	if typ := reflect.ValueOf(q).Elem().FieldByName("resident").Elem().Type().String(); !strings.Contains(typ, "VisionEncoder") {
		t.Fatalf("an int8 encoder attached %s, want the W8A8 *cuda.VisionEncoder", typ)
	}
	W := ref.Cfg.HiddenSize
	for _, img := range sigImages {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		want, err := ref.Forward(pv.Data)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		got, err := q.Forward(pv.Data)
		td := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		var num, den, sumCos float64
		worst := 1.0
		n := len(want) / W
		for i := range n {
			var dot, na, nb float64
			for j := range W {
				a, b := float64(want[i*W+j]), float64(got[i*W+j])
				dot += a * b
				na += a * a
				nb += b * b
				num += (a - b) * (a - b)
				den += a * a
			}
			c := dot / (math.Sqrt(na*nb) + 1e-30)
			sumCos += c
			worst = math.Min(worst, c)
		}
		fmt.Fprintf(os.Stderr, "[S4 G-S3d] %-30s int8 CUDA tower vs float32 CPU: relative L2 %.4f, worst token cosine %.4f, mean token cosine %.4f (int8 tower %s)\n",
			img, math.Sqrt(num/den), worst, sumCos/float64(n), td.Round(time.Millisecond))
	}
}

// TestSiglipCUDA_speed is the S4 addendum's speed record (no bar): Gemma 3's SigLIP tower time per image, the float32 CUDA tower against the int8 CUDA tower against the CPU float32 tower,
// interleaved per round on the same preprocessed image (SIGLIP_SPEED_ROUNDS, default 3; images: the first and last of the F2a four). Heavy; run under the timing lock (the night script does).
func TestSiglipCUDA_speed(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	newTestTower(t, 64)
	rounds := 3
	if v := os.Getenv("SIGLIP_SPEED_ROUNDS"); v != "" {
		fmt.Sscan(v, &rounds)
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	cpu, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	f32, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := f32.EnableResident(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(f32.Close)
	siglipAttachedTower(t, f32)
	i8, err := vision.LoadEncoder(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := i8.EnableResident(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(i8.Close)
	for _, img := range []string{sigImages[0], sigImages[3]} {
		data, err := os.ReadFile(filepath.Join("../testdata", img))
		if err != nil {
			t.Fatal(err)
		}
		pv, err := vision.Preprocess(data, vision.Gemma3())
		if err != nil {
			t.Fatal(err)
		}
		for _, enc := range []*vision.Encoder{f32, i8} { // warm: the first call builds scratch and JITs nothing new, but is not timed
			if _, err := enc.Forward(pv.Data); err != nil {
				t.Fatal(err)
			}
		}
		for r := 1; r <= rounds; r++ {
			tm := func(e *vision.Encoder) time.Duration {
				t0 := time.Now()
				if _, err := e.Forward(pv.Data); err != nil {
					t.Fatal(err)
				}
				return time.Since(t0)
			}
			a, b, c := tm(cpu), tm(f32), tm(i8)
			fmt.Fprintf(os.Stderr, "[S4 siglip speed] %-28s round %d: CPU f32 %.2fs | CUDA f32 %.2fs | CUDA int8 %.2fs\n", img, r, a.Seconds(), b.Seconds(), c.Seconds())
		}
	}
}

// TestSiglipCUDA_int8VRAM measures what the shipped default's int8 SigLIP tower claims on the device (S18's G-S18c needs the figure to price it): free VRAM before the encoder attaches, after it attaches, and after
// one forward (the peak, since scratch may be allocated lazily). Heavy; a record.
func TestSiglipCUDA_int8VRAM(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	newTestTower(t, 64)
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
	data, err := os.ReadFile("../testdata/" + sigImages[0])
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

// TestSiglipCUDA_fusedAttentionKernelDefectsReal: S17 lever A's kernel defects at real size, on the SigLIP tower (head dim 72, 4096 patches, 27 layers): defects 1 and 3 each alone drop the worst token cosine under the bar; defect 2 (one zero-score zero-value key, 1/4097 of the weight over 4096 keys) is invisible at real size, so it is logged, not asserted.
func TestSiglipCUDA_fusedAttentionKernelDefectsReal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	newTestTower(t, 64)
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-3-4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no %s: %v", dir, err)
	}
	cpu, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	dev, err := vision.LoadEncoder(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.EnableResident(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(dev.Close)
	siglipAttachedTower(t, dev)
	data, err := os.ReadFile("../testdata/" + sigImages[0])
	if err != nil {
		t.Fatal(err)
	}
	pv, err := vision.Preprocess(data, vision.Gemma3())
	if err != nil {
		t.Fatal(err)
	}
	want, err := cpu.Forward(pv.Data)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { towerAttnDefect = 0 }()
	if got, err := dev.Forward(pv.Data); err != nil {
		t.Fatal(err)
	} else if w := gridWorst(t, got, want, cpu.Cfg.HiddenSize); w < 0.9999 {
		t.Fatalf("the unplanted fused-kernel tower reads %.9f, under the bar", w)
	}
	for _, c := range []struct {
		d       int
		name    string
		visible bool // whether real size can see it: defect 2 adds one zero-score zero-value key, 1/4097 of the weight over 4096 keys
	}{{1, "no rescale when the running max moves", true}, {2, "key mask one past the segment", false}, {3, "V tile read one key late", true}} {
		towerAttnDefect = c.d
		got, err := dev.Forward(pv.Data)
		towerAttnDefect = 0
		if err != nil {
			t.Fatal(err)
		}
		w := gridWorst(t, got, want, cpu.Cfg.HiddenSize)
		fmt.Fprintf(os.Stderr, "[S17 lever A] real SigLIP, kernel defect (%d) %s: worst token cosine %.6f (visible at real size: %v)\n", c.d, c.name, w, c.visible)
		if c.visible && w >= 0.9999 {
			t.Errorf("kernel defect (%d) %s left the real SigLIP tower at %.9f: the test cannot see it", c.d, c.name, w)
		}
	}
}
