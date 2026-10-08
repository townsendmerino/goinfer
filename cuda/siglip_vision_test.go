//go:build cuda

package cuda

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/townsendmerino/aikit/vision"
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
