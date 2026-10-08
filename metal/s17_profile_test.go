//go:build darwin

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestS17Profile is S17 step 0 on Metal (docs/tasks/task-multimodal-support-2026-10.md, registered before it ran): each
// Metal grid tower's GPU time split by kernel class (gemm with its bias, attention, norm, elementwise) through the
// gvProfile seam, on the four F2a images at serve's caps: one unprofiled warm-up, one unprofiled timed forward (the
// overhead check's reference), then 3 profiled forwards, the median per class reported. The profiled output must equal
// the unprofiled one bit for bit (splitting command buffers changes no arithmetic). A profile, not a speed claim.
// GOINFER_HEAVY_TESTS=1; the checkpoints from ~/models.
func TestS17Profile(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	model := func(name string) string {
		dir := filepath.Join(home, "models", name)
		if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
			t.Fatalf("%s is the archive", dir)
		}
		if _, err := os.Stat(dir); err != nil {
			t.Skipf("no %s", dir)
		}
		return dir
	}
	image := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join("../testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	fmt.Fprintf(os.Stderr, "| tower | image | rows | unprofiled ms | profiled GPU ms | gemm ms (share, TFLOPS) | attention ms (share, TFLOPS) | norm ms (share) | elementwise ms (share) | buffers |\n|---|---|---|---|---|---|---|---|---|---|\n")
	profile := func(tower, img string, a *gridVAccel, rows int, fwd func() ([]float32, error)) {
		if _, err := fwd(); err != nil { // warm-up
			t.Fatal(err)
		}
		t0 := time.Now()
		ref, err := fwd()
		wall := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		per := map[string][]float64{}
		var flops map[string]float64
		var bufs int
		for range 3 {
			a.prof = &gvProfile{}
			a.prof.reset()
			got, err := fwd()
			p := a.prof
			a.prof = nil
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("%s %s: the profiled forward differs from the unprofiled one", tower, img)
			}
			total := 0.0
			for c, s := range p.gpu {
				per[c] = append(per[c], s)
				total += s
			}
			per["total"] = append(per["total"], total)
			flops, bufs = p.flops, p.bufs
		}
		med := func(c string) float64 {
			xs := slices.Clone(per[c])
			if len(xs) == 0 {
				return 0
			}
			sort.Float64s(xs)
			return xs[len(xs)/2]
		}
		tot := med("total")
		cell := func(c string, withRate bool) string {
			s := med(c)
			out := fmt.Sprintf("%.1f (%.0f%%", 1e3*s, 100*s/tot)
			if withRate && s > 0 {
				out += fmt.Sprintf(", %.2f", flops[c]/s/1e12)
			}
			return out + ")"
		}
		fmt.Fprintf(os.Stderr, "| %s | %s | %d | %.1f | %.1f | %s | %s | %s | %s | %d |\n", tower, img, rows,
			float64(wall.Microseconds())/1e3, 1e3*tot, cell("gemm", true), cell("attention", true), cell("norm", false),
			cell("elementwise", false), bufs)
		if tot > wall.Seconds() {
			fmt.Fprintf(os.Stderr, "  !! %s %s: profiled GPU time %.1f ms exceeds the unprofiled wall %.1f ms: the profile is not trustworthy here\n",
				tower, img, 1e3*tot, float64(wall.Microseconds())/1e3)
		}
	}

	t.Run("siglip", func(t *testing.T) {
		enc, err := vision.LoadEncoder(model("gemma-3-4b-it"), false)
		if err != nil {
			t.Fatal(err)
		}
		r, err := newSiglipVResident(enc)
		if err != nil {
			t.Fatal(err)
		}
		for _, img := range s3Images {
			pv, err := vision.Preprocess(image(img), vision.Gemma3())
			if err != nil {
				t.Fatal(err)
			}
			patches, err := enc.GridPatches(pv.Data)
			if err != nil {
				t.Fatal(err)
			}
			profile("SigLIP", img, r.a, r.np, func() ([]float32, error) { return r.hidden(patches) })
		}
	})
	t.Run("qwen2.5-vl", func(t *testing.T) {
		dir := model("qwen25vl-3b-instruct")
		enc, err := vision.LoadQwenVisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		r, err := newQwen25VResident(enc)
		if err != nil {
			t.Fatal(err)
		}
		pp, err := multimodal.LoadQwenPreprocessConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, img := range s3Images {
			px, grid, err := multimodal.QwenPreprocess(image(img), pp)
			if err != nil {
				t.Fatal(err)
			}
			g := [][3]int{grid}
			profile("Qwen2.5-VL", img, r.a, grid[0]*grid[1]*grid[2], func() ([]float32, error) { return r.ForwardViT(px, g) })
		}
	})
	t.Run("qwen3.5", func(t *testing.T) {
		dir := model("qwen3.5-0.8b")
		enc, err := vision.LoadQwen3VisionEncoder(dir, false)
		if err != nil {
			t.Fatal(err)
		}
		a, err := newQwen3VAccel(enc)
		if err != nil {
			t.Fatal(err)
		}
		pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit { // serve's cap
			pp.MaxPixels = limit
		}
		for _, img := range s3Images {
			px, grid, err := multimodal.QwenPreprocess(image(img), pp)
			if err != nil {
				t.Fatal(err)
			}
			g := [][3]int{grid}
			profile("Qwen3.5-0.8B", img, a, grid[0]*grid[1]*grid[2], func() ([]float32, error) { return a.Hidden(px, g) })
		}
	})
}
