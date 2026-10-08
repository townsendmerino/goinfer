//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestS16MRoPEPrefill_tiny is G-S16a and G-S16b (defects 1-3) of S16 (docs/tasks/task-multimodal-support-2026-10.md,
// registered before any S16 code): on one Metal-resident int4 model, the new resident m-RoPE prefill against today's path,
// on the tiny Qwen2.5-VL (contiguous sections) and Qwen3.5 (interleaved, a Gated-DeltaNet hybrid) fixtures.
//
// Today's path is what the decoder runs for the family: for Qwen2.5-VL the CPU's m-RoPE prefill with its K/V uploaded and
// Metal decode; for the hybrid, the CPU prefill and CPU decode (the upload bridge copies only K/V and would leave the
// recurrent state zeroed, so the decoder never takes it for a hybrid).
//
// Each fixture runs twice: its golden image prompt (a 2x3 merged grid, positions under 6, where most rotary pairs barely
// turn and a position defect can hide), and the same prompt with a 16x24 image (96 merged rows) whose features are random
// at the golden features' scale. The second sharpens the fixture along the axis the defects move (S3's lesson); it needs
// no HF golden, because both arms run the same weights on the same inputs.
//
// Compared: the last prompt row's logits, then 8 decode steps teacher-forced along today's path's own greedy tokens (so
// both arms see the same inputs), each step's logits at cosine >= 0.9999 with an equal argmax (0.999-0.9999 is parked).
// The decode steps exercise the K/V the prefill wrote, which the last-row logits alone do not. Bit-identity is not
// expected: the batched pass is f16 MMA against the CPU's int4 kernels.
//
// Each planted defect corrupts only the new arm's positions and must miss the bar on at least one fixture: (1) the
// height and width components swapped; (2) the image rows given 1-D text positions; (3) the text rows after the image
// given uncompressed (sequence) positions. Defects 4-6 (DeepStack) belong to S16's Qwen3-VL step.
func TestS16MRoPEPrefill_tiny(t *testing.T) {
	type golden struct {
		InputIDs      []int     `json:"input_ids"`
		ImageToken    int       `json:"image_token_id"`
		ImageStart    int       `json:"image_token_start"`
		NImageTokens  int       `json:"n_image_tokens"`
		GridTHW       [][3]int  `json:"grid_thw"`
		ImageFeatures []float32 `json:"image_features"`
	}
	const merge, steps, bar = 2, 8, 0.9999
	defects := []string{"none", "(1) height and width swapped", "(2) image rows given 1-D positions", "(3) text after the image uncompressed"}
	caught := make([]bool, len(defects))
	for _, fx := range []struct {
		dir, golden string
		hybrid, big bool
	}{
		{"qwen25vl-tiny", "qwen25vl_tiny_image_golden.json", false, false},
		{"qwen25vl-tiny", "qwen25vl_tiny_image_golden.json", false, true},
		{"qwen35vl-tiny", "qwen35vl_tiny_image_golden.json", true, false},
		{"qwen35vl-tiny", "qwen35vl_tiny_image_golden.json", true, true},
	} {
		path := filepath.Join("..", "testdata", fx.dir)
		raw, err := os.ReadFile(filepath.Join("..", "testdata", fx.golden))
		if err != nil {
			t.Fatalf("golden: %v", err)
		}
		if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
			t.Fatalf("no fixture at %s", path)
		}
		var g golden
		if err := json.Unmarshal(raw, &g); err != nil {
			t.Fatal(err)
		}
		label := fx.dir
		if fx.big {
			const gh, gw = 8, 12 // merged rows and columns: a 16x24 patch grid
			nb, hid := gh*gw, len(g.ImageFeatures)/g.NImageTokens
			var ss float64
			for _, v := range g.ImageFeatures {
				ss += float64(v) * float64(v)
			}
			rms := math.Sqrt(ss / float64(len(g.ImageFeatures)))
			rng := rand.New(rand.NewSource(16))
			feats := make([]float32, nb*hid)
			for i := range feats {
				feats[i] = float32(rms * rng.NormFloat64())
			}
			ids := append(append([]int{}, g.InputIDs[:g.ImageStart]...), make([]int, nb)...)
			for i := range nb {
				ids[g.ImageStart+i] = g.ImageToken
			}
			ids = append(ids, g.InputIDs[g.ImageStart+g.NImageTokens:]...)
			g.InputIDs, g.NImageTokens, g.GridTHW, g.ImageFeatures = ids, nb, [][3]int{{1, 2 * gh, 2 * gw}}, feats
			label += " (16x24 image)"
		}
		// Tiny prompts sit under the batched pass's TTFT floor; a real image turn is hundreds of rows.
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256,
			Knobs: &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"}})
		if err != nil {
			t.Fatalf("load %s: %v", fx.dir, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			m.Close()
			t.Fatalf("%s: not Metal-resident (%T): %s", fx.dir, m.ResidentForwardForTest(), m.ResidentDecline())
		}
		if a.r.mropeAxis == (Buffer{}) {
			m.Close()
			t.Fatalf("%s: no m-RoPE axis table built", fx.dir)
		}
		mrope, err := decoder.MRopePositionsForTest(g.InputIDs, g.ImageToken, g.GridTHW, merge)
		if err != nil {
			t.Fatal(err)
		}
		n := len(g.InputIDs)
		delta := mrope[n-1][0] + 1 - n // the decode positions continue from the last row's (compressed) position
		ctx := context.Background()

		// Today's path: the CPU prefill, its K/V uploaded, Metal decode. Its greedy tokens are the teacher for both arms.
		cache := m.NewCache(n + steps + 1)
		refLast, err := m.PrefillLogitsQwenVLForTest(ctx, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, mrope, cache)
		if err != nil {
			t.Fatalf("cpu prefill: %v", err)
		}
		refLast = append([]float32(nil), refLast...)
		a.Reset()
		if !fx.hybrid {
			if err := m.ResidentUploadPrefillForTest(cache); err != nil {
				t.Fatalf("upload: %v", err)
			}
		}
		decode := func(first []float32, forced []int) ([][]float32, []int) {
			out, toks := [][]float32{first}, []int{argmaxF(first)}
			for k := range steps {
				tok := toks[k]
				if forced != nil {
					tok = forced[k]
				}
				var lg []float32
				var err error
				if fx.hybrid && forced == nil { // the hybrid's reference arm decodes on the CPU, from the CPU prefill's cache
					lg, err = m.ForwardForTest(tok, cache)
				} else {
					lg, err = a.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
				}
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), lg...))
				toks = append(toks, argmaxF(lg))
			}
			return out, toks
		}
		ref, teacher := decode(refLast, nil)

		for di, name := range defects {
			pos := make([][3]int, n)
			copy(pos, mrope)
			for i := range pos {
				inImage := i >= g.ImageStart && i < g.ImageStart+g.NImageTokens
				switch {
				case di == 1:
					pos[i][1], pos[i][2] = pos[i][2], pos[i][1]
				case di == 2 && inImage:
					pos[i] = [3]int{i, i, i}
				case di == 3 && i >= g.ImageStart+g.NImageTokens:
					pos[i] = [3]int{i, i, i}
				}
			}
			a.Reset()
			newLast, _, err := m.ResidentMRoPEPrefillForTest(ctx, a, g.InputIDs, g.ImageFeatures, g.ImageStart, g.NImageTokens, pos)
			if err != nil {
				t.Fatalf("%s %s: resident m-RoPE prefill: %v", fx.dir, name, err)
			}
			got, _ := decode(append([]float32(nil), newLast...), teacher)
			worst, agree := 1.0, 0
			for k := range ref {
				worst = math.Min(worst, cosF(ref[k], got[k]))
				if argmaxF(ref[k]) == argmaxF(got[k]) {
					agree++
				}
			}
			pass := worst >= bar && agree == len(ref)
			fmt.Printf("[S16 G-S16a] %s %s: worst cosine %.7f over the last row and %d steps, argmax %d/%d\n",
				label, name, worst, steps, agree, len(ref))
			if di == 0 && !pass {
				t.Errorf("%s: the resident m-RoPE prefill misses the bar (worst %.7f, argmax %d/%d)", label, worst, agree, len(ref))
			}
			if di > 0 && !pass {
				caught[di] = true
			}
		}
		// Through the decoder's entry point (the golden prompt): GenerateQwenVL on Metal must take the resident m-RoPE prefill
		// for Qwen2.5-VL (ImgPrefillResident) and not for the unclaimed hybrid, and its greedy tokens must equal today's path's:
		// for Qwen2.5-VL a second Metal model with the batched prefill off (so the upload bridge runs, then Metal decode); for
		// the hybrid a CPU model (the decoder keeps a hybrid's image turn on the CPU without the marker).
		if !fx.big {
			refOpts := decoder.Options{Quant: "int4"}
			if !fx.hybrid {
				refOpts = decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256,
					Knobs: &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL": "0"}}
			}
			cpu, err := decoder.Load(path, refOpts)
			if err != nil {
				t.Fatal(err)
			}
			gen := func(mm *decoder.Model) ([]int, bool) {
				stream, gg := mm.GenerateQwenVL(ctx, g.InputIDs, g.ImageStart, g.NImageTokens, 0xfeed,
					func() ([]float32, error) { return g.ImageFeatures, nil }, g.GridTHW, merge, g.ImageToken, steps, decoder.SamplingParams{})
				var out []int
				for id := range stream {
					out = append(out, id)
				}
				if e := gg.Err(); e != nil {
					t.Fatalf("%s: generation: %v", label, e)
				}
				return out, gg.ImgPrefillResident
			}
			want, cpuRes := gen(cpu)
			got, metalRes := gen(m)
			cpu.Close()
			fmt.Printf("[S16 G-S16a] %s through GenerateQwenVL: resident prefill %v, tokens %v (today's path %v)\n", label, metalRes, got, want)
			// The hybrid is not claimed (HybridMRoPEPrefill is not implemented on Metal): its image turn must stay on today's path.
			if cpuRes || metalRes != !fx.hybrid {
				t.Errorf("%s: resident prefill flags today's path %v / new %v, want false / %v", label, cpuRes, metalRes, !fx.hybrid)
			}
			if fmt.Sprint(got) != fmt.Sprint(want) {
				t.Errorf("%s: the new path's greedy tokens %v, today's path's %v", label, got, want)
			}
		}
		m.Close()
	}
	for di := 1; di < len(defects); di++ {
		if !caught[di] {
			t.Errorf("planted defect %s left the bar green on every fixture", defects[di])
		}
	}
}

// TestS16MRoPEPrefill_real is G-S16c's first half (docs/tasks/task-multimodal-support-2026-10.md, S16, registered before
// any S16 code): Qwen2.5-VL-3B on the four F2a images, the resident m-RoPE prefill against today's path (the upload bridge,
// then Metal decode, so both arms decode on the same kernels) at G-S16a's bar (the last prompt row and 8 teacher-forced
// decode steps, cosine >= 0.9999 with an equal argmax). The hybrid branch is kept for when a hybrid is claimed: Qwen3.5's
// only reference decodes on the CPU, and on image prompts every pair of backends differs there by about 0.9 cosine
// (2026-10-08), so its per-step bar cannot isolate the prefill; the hybrid is not claimed. Features: Qwen2.5-VL's from the encoder with its Metal tower attached (serve's path), Qwen3.5's
// from the CPU encoder at serve's 1024-row cap; both arms see the same ones. The prompt is the model's own chat shape
// with the image placeholder expanded to the image's merged rows. Qwen3-VL (DeepStack) is S16's next step.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -timeout 30m -run '^TestS16MRoPEPrefill_real$' -v ./metal/
func TestS16MRoPEPrefill_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	const merge, steps, bar = 2, 8, 0.9999
	for _, fx := range []struct {
		dir    string
		hybrid bool
	}{{"qwen25vl-3b-instruct", false}} {
		t.Run(fx.dir, func(t *testing.T) {
			dir := filepath.Join(home, "models", fx.dir)
			if _, err := os.Stat(dir); err != nil {
				t.Skipf("no %s", dir)
			}
			tk, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
			if err != nil {
				t.Fatal(err)
			}
			imgTok, ok1 := tk.TokenID("<|image_pad|>")
			if !ok1 {
				t.Fatal("no <|image_pad|> token")
			}
			type img struct {
				px   []float32
				grid [3]int
			}
			var imgs []img
			var featsOf func(img) ([]float32, error)
			if fx.hybrid {
				enc, err := vision.LoadQwen3VisionEncoder(dir, false)
				if err != nil {
					t.Fatal(err)
				}
				pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
				if err != nil {
					t.Fatal(err)
				}
				if limit := 1024 * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
					pp.MaxPixels = limit
				}
				for _, name := range s3Images {
					data, _ := os.ReadFile(filepath.Join("../testdata", name))
					px, grid, err := multimodal.QwenPreprocess(data, pp)
					if err != nil {
						t.Fatal(err)
					}
					imgs = append(imgs, img{px, grid})
				}
				featsOf = func(i img) ([]float32, error) { return enc.Forward(i.px, [][3]int{i.grid}) }
			} else {
				enc, err := vision.LoadQwenVisionEncoder(dir, false)
				if err != nil {
					t.Fatal(err)
				}
				if err := enc.EnableResident(); err != nil || !enc.ResidentEnabled() {
					t.Fatalf("the Metal Qwen2.5-VL tower did not attach: %v", err)
				}
				defer enc.Close()
				pp, err := multimodal.LoadQwenPreprocessConfig(dir)
				if err != nil {
					t.Fatal(err)
				}
				for _, name := range s3Images {
					data, _ := os.ReadFile(filepath.Join("../testdata", name))
					px, grid, err := multimodal.QwenPreprocess(data, pp)
					if err != nil {
						t.Fatal(err)
					}
					imgs = append(imgs, img{px, grid})
				}
				featsOf = func(i img) ([]float32, error) { return enc.Forward(i.px, [][3]int{i.grid}) }
			}
			m, err := decoder.Load(dir, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: 8192})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok || a.r.mropeAxis == (Buffer{}) {
				t.Fatalf("not Metal-resident with an m-RoPE axis table: %s", m.ResidentDecline())
			}
			ctx := context.Background()
			for ii, im := range imgs {
				feats, err := featsOf(im)
				if err != nil {
					t.Fatal(err)
				}
				nImg := im.grid[0] * im.grid[1] * im.grid[2] / (merge * merge)
				pre, err := tk.Encode("<|im_start|>user\n<|vision_start|>", false)
				if err != nil {
					t.Fatal(err)
				}
				post, err := tk.Encode("<|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n", false)
				if err != nil {
					t.Fatal(err)
				}
				ids := append(append([]int{}, pre...), make([]int, nImg)...)
				for k := range nImg {
					ids[len(pre)+k] = imgTok
				}
				ids = append(ids, post...)
				n, start := len(ids), len(pre)
				mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{im.grid}, merge)
				if err != nil {
					t.Fatal(err)
				}
				delta := mrope[n-1][0] + 1 - n
				cache := m.NewCache(n + steps + 1)
				refLast, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nImg, mrope, cache)
				if err != nil {
					t.Fatal(err)
				}
				refLast = append([]float32(nil), refLast...)
				a.Reset()
				if !fx.hybrid {
					if err := m.ResidentUploadPrefillForTest(cache); err != nil {
						t.Fatal(err)
					}
				}
				decode := func(first []float32, forced []int) ([][]float32, []int) {
					out, toks := [][]float32{first}, []int{argmaxF(first)}
					for k := range steps {
						tok := toks[k]
						if forced != nil {
							tok = forced[k]
						}
						var lg []float32
						if fx.hybrid && forced == nil {
							lg, err = m.ForwardForTest(tok, cache)
						} else {
							lg, err = a.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
						}
						if err != nil {
							t.Fatal(err)
						}
						out = append(out, append([]float32(nil), lg...))
						toks = append(toks, argmaxF(lg))
					}
					return out, toks
				}
				ref, teacher := decode(refLast, nil)
				a.Reset()
				newLast, _, err := m.ResidentMRoPEPrefillForTest(ctx, a, ids, feats, start, nImg, mrope)
				if err != nil {
					t.Fatalf("%s: resident m-RoPE prefill: %v", s3Images[ii], err)
				}
				got, _ := decode(append([]float32(nil), newLast...), teacher)
				worst, agree := 1.0, 0
				for k := range ref {
					worst = math.Min(worst, cosF(ref[k], got[k]))
					if argmaxF(ref[k]) == argmaxF(got[k]) {
						agree++
					}
				}
				fmt.Fprintf(os.Stderr, "[S16 G-S16c] %s %s (%d rows, %d image): worst cosine %.7f, argmax %d/%d\n",
					fx.dir, s3Images[ii], n, nImg, worst, agree, len(ref))
				if worst < bar || agree != len(ref) {
					t.Errorf("%s %s: worst cosine %.7f, argmax %d/%d (bar %.4f, every argmax equal)", fx.dir, s3Images[ii], worst, agree, len(ref), bar)
				}
			}
		})
	}
}
