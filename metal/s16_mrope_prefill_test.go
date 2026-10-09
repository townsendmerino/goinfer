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
	defer func(m, d bool) { metalMRoPEPrefillOn, metalDeepstackPrefillOn = m, d }(metalMRoPEPrefillOn, metalDeepstackPrefillOn)
	metalMRoPEPrefillOn, metalDeepstackPrefillOn = true, true // the gates test both paths whatever production keeps on
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

// TestS16MRoPEPrefill_real is G-S16c's first half (docs/tasks/task-multimodal-support-2026-10.md, S16): Qwen2.5-VL-3B and
// Qwen3-VL-2B (with its DeepStack sets, S16 step 2) on the four F2a images, the resident m-RoPE prefill against today's path
// (the CPU prefill and upload, then Metal decode, so both arms decode on the same kernels), over the last prompt row and 8
// teacher-forced decode steps.
//
// The bar, re-registered 2026-10-08 (owner, option a) because Metal's shipped f16 text prefill itself reads 0.977-0.997
// against the CPU's: non-inferiority to a text control. Per image prompt, four text-only prompts of the same length run
// the same isolated comparison (the CPU's batched prefill and upload against Metal's batched pass). PASS when the image
// turn's worst per-step cosine is at least the control's minimum minus 0.005 and every argmax difference is an R10
// near-tie; PARKED 0.005-0.015 below; FAIL otherwise.
//
// Features: Qwen2.5-VL's from the encoder with its Metal tower attached (serve's path), Qwen3-VL's merged features and
// DeepStack sets from the CPU encoder at serve's 1024-row cap; both arms see the same ones. The prompt is the model's own
// chat shape with the image placeholder expanded to the image's merged rows. The hybrid branch is kept for when a hybrid
// is claimed (its only reference decodes on the CPU, so it cannot isolate the prefill; not claimed).
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -timeout 60m -run '^TestS16MRoPEPrefill_real$' -v ./metal/
func TestS16MRoPEPrefill_real(t *testing.T) {
	defer func(m, d bool) { metalMRoPEPrefillOn, metalDeepstackPrefillOn = m, d }(metalMRoPEPrefillOn, metalDeepstackPrefillOn)
	metalMRoPEPrefillOn, metalDeepstackPrefillOn = true, true // the gates test both paths whatever production keeps on
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	const merge, steps = 2, 8
	for _, fx := range []struct {
		dir          string
		hybrid, deep bool
	}{{"qwen25vl-3b-instruct", false, false}, {"qwen3-vl-2b-instruct", false, true}} {
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
			var deepOf func(img) ([]float32, [][]float32, error) // Qwen3-VL: the merged features and the DeepStack sets
			if fx.hybrid || fx.deep {
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
				if fx.deep {
					deepOf = func(i img) ([]float32, [][]float32, error) { return enc.ForwardDeepstack(i.px, [][3]int{i.grid}) }
				}
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
			ctxLen := 8192 // Qwen2.5-VL's formula.png is 6,192 image rows at its default budget
			if fx.hybrid || fx.deep {
				ctxLen = 2048 // the Qwen3.5+ / Qwen3-VL towers are capped at serve's 1,024 merged rows
			}
			// Through the directory's sidecar when it exists (S18: `<dir>.int4.metal.giw`, what serve's Metal load reads):
			// its int4 weights are mapped, not held twice, which is what the 2026-10-08 night's live budget (2.80 GB
			// against 3.77 GB for Qwen2.5-VL) lacked. It generates exactly what the direct load does
			// (prequant.TestDirSidecar_matchesDirectLoad).
			src := dir
			if g := dir + ".int4.metal.giw"; func() bool { _, err := os.Stat(g); return err == nil }() {
				src = g
				fmt.Fprintf(os.Stderr, "[S16] %s: the decoder through its sidecar\n", filepath.Base(fx.dir))
			}
			m, err := decoder.Load(src, decoder.Options{Quant: "int4", Backend: "metal", ResidentContext: ctxLen})
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
				var feats []float32
				var deep [][]float32
				if deepOf != nil {
					feats, deep, err = deepOf(im)
				} else {
					feats, err = featsOf(im)
				}
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
				cache.SetDeepstackForTest(start, nImg, deep)
				refLast, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nImg, mrope, cache)
				cache.SetDeepstackForTest(0, 0, nil)
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
				newLast, _, err := m.ResidentMRoPEDeepstackPrefillForTest(ctx, a, ids, feats, start, nImg, mrope, deep)
				if err != nil {
					t.Fatalf("%s: resident m-RoPE prefill: %v", s3Images[ii], err)
				}
				got, _ := decode(append([]float32(nil), newLast...), teacher)
				worst, ties, real := 1.0, 0, 0
				for k := range ref {
					worst = math.Min(worst, cosF(ref[k], got[k]))
					if ra, ga := argmaxF(ref[k]), argmaxF(got[k]); ra != ga {
						if lp := logSoftmaxF(ref[k]); math.Exp(lp[ga]) >= math.Exp(lp[ra])/2 {
							ties++
						} else {
							real++
						}
					}
				}
				// The re-registered bar (owner, 2026-10-08, option a): the same comparison on four text-only prompts of the same
				// length (the CPU's batched prefill and upload against Metal's batched pass, both decoding on Metal).
				ctlMin := 1.0
				for seed := 1; seed <= 4; seed++ {
					tids := make([]int, n)
					for i := range tids {
						tids[i] = 1000 + (i*7919+seed*104729)%50000
					}
					tc := m.NewCache(n + steps + 1)
					tl, err := m.PrefillLogitsForTest(ctx, tids, tc)
					if err != nil {
						t.Fatal(err)
					}
					a.Reset()
					if err := m.ResidentUploadPrefillForTest(tc); err != nil {
						t.Fatal(err)
					}
					textDecode := func(first []float32, forced []int) ([][]float32, []int) {
						out, toks := [][]float32{append([]float32(nil), first...)}, []int{argmaxF(first)}
						for k := range steps {
							tok := toks[k]
							if forced != nil {
								tok = forced[k]
							}
							lg, err := a.Forward(m.EmbedResidentForTest(tok), n+k)
							if err != nil {
								t.Fatal(err)
							}
							out = append(out, append([]float32(nil), lg...))
							toks = append(toks, argmaxF(lg))
						}
						return out, toks
					}
					cRef, cTeach := textDecode(tl, nil)
					embs := make([][]float32, n)
					for i, id := range tids {
						embs[i] = m.EmbedResidentForTest(id)
					}
					a.Reset()
					first, err := a.batchedPrefill(embs, 0, 0, nil, nil)
					if err != nil {
						t.Fatalf("control's batched pass: %v", err)
					}
					cGot, _ := textDecode(first, cTeach)
					for k := range cRef {
						ctlMin = math.Min(ctlMin, cosF(cRef[k], cGot[k]))
					}
				}
				verdict := "PASS"
				switch {
				case real > 0 || worst < ctlMin-0.015:
					verdict = "FAIL"
				case worst < ctlMin-0.005:
					verdict = "PARKED"
				}
				fmt.Fprintf(os.Stderr, "[S16 G-S16c] %s %s (%d rows, %d image): worst cosine %.7f against the text control's minimum %.7f (4 prompts); argmax: %d near-ties, %d real: %s\n",
					fx.dir, s3Images[ii], n, nImg, worst, ctlMin, ties, real, verdict)
				if verdict != "PASS" {
					t.Errorf("%s %s: %s (worst %.7f, control minimum %.7f, %d non-tie argmax differences)", fx.dir, s3Images[ii], verdict, worst, ctlMin, real)
				}
			}
		})
	}
}

// TestS16DeepstackPrefill_tiny is S16 step 2's G-S16a and G-S16b defects 4-6 (docs/tasks/task-multimodal-support-2026-10.md):
// the tiny Qwen3-VL decoder (2 layers, DeepStack after layers 0 and 1), a synthetic prompt with a 16x24 image (96 merged
// rows) whose features and DeepStack sets are random at unit scale (the comparison is Go path against Go path, so no HF
// golden is needed). Today's path: the CPU prefill with the sets, its K/V uploaded, Metal decode; the new path: Metal's
// DeepStack m-RoPE prefill. G-S16a's bar (the last row and 8 teacher-forced steps, cosine >= 0.9999, every argmax equal).
// Defects, each on the new arm only and each red: (4) the sets not added; (5) added one layer late; (6) added to the text
// rows too (the first image row's set vector on every row before the image).
//
// This fixture's next-token distribution is nearly uniform (its top token near p 0.005 over a ~300-token vocabulary), so an
// argmax difference is read by R10's near-tie rule, as the served gates read one: it counts only when the new token's
// reference probability is under half the reference's top. Cosine carries the bar.
func TestS16DeepstackPrefill_tiny(t *testing.T) {
	defer func(m, d bool) { metalMRoPEPrefillOn, metalDeepstackPrefillOn = m, d }(metalMRoPEPrefillOn, metalDeepstackPrefillOn)
	metalMRoPEPrefillOn, metalDeepstackPrefillOn = true, true // the gates test both paths whatever production keeps on
	const merge, steps, bar, gh, gw = 2, 8, 0.9999, 8, 12
	path := filepath.Join("..", "testdata", "qwen3vl-tiny")
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err != nil {
		t.Fatalf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256,
		Knobs: &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": "0"}})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || a.r.mropeAxis == (Buffer{}) {
		t.Fatalf("not Metal-resident with an m-RoPE axis table: %s", m.ResidentDecline())
	}
	const imgTok, visStart, hid = 299, 298, 64
	nb := gh * gw
	ids := []int{10, 11, 12, visStart}
	start := len(ids)
	for range nb {
		ids = append(ids, imgTok)
	}
	ids = append(ids, 13, 14, 15, 16, 17)
	n := len(ids)
	rng := rand.New(rand.NewSource(17))
	rnd := func(k int) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	feats, sets := rnd(nb*hid), [][]float32{rnd(nb * hid), rnd(nb * hid)}
	mrope, err := decoder.MRopePositionsForTest(ids, imgTok, [][3]int{{1, 2 * gh, 2 * gw}}, merge)
	if err != nil {
		t.Fatal(err)
	}
	delta := mrope[n-1][0] + 1 - n
	ctx := context.Background()
	cache := m.NewCache(n + steps + 1)
	cache.SetDeepstackForTest(start, nb, sets)
	refLast, err := m.PrefillLogitsQwenVLForTest(ctx, ids, feats, start, nb, mrope, cache)
	cache.SetDeepstackForTest(0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	refLast = append([]float32(nil), refLast...)
	a.Reset()
	if err := m.ResidentUploadPrefillForTest(cache); err != nil {
		t.Fatal(err)
	}
	decode := func(first []float32, forced []int) ([][]float32, []int) {
		out, toks := [][]float32{first}, []int{argmaxF(first)}
		for k := range steps {
			tok := toks[k]
			if forced != nil {
				tok = forced[k]
			}
			lg, err := a.ForwardMRoPE(m.EmbedResidentForTest(tok), n+k, n+k+delta)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
			toks = append(toks, argmaxF(lg))
		}
		return out, toks
	}
	ref, teacher := decode(refLast, nil)
	zeros := make([]float32, nb*hid)
	for di, c := range []struct {
		name         string
		sets         [][]float32
		start, count int
	}{
		{"none", sets, start, nb},
		{"(4) the sets not added", nil, start, nb},
		{"(5) added one layer late", [][]float32{zeros, sets[0]}, start, nb},
		{"(6) added to the text rows too", nil, 0, start + nb},
	} {
		s, st, cnt := c.sets, c.start, c.count
		if di == 3 { // the first image row's set vector on every row before the image, then the real sets
			s = make([][]float32, len(sets))
			for k, set := range sets {
				s[k] = make([]float32, 0, cnt*hid)
				for range start {
					s[k] = append(s[k], set[:hid]...)
				}
				s[k] = append(s[k], set...)
			}
		}
		a.Reset()
		var newLast []float32
		if s == nil {
			newLast, _, err = m.ResidentMRoPEPrefillForTest(ctx, a, ids, feats, start, nb, mrope)
		} else {
			newLast, err = a.PrefillMRoPEDeepstackLast(ctx, spliceRows(m, ids, feats, start, nb, hid), 0, mrope, s, st, cnt)
		}
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got, _ := decode(append([]float32(nil), newLast...), teacher)
		worst, agree, real := 1.0, 0, 0
		for k := range ref {
			worst = math.Min(worst, cosF(ref[k], got[k]))
			ra, ga := argmaxF(ref[k]), argmaxF(got[k])
			if ra == ga {
				agree++
				continue
			}
			lp := logSoftmaxF(ref[k])
			tie := math.Exp(lp[ga]) >= math.Exp(lp[ra])/2
			if !tie {
				real++
			}
			fmt.Printf("    step %d: today's top %d p %.4f, new %d (today's p %.4f): near-tie %v\n", k, ra, math.Exp(lp[ra]), ga, math.Exp(lp[ga]), tie)
		}
		pass := worst >= bar && real == 0
		fmt.Printf("[S16 DeepStack] qwen3vl-tiny (16x24 image) %s: worst cosine %.7f, argmax %d/%d, non-tie differences %d\n", c.name, worst, agree, len(ref), real)
		if di == 0 && !pass {
			t.Errorf("the DeepStack prefill misses the bar (worst %.7f, %d non-tie argmax differences)", worst, real)
		}
		if di > 0 && pass {
			t.Errorf("planted defect %s left the bar green (worst %.7f)", c.name, worst)
		}
	}
	// Through the decoder's entry point: a turn with DeepStack sets now takes the resident prefill.
	stream, gen := m.GenerateQwenVLDeepstack(ctx, ids, start, nb, 0xfeed, func() ([]float32, [][]float32, error) { return feats, sets, nil },
		[][3]int{{1, 2 * gh, 2 * gw}}, merge, imgTok, steps, decoder.SamplingParams{})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if e := gen.Err(); e != nil {
		t.Fatal(e)
	}
	fmt.Printf("[S16 DeepStack] through GenerateQwenVLDeepstack: resident prefill %v, tokens %v (today's path %v)\n", gen.ImgPrefillResident, got, teacher[:steps])
	if !gen.ImgPrefillResident {
		t.Error("GenerateQwenVLDeepstack did not take the resident DeepStack prefill")
	}
	for k := range min(len(got), steps) { // the first differing token must be a near-tie in today's logits at that step
		if got[k] != teacher[k] {
			lp := logSoftmaxF(ref[k])
			if math.Exp(lp[got[k]]) < math.Exp(lp[teacher[k]])/2 {
				t.Errorf("the free runs first differ at step %d (%d against %d) and it is not a near-tie (p %.4f against %.4f)",
					k, got[k], teacher[k], math.Exp(lp[got[k]]), math.Exp(lp[teacher[k]]))
			}
			break
		}
	}
}

// spliceRows is residentMRoPEPrefill's row building (the embedding lookup, the raw image features over the image block),
// for a test that calls a resident prefill entry directly.
func spliceRows(m *decoder.Model, ids []int, feats []float32, start, n, hid int) [][]float32 {
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		if i >= start && i < start+n {
			rows[i] = feats[(i-start)*hid : (i-start+1)*hid]
		} else {
			rows[i] = m.EmbedResidentForTest(id)
		}
	}
	return rows
}

func logSoftmaxF(v []float32) []float64 {
	mx := math.Inf(-1)
	for _, x := range v {
		mx = math.Max(mx, float64(x))
	}
	sum := 0.0
	for _, x := range v {
		sum += math.Exp(float64(x) - mx)
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x) - mx - math.Log(sum)
	}
	return out
}
