//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestGemma3ResidentReal_imageBlockAtomicity is P9(a)'s pre-registered kill condition (docs/multimodal.md's P9(a) plan,
// "Top risk: the atomicity boundary"), run against the REAL GenerateVL entrypoint on gemma-3-4b-it's resident CUDA backend,
// not the decoder package's fakeResident unit gates (decoder/generate_vl_resident_test.go's
// TestGenerateVL_imageReuseFastPath_* cover the control flow in isolation; this proves the same claim against real
// kernels). CUDA is the witness because gemma-3-4b-it's text decoder declines residency on WebGPU entirely.
//
// Same token-IDENTITY methodology as gpu/resident_reuse_parity_test.go's TestResidentPrefixReuse_tokenIdentical, for the
// same reason: a wrong image-block match produces fluent, confidently wrong output with no error anywhere, so only
// bit-for-bit identity against a reuse-disabled cold run is acceptable evidence. Two claims, both checked, across a 3-turn
// transcript (image A, the SAME image A resent, then a DIFFERENT image B at the same placeholder slot):
//
//   - Resending the SAME image must reuse the resident KV (the vision tower's own call counter stays flat across that turn,
//     and Generation.PrefillReused reports a reuse length reaching at least past the image block) AND the warm run's output
//     must be bitwise identical to a cold (reuse-forced-off) run's.
//   - A DIFFERENT image at the SAME slot must NOT reuse (the tower is called again, PrefillReused reports 0) AND the output
//     is bitwise identical to a cold run: a bug that claims reuse it should not would still show up here even if it
//     produced plausible-looking text.
//
// The vision tower's forward pass is real but MEMOIZED by image identity: a real forward pass on this checkpoint is
// minutes-scale on CPU (demo/agent/agent.go's TurnImage doc comment) and deterministic, and reuse-vs-not is a property of
// GenerateVL and the resident KV, not of the tower. The call counter still increments on EVERY invocation (that is what
// proves whether GenerateVL called the closure); only the compute behind it is paid once per distinct image (2 real
// forward passes total, not up to 6).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestGemma3ResidentReal_imageBlockAtomicity -v -timeout 30m
func TestGemma3ResidentReal_imageBlockAtomicity(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GEMMA3_4B")
	if ckpt == "" {
		ckpt = home + "/models/gemma-3-4b-it"
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", ckpt, err)
	}
	const golden = "../testdata/gemma3_real_golden.json.gz"
	var g struct {
		InputIDs        []int     `json:"input_ids"`
		ImageTokenStart int       `json:"image_token_start"`
		MMTokens        int       `json:"mm_tokens_per_image"`
		PixelValues     []float32 `json:"pixel_values"`
	}
	if err := decoder.ReadGoldenJSONForTest(golden, &g); err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_gemma3_real.py", err)
	}

	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}

	enc, err := vision.LoadEncoder(ckpt, false) // f32, matches the pin script's own reference
	if err != nil {
		t.Fatalf("LoadEncoder: %v", err)
	}
	proj, err := multimodal.LoadProjector(ckpt)
	if err != nil {
		t.Fatalf("LoadProjector: %v", err)
	}

	// Image B: same shape as the golden's real pixel_values, deliberately different content (a sign flip guarantees a
	// different SigLIP forward pass); all that matters is that it is NOT image A's content.
	pixelsB := make([]float32, len(g.PixelValues))
	for i, v := range g.PixelValues {
		pixelsB[i] = -v
	}

	towerCalls := 0
	towerCache := map[string][]float32{}
	metered := func(id string, pixels []float32) func() ([]float32, error) {
		return func() ([]float32, error) {
			towerCalls++
			if f, ok := towerCache[id]; ok {
				return f, nil
			}
			hidden, err := enc.Forward(pixels)
			if err != nil {
				return nil, err
			}
			feats, err := proj.Forward(hidden)
			if err != nil {
				return nil, err
			}
			towerCache[id] = feats
			return feats, nil
		}
	}

	greedy := decoder.SamplingParams{Temperature: 0}
	const imgHashA, imgHashB = 42, 99
	const maxNew = 6

	// run drives the 3-turn image transcript (cold image A, resend the SAME image A, then a
	// DIFFERENT image B at the same slot) and returns each turn's emitted ids, PrefillReused,
	// and the running tower-call count observed immediately after that turn.
	run := func(t *testing.T, reuse bool) (outs [][]int, reused []int, towerAfter []int) {
		if reuse {
			os.Unsetenv("GOINFER_NO_RESIDENT_REUSE")
		} else {
			os.Setenv("GOINFER_NO_RESIDENT_REUSE", "1")
		}
		t.Cleanup(func() { os.Unsetenv("GOINFER_NO_RESIDENT_REUSE") })

		m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		if !m.ResidentActive() {
			t.Skip("gemma-3-4b-it did not go resident on cuda — this gate needs the resident path")
		}

		var prevIDs, prevGen []int
		for turn := range 3 {
			var prompt []int
			var id string
			var pixels []float32
			var hash uint64
			switch turn {
			case 0: // cold: image A
				prompt, id, pixels, hash = g.InputIDs, "A", g.PixelValues, imgHashA
			case 1: // the SAME image A, resent as a strict extension (the agent-loop shape)
				next := append(append([]int(nil), prevIDs...), prevGen...)
				prompt = append(next, prevIDs[len(prevIDs)-1])
				id, pixels, hash = "A", g.PixelValues, imgHashA
			case 2: // a DIFFERENT image B at the SAME placeholder slot — the adversarial case.
				// The placeholder token ids are identical to turn 1's either way (that is
				// exactly why the hash exists at all), so g.InputIDs is reused verbatim.
				prompt, id, pixels, hash = g.InputIDs, "B", pixelsB, imgHashB
			}
			ch, gen := m.GenerateVL(context.Background(), prompt, g.ImageTokenStart, g.MMTokens, hash,
				metered(id, pixels), maxNew, greedy)
			var got []int
			for tok := range ch {
				got = append(got, tok)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("turn %d: %v", turn+1, err)
			}
			outs = append(outs, got)
			reused = append(reused, gen.PrefillReused)
			towerAfter = append(towerAfter, towerCalls)
			prevIDs, prevGen = prompt, got
		}
		return outs, reused, towerAfter
	}

	warm, warmReused, warmTower := run(t, true)
	cold, coldReused, _ := run(t, false)

	t.Logf("PrefillReused per turn: warm %v · cold %v", warmReused, coldReused)
	t.Logf("tower calls after each turn (warm): %v", warmTower)

	if want := g.ImageTokenStart + g.MMTokens; warmReused[1] < want {
		t.Fatalf("warm turn 2 (same image resent) PrefillReused = %d, want >= %d — reuse never fired, "+
			"so the bitwise-identity check below would prove nothing", warmReused[1], want)
	}
	if warmTower[1] != warmTower[0] {
		t.Errorf("warm turn 2: tower called again (count %d -> %d) — the same resent image must skip it entirely",
			warmTower[0], warmTower[1])
	}
	if warmReused[2] != 0 {
		t.Errorf("warm turn 3 (different image, same slot) PrefillReused = %d, want 0 — the block boundary "+
			"must be respected, not short-circuited", warmReused[2])
	}
	if warmTower[2] != warmTower[1]+1 {
		t.Errorf("warm turn 3: tower call count %d -> %d, want +1 — a different image must always run it",
			warmTower[1], warmTower[2])
	}
	for i, n := range coldReused {
		if n != 0 {
			t.Errorf("cold run turn %d PrefillReused = %d, want 0 — GOINFER_NO_RESIDENT_REUSE must disable "+
				"image reuse entirely too", i+1, n)
		}
	}

	for i := range warm {
		if len(warm[i]) != len(cold[i]) {
			t.Fatalf("turn %d: warm emitted %d tokens, cold %d — reuse changed the output length",
				i+1, len(warm[i]), len(cold[i]))
		}
		for j := range warm[i] {
			if warm[i][j] != cold[i][j] {
				t.Fatalf("turn %d diverges at token %d: warm %d vs cold %d — reuse changed the output",
					i+1, j, warm[i][j], cold[i][j])
			}
		}
	}
}
