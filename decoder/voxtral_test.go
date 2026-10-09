package decoder

import (
	"archive/zip"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14e2 of docs/tasks/task-multimodal-support-2026-10.md (registered 2026-10-09 before any code): Voxtral's text decoder loads from the checkpoint's own layout (nested text_config, tensors
// under language_model.*, an explicit head_dim that is not hidden/heads, rope_theta 1e8, an untied head) and the composition audio samples -> Go front end (one pass over the whole padded
// signal) -> Go tower -> stack four frames -> projector -> soft-token splice -> Go llama decoder matches transformers' VoxtralForConditionalGeneration on the same prompt, on a 6 s clip (one
// 30 s window, 375 audio tokens) and a 35 s clip (two windows, 750). The checkpoint is testdata/voxtral-tiny (scripts/pin_voxtral_tiny.py).

const voxtralTiny = "voxtral-tiny"

func vxGolden(t *testing.T) map[string][]byte {
	t.Helper()
	zr, err := zip.OpenReader(filepath.Join("..", "testdata", voxtralTiny, "golden.zip"))
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	defer zr.Close()
	files := map[string][]byte{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatal(err)
		}
		files[f.Name] = b
	}
	return files
}

// vxSamples turns the golden's 16-bit PCM into the float32 samples the reference used: int16 / 32767, a correctly rounded float64 division and a cast.
func vxSamples(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = float32(float64(int16(binary.LittleEndian.Uint16(b[2*i:]))) / 32767.0)
	}
	return out
}

func loadTinyVoxtral(t *testing.T, dir string) *Model {
	t.Helper()
	m, err := Load(dir, Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if m.w.arch.Name != "voxtral" {
		t.Fatalf("architecture %q, want voxtral", m.w.arch.Name)
	}
	return m
}

func tinyVoxtralDir() string { return filepath.Join("..", "testdata", voxtralTiny) }

func TestVoxtral_tinyTextParity(t *testing.T) {
	files := vxGolden(t)
	m := loadTinyVoxtral(t, tinyVoxtralDir())
	a := m.w.arch
	if a.HeadDim != 16 || a.HiddenDim/a.NumHeads == a.HeadDim {
		t.Fatalf("head dim %d with hidden %d / %d heads: this fixture must have a head_dim that is not hidden/heads", a.HeadDim, a.HiddenDim, a.NumHeads)
	}
	ids := q3aInts(t, files["text.ids.json"])
	want := q3aInts(t, files["text.cont.json"])
	cache := m.NewCache(len(ids) + 9)
	logits, err := m.prefillLogits(context.Background(), ids, cache)
	if err != nil {
		t.Fatal(err)
	}
	ref := q3aF32(files["text.logits.f32"])
	cos := q3aCos(logits, ref)
	got := q3aGreedy(t, m, cache, logits, len(want))
	t.Logf("text path: last-row cosine %.9f, argmax %d (want %d), continuation %v (want %v)", cos, argmax(logits), argmax(ref), got, want)
	if cos < 0.99999 || argmax(logits) != argmax(ref) || !slices.Equal(got, want) {
		t.Errorf("text path differs from transformers: cosine %.9f, continuation %v against %v", cos, got, want)
	}
}

type vxRun struct {
	embeds []float32
	logits []float32
	cont   []int
}

// voxtralCompose runs the whole pipeline for one tiny clip. spliceDefect changes the composition, not the parts (0 = none); it is how the gate shows it can see a wrong splice.
func voxtralCompose(t *testing.T, m *Model, va *audio.VoxtralAudio, files map[string][]byte, clip string, spliceDefect int) vxRun {
	t.Helper()
	wins, err := audio.WhisperFeaturesWindows(vxSamples(files[clip+".pcm16"]), 128)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := va.Embed(wins)
	if err != nil {
		t.Fatal(err)
	}
	ids := q3aInts(t, files[clip+".prompt.json"])
	hidden := m.w.arch.HiddenDim
	n := len(emb) / hidden
	pos := slices.Index(ids, 24) // [AUDIO]
	if pos < 0 {
		t.Fatal("no audio placeholder in the prompt")
	}
	placeholders := 0
	for _, id := range ids {
		if id == 24 {
			placeholders++
		}
	}
	if placeholders != n {
		t.Fatalf("prompt has %d placeholders for %d audio rows", placeholders, n)
	}
	sent := emb
	switch spliceDefect {
	case 1: // the splice one position late
		pos++
	case 2: // one audio row short: the last placeholder keeps its pad embedding
		n--
		sent = emb[:n*hidden]
	case 3: // the audio added to the pad embeddings instead of replacing them
		pad := m.embedN([]int{24})
		sent = slices.Clone(emb)
		for i := range n {
			for j := range hidden {
				sent[i*hidden+j] += pad[j]
			}
		}
	case 4: // the splice keyed on the wrong id: the first [BEGIN_AUDIO] (25) instead of the first [AUDIO] (24)
		pos = slices.Index(ids, 25)
	}
	cache := m.NewCache(len(ids) + 9)
	logits, err := m.prefillLogitsAudio(context.Background(), ids, sent, pos, n, cache)
	if err != nil {
		t.Fatalf("clip %s splice defect %d: %v", clip, spliceDefect, err)
	}
	return vxRun{embeds: emb, logits: slices.Clone(logits), cont: q3aGreedy(t, m, cache, logits, 8)}
}

// rowCosines is the cosine of every row of a [n][h] pair, and the largest absolute difference.
func rowCosines(a, b []float32, h int) (worst float64, maxAbs float64) {
	worst = 1.0
	for r := 0; r < len(a)/h; r++ {
		worst = math.Min(worst, q3aCos(a[r*h:(r+1)*h], b[r*h:(r+1)*h]))
	}
	for i := range a {
		maxAbs = math.Max(maxAbs, math.Abs(float64(a[i]-b[i])))
	}
	return
}

func TestVoxtral_tinyCompositionParity(t *testing.T) {
	files := vxGolden(t)
	m := loadTinyVoxtral(t, tinyVoxtralDir())
	va, err := audio.LoadVoxtralAudio(tinyVoxtralDir())
	if err != nil {
		t.Fatal(err)
	}
	if va.Stack != 4 || va.Rows() != 375 {
		t.Fatalf("stack %d, rows %d per window; want 4 and 375", va.Stack, va.Rows())
	}
	hidden := m.w.arch.HiddenDim
	for _, clip := range []string{"clip6s", "clip35s"} {
		r := voxtralCompose(t, m, va, files, clip, 0)
		refEmb := q3aF32(files[clip+".embeds.f32"])
		if len(refEmb) != len(r.embeds) {
			t.Fatalf("%s: %d embedding values, reference %d", clip, len(r.embeds), len(refEmb))
		}
		worst, maxAbs := rowCosines(r.embeds, refEmb, hidden)
		ref := q3aF32(files[clip+".logits.f32"])
		want := q3aInts(t, files[clip+".cont.json"])
		cos := q3aCos(r.logits, ref)
		t.Logf("%-8s %d audio rows: worst per-row cosine %.9f, max |diff| %.3e; last-position logit cosine %.9f, argmax %d (want %d), continuation %v (want %v)",
			clip, len(r.embeds)/hidden, worst, maxAbs, cos, argmax(r.logits), argmax(ref), r.cont, want)
		if worst < 0.99999 || maxAbs > 4e-3 || cos < 0.99999 || argmax(r.logits) != argmax(ref) || !slices.Equal(r.cont, want) {
			t.Errorf("%s: composition differs from transformers (worst row cosine %.9f, max |diff| %.3e, logit cosine %.9f, continuation %v against %v)", clip, worst, maxAbs, cos, r.cont, want)
		}
	}
}

// variantDir copies the fixture with its config.json rewritten by edit, so a decoder-level defect (a wrong rope_theta, a derived head_dim) can be loaded as a model.
func variantDir(t *testing.T, edit func(map[string]any)) string {
	t.Helper()
	dir := t.TempDir()
	raw, err := os.ReadFile(filepath.Join(tinyVoxtralDir(), "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	edit(cfg)
	out, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), out, 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := os.ReadFile(filepath.Join(tinyVoxtralDir(), "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), w, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestVoxtral_tinyPlantedDefectsAreRed(t *testing.T) {
	files := vxGolden(t)
	m := loadTinyVoxtral(t, tinyVoxtralDir())
	hidden := m.w.arch.HiddenDim
	newVA := func() *audio.VoxtralAudio {
		va, err := audio.LoadVoxtralAudio(tinyVoxtralDir())
		if err != nil {
			t.Fatal(err)
		}
		return va
	}
	// A defect is red when ANY registered criterion sees it: the cosine bar, the max |diff| bar on the projector rows (4e-3: both from the start, G-S14b2's lesson), or the continuation.
	red := func(name string, cosWorst, maxAbs float64, diverged bool) {
		t.Helper()
		t.Logf("planted defect %-62s worst cosine %.6f, max |diff| %.3e, continuation differs: %v", name, cosWorst, maxAbs, diverged)
		if cosWorst >= 0.99999 && maxAbs <= 4e-3 && !diverged {
			t.Errorf("planted defect %q stayed green: cosine %.6f, max |diff| %.3e", name, cosWorst, maxAbs)
		}
	}
	// Defects of the audio path (aikit's seams), over both clips; the 35 s clip is the only one where the window order can show.
	names := []string{"", "the four frames not stacked (only the first of each four)", "the two windows in the wrong order", "no GELU between the projector's linears", "tanh GELU in the projector",
		"a bias of 0.1 after linear_1", "the second linear skipped"}
	for d := 1; d <= audio.VoxtralDefectCountForTest; d++ {
		va := newVA()
		va.SetDefectForTest(d)
		worst, maxAbs, diverged := 1.0, 0.0, false
		for _, clip := range []string{"clip6s", "clip35s"} {
			r := voxtralCompose(t, m, va, files, clip, 0)
			w, ma := rowCosines(r.embeds, q3aF32(files[clip+".embeds.f32"]), hidden)
			worst = math.Min(worst, math.Min(w, q3aCos(r.logits, q3aF32(files[clip+".logits.f32"]))))
			maxAbs = math.Max(maxAbs, ma)
			diverged = diverged || !slices.Equal(r.cont, q3aInts(t, files[clip+".cont.json"]))
		}
		if names[d] == "tanh GELU in the projector" {
			// NOT part of the registered list and NOT red: a BLIND SPOT of this gate, recorded rather than tuned away. The projector rows move by about 2e-3 (5e-4 of their rms), inside the 4e-3 bar
			// registered before the first reading, and the logits and continuation do not move at all. Tightening the bar until it catches a defect I added afterwards would be moving it with
			// the data in hand. What is asserted is only that the defect is real (the rows do differ), so the figure is not a harness no-op.
			t.Logf("BLIND SPOT %-62s worst cosine %.6f, max |diff| %.3e, continuation differs: %v", names[d], worst, maxAbs, diverged)
			if maxAbs == 0 {
				t.Error("the tanh GELU defect changed nothing at all: the seam is not wired")
			}
			continue
		}
		red(names[d], worst, maxAbs, diverged)
	}
	// One defect of the tower itself, through the same composition (the encoder's own gate is G-S14d1).
	va := newVA()
	va.Enc.SetDefectForTest(1) // no position embedding
	r := voxtralCompose(t, m, va, files, "clip6s", 0)
	w, ma := rowCosines(r.embeds, q3aF32(files["clip6s.embeds.f32"]), hidden)
	red("the tower's positions omitted", w, ma, !slices.Equal(r.cont, q3aInts(t, files["clip6s.cont.json"])))
	// Defects of the splice.
	good := newVA()
	for d, name := range map[int]string{1: "the splice one position late", 2: "one audio row short", 3: "the audio added to the pad embeddings", 4: "the splice keyed on [BEGIN_AUDIO]'s id"} {
		worst, diverged := 1.0, false
		for _, clip := range []string{"clip6s", "clip35s"} {
			r := voxtralCompose(t, m, good, files, clip, d)
			worst = math.Min(worst, q3aCos(r.logits, q3aF32(files[clip+".logits.f32"])))
			diverged = diverged || !slices.Equal(r.cont, q3aInts(t, files[clip+".cont.json"]))
		}
		red(name, worst, 0, diverged)
	}
	// Defects of the decoder: rope_theta left at the Llama default; the head dimension derived as hidden/heads (the loader must refuse, because every attention tensor has the real shape).
	dir := variantDir(t, func(c map[string]any) { c["text_config"].(map[string]any)["rope_theta"] = 10000.0 })
	mv := loadTinyVoxtral(t, dir)
	logits, err := mv.prefillLogits(context.Background(), q3aInts(t, files["text.ids.json"]), mv.NewCache(32))
	if err != nil {
		t.Fatal(err)
	}
	ref := q3aF32(files["text.logits.f32"])
	red("rope_theta at the Llama default (1e4)", q3aCos(logits, ref), 0, argmax(logits) != argmax(ref))
	dir = variantDir(t, func(c map[string]any) { delete(c["text_config"].(map[string]any), "head_dim") })
	if mm, err := Load(dir, Options{Backend: "cpu"}); err == nil {
		mm.Close()
		t.Error("the head dimension derived as hidden/heads loaded without complaint: a silent wrong-shape model")
	} else {
		t.Logf("planted defect %-62s refused at load, loudly: %.140v", "the head dimension derived as hidden/heads", err)
	}
}
