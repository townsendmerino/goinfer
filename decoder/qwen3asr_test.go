package decoder

import (
	"archive/zip"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14c1 and G-S14c2 of docs/tasks/task-multimodal-support-2026-10.md: Qwen3-ASR's decoder loads from the checkpoint's own layout (nested thinker_config,
// tensors under thinker.*), its text path matches transformers, and the composition audio samples -> Go front end -> Go encoder and projector -> soft-token splice -> Go decoder matches
// transformers' Qwen3ASRForConditionalGeneration on the same prompt. The checkpoint is testdata/qwen3asr-tiny (scripts/pin_qwen3asr_tiny.py).

func q3aGolden(t *testing.T) map[string][]byte {
	t.Helper()
	zr, err := zip.OpenReader(filepath.Join("..", "testdata", "qwen3asr-tiny", "golden.zip"))
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

func q3aF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func q3aInts(t *testing.T, b []byte) []int {
	var v []int
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func q3aCos(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func q3aGreedy(t *testing.T, m *Model, cache *KVCache, first []float32, n int) []int {
	t.Helper()
	var out []int
	l := slices.Clone(first)
	for range n {
		id := argmax(l)
		out = append(out, id)
		var err error
		if l, err = m.forward(id, cache); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func loadTinyASR(t *testing.T) *Model {
	t.Helper()
	m, err := Load(filepath.Join("..", "testdata", "qwen3asr-tiny"), Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if m.w.arch.Name != "qwen3_asr" {
		t.Fatalf("architecture %q, want qwen3_asr", m.w.arch.Name)
	}
	return m
}

func TestQwen3ASR_tinyTextParity(t *testing.T) {
	files := q3aGolden(t)
	m := loadTinyASR(t)
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

// qwen3ASRCompose runs the whole pipeline for one tiny clip and returns the last-position logits and the greedy continuation. A defect (0 = none) changes the composition, not the
// parts: it is how the gate shows it can see a wrong splice.
func qwen3ASRCompose(t *testing.T, m *Model, enc *audio.QwenASREncoder, files map[string][]byte, clip string, defect int) (logits []float32, cont []int) {
	t.Helper()
	samples := q3aF32(files[clip+".samples.f32"])
	f, err := audio.QwenASRFeatures(samples)
	if err != nil {
		t.Fatal(err)
	}
	emb, n, err := enc.Forward(f.Data, f.T, f.Valid)
	if err != nil {
		t.Fatal(err)
	}
	ids := q3aInts(t, files[clip+".prompt.json"])
	pos := slices.Index(ids, 150)
	if pos < 0 {
		t.Fatal("no audio placeholder in the prompt")
	}
	hidden := m.w.arch.HiddenDim
	ctx := context.Background()
	cache := m.NewCache(len(ids) + 9)
	switch defect {
	case 1: // the splice one position early
		pos--
	case 2: // one audio row short: the last placeholder keeps its pad embedding
		n--
		emb = emb[:n*hidden]
	case 3: // the audio added to the pad embeddings instead of replacing them
		pad := m.embedN([]int{150})
		for i := range n {
			for j := range hidden {
				emb[i*hidden+j] += pad[j]
			}
		}
	}
	if defect == 4 { // the wrong prefill reused for audio: Gemma 3's, whose image block is bidirectional
		logits, err = m.prefillLogitsVL(ctx, ids, emb, pos, n, cache)
	} else {
		logits, err = m.prefillLogitsAudio(ctx, ids, emb, pos, n, cache)
	}
	if err != nil {
		t.Fatalf("clip %s defect %d: %v", clip, defect, err)
	}
	return slices.Clone(logits), q3aGreedy(t, m, cache, logits, 8)
}

func TestQwen3ASR_tinyCompositionParity(t *testing.T) {
	files := q3aGolden(t)
	m := loadTinyASR(t)
	enc, err := audio.LoadQwenASREncoder(filepath.Join("..", "testdata", "qwen3asr-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	for _, clip := range []string{"clip2s", "clip5s3", "clip7s"} {
		logits, cont := qwen3ASRCompose(t, m, enc, files, clip, 0)
		ref := q3aF32(files[clip+".logits.f32"])
		want := q3aInts(t, files[clip+".cont.json"])
		cos := q3aCos(logits, ref)
		t.Logf("%-8s composition: last-position cosine %.9f, argmax %d (want %d), continuation %v (want %v)", clip, cos, argmax(logits), argmax(ref), cont, want)
		if cos < 0.99999 || argmax(logits) != argmax(ref) || !slices.Equal(cont, want) {
			t.Errorf("%s: composition differs from transformers (cosine %.9f, continuation %v against %v)", clip, cos, cont, want)
		}
	}
}

func TestQwen3ASR_tinyCompositionPlantedDefectsAreRed(t *testing.T) {
	files := q3aGolden(t)
	m := loadTinyASR(t)
	enc, err := audio.LoadQwenASREncoder(filepath.Join("..", "testdata", "qwen3asr-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"", "splice one position early", "one audio row short", "added to the pad embeddings", "bidirectional block (Gemma 3's prefill)"}
	for d := 1; d <= 4; d++ {
		worst := 1.0
		diverged := false
		for _, clip := range []string{"clip2s", "clip5s3", "clip7s"} {
			logits, cont := qwen3ASRCompose(t, m, enc, files, clip, d)
			worst = math.Min(worst, q3aCos(logits, q3aF32(files[clip+".logits.f32"])))
			diverged = diverged || !slices.Equal(cont, q3aInts(t, files[clip+".cont.json"]))
		}
		t.Logf("planted defect %d (%s): worst last-position cosine %.6f, continuation differs: %v", d, names[d], worst, diverged)
		if worst >= 0.99999 && !diverged {
			t.Errorf("planted defect %d (%s) stayed green: cosine %.6f", d, names[d], worst)
		}
	}
	// A fifth: the projector's GELU removed (the encoder's own seam), through the same composition.
	enc.SetDefectForTest(6)
	defer enc.SetDefectForTest(0)
	logits, _ := qwen3ASRCompose(t, m, enc, files, "clip5s3", 0)
	cos := q3aCos(logits, q3aF32(files["clip5s3.logits.f32"]))
	t.Logf("planted defect 5 (projector GELU removed): last-position cosine %.6f", cos)
	if cos >= 0.99999 {
		t.Errorf("planted defect 5 stayed green: cosine %.6f", cos)
	}
}
