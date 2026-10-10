package whisper

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14g2 of docs/tasks/task-multimodal-support-2026-10.md: timestamps and sequential long-form against transformers' generate(return_timestamps=True, return_segments=True) on a tiny random-weight Whisper
// whose vocabulary has the real layout in miniature (testdata/whisper-tiny-ts; scripts/pin_whisper_ts_tiny.py). Random weights make every branch of the timestamp rules fire.

func tsDir() string { return filepath.Join("..", "..", "testdata", "whisper-tiny-ts") }

// tsClip rebuilds scripts/pin_whisper_ts_tiny.py's clip: every step is exact in float64, so the float32 samples are the reference's.
func tsClip(n, a, b int) []float32 {
	out := make([]float32, n)
	for i := range n {
		saw1 := float64((i*a)%1000)/1000 - 0.5
		saw2 := float64((i*b)%1733)/1733 - 0.5
		env := float64((i/4000)%3) / 2.0
		noise := float64((uint64(i)*2654435761)&0xFFFFFFFF)/4294967296.0 - 0.5
		out[i] = float32(0.25*saw1 + 0.2*saw2*env + 0.1*noise*float64(1+(i/9000)%2))
	}
	return out
}

type tsGolden struct {
	Cases map[string]struct {
		LanguageID int `json:"language_id"`
		Samples    int
		A, B       int
		Segments   []struct {
			Start, End float64
			Tokens     []int
		}
		Sequence []int
	}
}

func tinyTS(t *testing.T) (*Transcriber, tsGolden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(tsDir(), "golden.json"))
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g tsGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadWhisperEncoder(tsDir())
	if err != nil {
		t.Fatal(err)
	}
	dec, err := Load(tsDir())
	if err != nil {
		t.Fatal(err)
	}
	gen, err := LoadGenConfig(tsDir())
	if err != nil {
		t.Fatal(err)
	}
	return &Transcriber{Mels: enc.Cfg.MelBins, Enc: enc, Dec: dec, Gen: gen}, g
}

// diff names how a result differs from the reference's, or "" when it does not.
func diffTS(r *LongResult, want struct {
	LanguageID int `json:"language_id"`
	Samples    int
	A, B       int
	Segments   []struct {
		Start, End float64
		Tokens     []int
	}
	Sequence []int
}) string {
	if len(r.Segments) != len(want.Segments) {
		return "segment count"
	}
	for i, s := range r.Segments {
		w := want.Segments[i]
		if math.Abs(s.Start-w.Start) > 1e-9 || math.Abs(s.End-w.End) > 1e-9 || !slices.Equal(s.Tokens, w.Tokens) {
			return "segment content"
		}
	}
	if !slices.Equal(r.Tokens, want.Sequence) {
		return "sequence"
	}
	return ""
}

func TestTimestamps_tinyMatchesTransformers(t *testing.T) {
	tr, g := tinyTS(t)
	for name, c := range g.Cases {
		r, err := tr.TranscribeTimestamps(tsClip(c.Samples, c.A, c.B), "", "")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		t.Logf("%s: %d segments, %d tokens, language %s; first %v..%v, last end %.2f", name, len(r.Segments), len(r.Tokens), r.Language, r.Segments[0].Start, r.Segments[0].End, r.Segments[len(r.Segments)-1].End)
		if d := diffTS(r, c); d != "" {
			t.Errorf("%s: differs from transformers: %s\n got %+v\nwant %+v", name, d, r.Segments, c.Segments)
		}
	}
}

// TestTimestamps_plantedDefectsAreRed: every planted defect must change the long-form result somewhere, or the gate could not see that mistake.
func TestTimestamps_plantedDefectsAreRed(t *testing.T) {
	tr, g := tinyTS(t)
	for name, defect := range map[string]int{"pair rule ignores the penultimate": tsNoPairRule, "timestamps may decrease": tsNoMonotonic, "no first-position restriction": tsNoInitialRule,
		"logprob rule off": tsNoLogprobRule, "window always advances 30 s": tsAdvanceWindow, "single ending treated as a pair": tsSingleAsPair, "no time offset": tsNoTimeOffset} {
		tr.Dec.tsDefect = defect
		red := false
		for _, c := range g.Cases {
			r, err := tr.TranscribeTimestamps(tsClip(c.Samples, c.A, c.B), "", "")
			if err != nil || diffTS(r, c) != "" {
				red = true
			}
		}
		t.Logf("%-34s red %v", name, red)
		if !red {
			t.Errorf("defect %q left the result equal to transformers': the gate cannot see it", name)
		}
	}
	tr.Dec.tsDefect = tsNone
}

// TestProcessLogits_order: suppress_tokens must run BEFORE the timestamp rule. A suppressed token that would be the best text token must not count in the "timestamps outweigh any text token" comparison;
// with the order reversed the comparison is made against it and the timestamp is not forced. (Not seen on the random clips: a suppressed id is rarely the best text id, 56 clips searched.)
func TestProcessLogits_order(t *testing.T) {
	g := GenConfig{EOS: 8, NoTimestamps: 10, Suppress: []int{3}}
	logits := func() []float32 { // ids 0-9 text, 11-19 timestamps; the suppressed token 3 leads the text, the timestamps' summed mass sits between it and the next text token
		l := make([]float32, 20)
		for i := range l {
			l[i] = -5
		}
		l[3], l[4] = 3.0, 1.0
		for i := 11; i < 20; i++ {
			l[i] = 0.5
		}
		return l
	}
	argmax := func(l []float32) int {
		b := 0
		for i, v := range l {
			if v > l[b] {
				b = i
			}
		}
		return b
	}
	d := &Decoder{}
	l := logits()
	d.processLogits(l, []int{2}, g, true) // one text token of history, so the first-position rule is out of the way
	if got := argmax(l); got < 11 {
		t.Errorf("suppress first: the next token is %d, want a timestamp (the suppressed 3 must not count as text)", got)
	}
	d.tsDefect = tsSuppressAfter
	l = logits()
	d.processLogits(l, []int{2}, g, true)
	if got := argmax(l); got != 4 {
		t.Errorf("suppress last (the defect): the next token is %d, want text token 4 (the comparison was made against the suppressed 3)", got)
	}
}
