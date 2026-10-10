package whisper

import (
	"slices"
	"testing"
)

// The generation rules' mechanics on the tiny decoder (G-S14f; the rules against transformers are G-S14f2's, on the real checkpoint): the prompt, the language choice, suppress_tokens at every position,
// begin_suppress_tokens at the first only, the stop token, and the cap.
func tinyGen() GenConfig {
	return GenConfig{DecoderStart: 3, EOS: 2, NoTimestamps: 7, MaxLength: 24, LangToID: map[string]int{"<|en|>": 150, "<|fr|>": 151, "<|de|>": 152}, TaskToID: map[string]int{"transcribe": 153, "translate": 154}}
}

func TestGenerate_rules(t *testing.T) {
	_, _, cases := tinyCases(t)
	d := loadTiny(t, defectNone)
	enc := cases[0].enc
	g := tinyGen()
	base, err := d.Generate(enc, g, "en", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(base.Prompt, []int{3, 150, 153, 7}) || base.Language != "<|en|>" {
		t.Fatalf("prompt %v, language %s", base.Prompt, base.Language)
	}
	// max_length is raised by the prompt's length (at most max_target_positions/2 - 1 of it) and never past max_target_positions: 24 + 4 here
	if cap := d.maxLength(g, len(base.Prompt)); cap != 28 || len(base.Prompt)+len(base.IDs) > cap {
		t.Fatalf("%d tokens past max_length %d (want 28)", len(base.Prompt)+len(base.IDs), cap)
	}
	// suppress_tokens: a token the model wrote is never written once suppressed, at any position
	g2 := tinyGen()
	g2.Suppress = []int{base.IDs[0]}
	res, err := d.Generate(enc, g2, "en", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(res.IDs, base.IDs[0]) {
		t.Errorf("suppressed token %d written: %v", base.IDs[0], res.IDs)
	}
	// begin_suppress_tokens: only the first position
	g3 := tinyGen()
	g3.BeginSuppress = []int{base.IDs[0]}
	res3, err := d.Generate(enc, g3, "en", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	if res3.IDs[0] == base.IDs[0] {
		t.Errorf("begin-suppressed token %d written first", base.IDs[0])
	}
	// the stop token ends the text
	g4 := tinyGen()
	g4.EOS = base.IDs[1]
	g4.MaxLength = 24
	res4, err := d.Generate(enc, g4, "en", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	if !res4.Stopped || res4.IDs[len(res4.IDs)-1] != g4.EOS || len(res4.IDs) != 2 {
		t.Errorf("stop token %d: ids %v stopped %v", g4.EOS, res4.IDs, res4.Stopped)
	}
	// a language the checkpoint does not know, a task it does not know, and the detection
	if _, err := d.Generate(enc, g, "xx", "transcribe"); err == nil {
		t.Error("an unknown language was accepted")
	}
	if _, err := d.Generate(enc, g, "en", "dance"); err == nil {
		t.Error("an unknown task was accepted")
	}
	tok, id, err := d.Detect(enc, g)
	if err != nil || g.LangToID[tok] != id {
		t.Errorf("Detect: %q %d %v", tok, id, err)
	}
	if det, err := d.Generate(enc, g, "", ""); err != nil || det.Prompt[1] != id {
		t.Errorf("detected-language prompt %v (err %v), Detect said %d", det.Prompt, err, id)
	}
	// the length cap
	g5 := tinyGen()
	g5.MaxLength = 6
	res5, err := d.Generate(enc, g5, "en", "transcribe")
	if err != nil || len(res5.IDs) > 6 || len(res5.IDs) < 6 && !res5.Stopped {
		t.Errorf("max_length 6 (raised to 10 by the 4-token prompt): %d generated ids, stopped %v (err %v)", len(res5.IDs), res5.Stopped, err)
	}
	if d.maxLength(GenConfig{MaxLength: 60}, 4) != 64 || d.maxLength(GenConfig{}, 4) != 64 {
		t.Errorf("max_length is never past max_target_positions (64): got %d and %d", d.maxLength(GenConfig{MaxLength: 60}, 4), d.maxLength(GenConfig{}, 4))
	}
}
