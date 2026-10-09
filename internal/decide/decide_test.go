package decide

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The rendered bare-v1 noul example from the D0 record (docs/measurements/decisions-d0-prior-art-2026-09-27.md),
// which the record tokenized with JEV-9B's tokenizer: Render must reproduce it byte for byte.
func TestRender_bareV1(t *testing.T) {
	got := Render(KindNoul, "Customer says the parcel arrived damaged and wants their money back.",
		"Is the customer asking for a refund?", noulOptions)
	want := "[kind] noul\n[state] Customer says the parcel arrived damaged and wants their money back.\n[question] Is the customer asking for a refund?\n[options]\nfalse\ntrue\n[decision]:"
	if got != want {
		t.Errorf("noul render:\n%q\nwant\n%q", got, want)
	}
	got = Render(KindChoice, "s", "q", []string{"red", "blue", "green"})
	if want := "[kind] choice\n[state] s\n[question] q\n[options]\nA) red\nB) blue\nC) green\n[decision]:"; got != want {
		t.Errorf("choice render %q, want %q", got, want)
	}
}

func TestValidate(t *testing.T) {
	ok := []Request{
		{Kind: KindNoul, Options: noulOptions},
		{Kind: KindScore, Options: scoreOptions},
		{Kind: KindScore, Options: []string{"0", "1"}}, // two levels (TypeSafe's minimum)
		{Kind: KindScore, Options: ScoreOptions(10)},   // ten, the digit limit
		{Kind: KindChoice, Options: []string{"a", "b"}, Permute: 2},
		{Kind: KindChoice, Options: []string{"a", "b"}, Descriptions: []string{"first", ""}},
	}
	for _, r := range ok {
		if err := Validate(r); err != nil {
			t.Errorf("%+v: %v", r, err)
		}
	}
	bad := []Request{
		{Kind: KindNoul, Options: []string{"true", "false"}},                         // the order is fixed
		{Kind: KindScore, Options: []string{"1", "2"}},                               // levels start at 0
		{Kind: KindScore, Options: []string{"0"}},                                    // one level
		{Kind: KindChoice, Options: []string{"a", "b"}, Descriptions: []string{"x"}}, // misaligned
		{Kind: KindChoice, Options: []string{"a"}},
		{Kind: KindChoice, Options: []string{"a", "a"}},
		{Kind: KindChoice, Options: strings.Split("abcdefghijklmnopq", "")}, // 17
		{Kind: KindNoul, Options: noulOptions, Permute: 2},
		{Kind: "rank", Options: []string{"a", "b"}},
	}
	for _, r := range bad {
		if Validate(r) == nil {
			t.Errorf("%+v validated", r)
		}
	}
}

// fakeTok maps each distinct word to an id and the verbalizers to fixed ids, so the readout's arithmetic can be
// checked without a model.
type fakeTok struct{ ids map[string]int }

func (f fakeTok) EncodePlain(s string) ([]int, error) {
	if id, ok := f.ids[s]; ok {
		return []int{id}, nil
	}
	return []int{1, 2, 3}, nil // any prompt: three tokens
}

func (f fakeTok) EncodeChat(string) ([]int, error) { return []int{4, 5, 6, 7}, nil }

func newFake() fakeTok {
	f := fakeTok{ids: map[string]int{}}
	for i, v := range append(append(append([]string(nil), noulOptions...), strings.Split(digits, "")...), strings.Split(letters, "")...) {
		f.ids[v] = 10 + i
	}
	return f
}

func TestDecide_readout(t *testing.T) {
	tok := newFake()
	logits := make([]float32, 64)
	// choice labels A, B, C are ids 18, 19, 20: logits ln 1, ln 3, ln 4 → p = 1/8, 3/8, 4/8.
	logits[tok.ids["A"]], logits[tok.ids["B"]], logits[tok.ids["C"]] = 0, float32(math.Log(3)), float32(math.Log(4))
	logits[tok.ids["D"]] = 100 // an inactive slot must take no mass
	d, err := New(tok, func(context.Context, []int) ([]float32, error) { return logits, nil }, Options{})
	if err != nil {
		t.Fatal(err)
	}
	r, err := d.Decide(context.Background(), Request{Kind: KindChoice, State: "s", Question: "q", Options: []string{"x", "y", "z"}})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{0.125, 0.375, 0.5}
	for i := range want {
		if math.Abs(r.Distribution[i]-want[i]) > 1e-6 {
			t.Errorf("p[%d] = %v, want %v", i, r.Distribution[i], want[i])
		}
	}
	if r.Decision != "z" || r.Index != 2 || math.Abs(r.Confidence-0.5) > 1e-6 || r.Calibrated || r.Temperature != 1 {
		t.Errorf("decision %q idx %d conf %v calibrated %v T %v", r.Decision, r.Index, r.Confidence, r.Calibrated, r.Temperature)
	}

	// A calibrated temperature divides the logits: T = 2 on ln 3 / ln 4 etc.
	cal := &Calibration{Version: 1, PerKind: map[string]float64{KindChoice: 2}}
	d, _ = New(tok, func(context.Context, []int) ([]float32, error) { return logits, nil }, Options{Calibration: cal})
	r, _ = d.Decide(context.Background(), Request{Kind: KindChoice, State: "s", Question: "q", Options: []string{"x", "y", "z"}})
	z := 1 + math.Sqrt(3) + 2
	if math.Abs(r.Distribution[2]-2/z) > 1e-6 || !r.Calibrated {
		t.Errorf("T=2: p(z) = %v (want %v), calibrated %v", r.Distribution[2], 2/z, r.Calibrated)
	}

	// score: the expected level.
	for i, v := range scoreOptions {
		logits[tok.ids[v]] = -100
		if i == 4 || i == 5 {
			logits[tok.ids[v]] = 0 // p(4) = p(5) = 1/2 → E = 4.5
		}
	}
	d, _ = New(tok, func(context.Context, []int) ([]float32, error) { return logits, nil }, Options{})
	r, _ = d.Decide(context.Background(), Request{Kind: KindScore, State: "s", Question: "q", Options: scoreOptions})
	if r.ExpectedScore == nil || math.Abs(*r.ExpectedScore-4.5) > 1e-6 {
		t.Errorf("expected score %v, want 4.5", r.ExpectedScore)
	}
}

// Permute averages over rotations and maps each back to its option: a model that always prefers the FIRST letter
// shown (pure position bias) must come out uniform when every option sits first once.
func TestDecide_permuteCancelsPositionBias(t *testing.T) {
	tok := newFake()
	logits := make([]float32, 64)
	logits[tok.ids["A"]] = 5 // always the first label, whatever option it shows
	calls := 0
	d, _ := New(tok, func(context.Context, []int) ([]float32, error) { calls++; return logits, nil }, Options{})
	opts := []string{"x", "y", "z", "w"}
	r, err := d.Decide(context.Background(), Request{Kind: KindChoice, State: "s", Question: "q", Options: opts, Permute: 4})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 4 || r.Prefills != 4 {
		t.Errorf("%d prefills, want 4", calls)
	}
	for i, p := range r.Distribution {
		if math.Abs(p-0.25) > 1e-9 {
			t.Errorf("p[%s] = %v, want 0.25 (position bias not cancelled)", opts[i], p)
		}
	}
}

func TestNew_refusesMultiTokenLabels(t *testing.T) {
	tok := newFake()
	delete(tok.ids, "true") // "true" now encodes as three tokens
	if _, err := New(tok, nil, Options{}); err == nil || !strings.Contains(err.Error(), `"true"`) {
		t.Errorf("New with a multi-token label: %v, want an error naming \"true\"", err)
	}
}

// FitTemperatures must recover a known temperature: targets generated as softmax(logp / T*) are fitted by T*.
func TestFitTemperatures_recoversKnownT(t *testing.T) {
	const tStar = 1.7
	var rows []FitRow
	for i := range 200 {
		lp := []float64{float64(i%7) * 0.9, float64(i%5) * -0.6, float64(i%3) * 1.3}
		var z float64
		tgt := make([]float64, len(lp))
		for j, v := range lp {
			tgt[j] = math.Exp(v / tStar)
			z += tgt[j]
		}
		for j := range tgt {
			tgt[j] /= z
		}
		rows = append(rows, FitRow{Kind: KindChoice, LogP: lp, Target: tgt})
	}
	c := FitTemperatures(rows, "test", TemplateBare)
	if got := c.PerKind[KindChoice]; math.Abs(got-tStar) > 1e-3 {
		t.Errorf("fitted T = %v, want %v", got, tStar)
	}
	if f := c.Fit[KindChoice]; !(f.KLAfter < f.KLBefore) || f.KLAfter > 1e-6 || f.N != 200 {
		t.Errorf("fit summary %+v", f)
	}
	path := filepath.Join(t.TempDir(), "calibration.json")
	if err := c.Save(path); err != nil {
		t.Fatal(err)
	}
	back, err := LoadCalibration(path)
	if err != nil {
		t.Fatal(err)
	}
	if tt, ok := back.Temperature(KindChoice); !ok || math.Abs(tt-c.PerKind[KindChoice]) > 1e-12 {
		t.Errorf("round trip T %v ok %v", tt, ok)
	}
}

// LoadCalibration reads autotrust's own file: JEV-9B's calibration.json, verbatim from the D0 record.
func TestLoadCalibration_autotrustFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "calibration.json")
	jev := `{"version": 1, "per_kind": {"noul": 1.0022178236691426, "choice": 0.9840233476014968, "score": 1.012184473350984},
"per_kind_family": {}, "fit": {"noul": {"n": 4363, "T": 1.0022178236691426, "kl_before": 0.004595278762280941, "kl_after": 0.0045948573388159275}},
"diagnostic_calibration_split": {"raw": {"kl": 0.02229316346347332}}, "source": "checkpoints/s2_9b_epoch1/best", "fit_rows": 10954, "d1_excluded": true}`
	if err := os.WriteFile(path, []byte(jev), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadCalibration(path)
	if err != nil {
		t.Fatal(err)
	}
	if tt, ok := c.Temperature(KindChoice); !ok || tt != 0.9840233476014968 {
		t.Errorf("choice T %v ok %v", tt, ok)
	}
	os.WriteFile(path, []byte(`{"version": 2, "per_kind": {}}`), 0o644)
	if _, err := LoadCalibration(path); err == nil {
		t.Error("version 2 loaded")
	}
	// A calibration fitted under chat-v1 must not drive a bare-v1 decider (and vice versa).
	chatCal := &Calibration{Version: 1, PerKind: map[string]float64{KindNoul: 1.3}, Template: TemplateChat}
	if _, err := New(newFake(), nil, Options{Calibration: chatCal}); err == nil {
		t.Error("a chat-v1 calibration was accepted by a bare-v1 decider")
	}
	if _, err := New(newFake(), nil, Options{Template: TemplateChat, Calibration: chatCal}); err != nil {
		t.Errorf("a chat-v1 calibration on a chat-v1 decider: %v", err)
	}
}

func TestRenderChat(t *testing.T) {
	got := RenderChat(KindChoice, "s", "q", []string{"red", "blue"})
	if want := "State:\ns\n\nQuestion: q\n\nOptions:\nA) red\nB) blue\n\nAnswer with exactly one letter: the letter of the best option."; got != want {
		t.Errorf("chat render %q, want %q", got, want)
	}
	if got := RenderChat(KindNoul, "s", "q", noulOptions); !strings.HasSuffix(got, "Answer with exactly one word: true or false.") {
		t.Errorf("noul chat render %q", got)
	}
}

func TestAtSearchBound(t *testing.T) {
	for _, c := range []struct {
		t    float64
		want bool
	}{{50, true}, {49.99989571437015, true}, {1.0, false}, {3.18, false}, {0.02, true}, {0.0200001, true}, {0.05, false}} {
		if got := AtSearchBound(c.t); got != c.want {
			t.Errorf("AtSearchBound(%v) = %v, want %v", c.t, got, c.want)
		}
	}
}

func TestRender_descriptions(t *testing.T) {
	got := RenderChat(KindChoice, "s", "q", []string{"billing", "sales"}, "payments", "")
	if !strings.Contains(got, "A) billing: payments\nB) sales\n") {
		t.Errorf("chat render with descriptions:\n%s", got)
	}
	got = Render(KindScore, "s", "q", ScoreOptions(3), "calm", "upset", "angry")
	if !strings.Contains(got, "[options]\n0: calm\n1: upset\n2: angry\n[decision]:") {
		t.Errorf("bare render of a described 3-level score:\n%s", got)
	}
	if got := RenderChat(KindScore, "s", "q", ScoreOptions(3)); !strings.HasSuffix(got, "one digit from 0 to 2.") {
		t.Errorf("a 3-level score's instruction: %q", got)
	}
}

// recTok records each chat prompt it is asked to encode.
type recTok struct {
	fakeTok
	seen *[]string
}

func (r recTok) EncodeChat(s string) ([]int, error) {
	*r.seen = append(*r.seen, s)
	return []int{4, 5, 6, 7}, nil
}

// A permutation moves each option's description with it: order 2 of 2 shows "A) y: dy" and "B) x: dx".
func TestDecide_permuteCarriesDescriptions(t *testing.T) {
	var seen []string
	tok := recTok{newFake(), &seen}
	d, err := New(tok, func(context.Context, []int) ([]float32, error) { return make([]float32, 64), nil }, Options{Template: TemplateChat})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Decide(context.Background(), Request{Kind: KindChoice, State: "s", Question: "q",
		Options: []string{"x", "y"}, Descriptions: []string{"dx", "dy"}, Permute: 2}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || !strings.Contains(seen[0], "A) x: dx\nB) y: dy") || !strings.Contains(seen[1], "A) y: dy\nB) x: dx") {
		t.Errorf("rendered prompts:\n%v", seen)
	}
}
