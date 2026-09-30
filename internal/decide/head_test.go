package decide

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const judgeTiny = "../../testdata/decisions/judge-tiny"

// fixedTokenizer serves the golden's token ids for the golden's prompt text. The tiny backbone has no tokenizer, so
// the ids are the pin script's; the prompt text is still checked against Render, so the template is covered.
type fixedTokenizer map[string][]int

func (f fixedTokenizer) EncodePlain(s string) ([]int, error) {
	if ids, ok := f[s]; ok {
		return ids, nil
	}
	if len(s) < 200 { // a short state, which the head route counts before rendering: well under the 1024-token cut
		return []int{0}, nil
	}
	return nil, errors.New("fixedTokenizer: prompt not in the golden")
}

func (f fixedTokenizer) EncodeChat(string) ([]int, error) {
	return nil, errors.New("fixedTokenizer: no chat template")
}

type headGolden struct {
	Cases []struct {
		Kind, State, Question, Prompt string
		Options                       []string
		IDs                           []int
		Logits, P                     []float64
		T                             float64
	}
}

// TestHead_matchesReference is D4's gate (docs/tasks/task-constrained-confidence.md): Route B end to end on a tiny
// JEV-shaped judge (testdata/decisions/judge-tiny, from scripts/pin_decisions_head_tiny.py), against the reference
// decide() as the demo Space runs it, with PEFT applying the adapter UNMERGED. goinfer instead merges the adapter at
// load (D3), reads PromptHidden (D2), applies head.safetensors, keeps the kind's first n slots, divides by
// calibration.json's temperature and softmaxes. Five cases: noul twice, score, and choice at 3 and 16 options (the
// masking of unused choice slots). The bar is max |Δp| 1e-5 per case.
func TestHead_matchesReference(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/decisions/head_tiny_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden: run scripts/pin_decisions_head_tiny.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g headGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	h, err := LoadHead(judgeTiny)
	if err != nil {
		t.Fatal(err)
	}
	if h.AdapterDir() == "" || h.Calibration == nil {
		t.Fatalf("judge-tiny should load unmerged with a calibration: adapter %q, calibration %v", h.AdapterDir(), h.Calibration)
	}
	m, err := decoder.Load("../../decoder/testdata/qwen3_5-tiny", decoder.Options{LoRA: h.AdapterDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	tok := fixedTokenizer{}
	for _, c := range g.Cases {
		tok[c.Prompt] = c.IDs
	}
	d, err := New(tok, nil, Options{Head: h, Hidden: ModelHidden(m)})
	if err != nil {
		t.Fatal(err)
	}
	if d.Route() != RouteHead {
		t.Fatalf("route %q", d.Route())
	}
	for i, c := range g.Cases {
		if got := Render(c.Kind, c.State, c.Question, c.Options); got != c.Prompt {
			t.Fatalf("case %d: Render differs from the reference's template:\n got %q\nwant %q", i, got, c.Prompt)
		}
		res, err := d.Decide(context.Background(), Request{Kind: c.Kind, State: c.State, Question: c.Question, Options: c.Options})
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if !res.Calibrated || res.Temperature != c.T || res.Route != RouteHead {
			t.Fatalf("case %d: calibrated %v T %v route %q, want the head's T %v", i, res.Calibrated, res.Temperature, res.Route, c.T)
		}
		var maxd float64
		for j, p := range c.P {
			maxd = math.Max(maxd, math.Abs(res.Distribution[j]-p))
		}
		t.Logf("case %d (%s, %d options, %d tokens): max |Δp| %.2g", i, c.Kind, len(c.Options), len(c.IDs), maxd)
		if maxd > 1e-5 {
			t.Errorf("case %d (%s, %d options): max |Δp| %.3g > 1e-5\n got %v\nwant %v", i, c.Kind, len(c.Options), maxd, res.Distribution, c.P)
		}
	}
}

// TestHead_refuses pins the loader's and the decider's refusals: each is a way to run a head differently from the
// reference while looking fine.
func TestHead_refuses(t *testing.T) {
	cfg, err := os.ReadFile(filepath.Join(judgeTiny, "judge_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	variant := func(t *testing.T, edit func(map[string]any)) string {
		t.Helper()
		var jc map[string]any
		if err := json.Unmarshal(cfg, &jc); err != nil {
			t.Fatal(err)
		}
		edit(jc)
		dir := t.TempDir()
		raw, _ := json.Marshal(jc)
		for _, fn := range []string{"head.safetensors", "calibration.json"} {
			b, err := os.ReadFile(filepath.Join(judgeTiny, fn))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, fn), b, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(dir, "judge_config.json"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	slots := func(jc map[string]any) map[string]any { return jc["slots"].(map[string]any) }
	for _, tc := range []struct {
		name, want string
		edit       func(map[string]any)
	}{
		{"unknown weights_mode", `weights_mode "fused"`, func(jc map[string]any) { jc["weights_mode"] = "fused" }},
		{"unmerged without adapter", "the adapter is missing", func(map[string]any) {}},
		{"unknown template", `template_version "bare-v2"`, func(jc map[string]any) { slots(jc)["template_version"] = "bare-v2" }},
		{"softcap", "softcap 30 is set", func(jc map[string]any) { jc["softcap"] = 30 }},
		{"verbalizers out of order", `slots [8,24) are`, func(jc map[string]any) {
			v := slots(jc)["verbalizers"].([]any)
			v[8], v[9] = v[9], v[8]
		}},
		{"range outside the head", "outside the head's 24", func(jc map[string]any) {
			slots(jc)["ranges"].(map[string]any)["choice"] = []any{8, 25}
		}},
		{"unknown kind", `kind "rank"`, func(jc map[string]any) {
			jc["kinds"] = append(jc["kinds"].([]any), "rank")
			slots(jc)["ranges"].(map[string]any)["rank"] = []any{0, 2}
		}},
		{"hidden size", "proj.weight", func(jc map[string]any) { jc["hidden_size"] = 32 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			edit := tc.edit
			if tc.name != "unmerged without adapter" {
				edit = func(jc map[string]any) { jc["weights_mode"] = "merged"; tc.edit(jc) }
			}
			_, err := LoadHead(variant(t, edit))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}

	h, err := LoadHead(judgeTiny)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := New(fixedTokenizer{}, nil, Options{Head: h}); err == nil || !strings.Contains(err.Error(), "hidden state") {
		t.Fatalf("a head without Hidden: %v", err)
	}
	hidden := func(context.Context, []int) ([]float32, error) { return make([]float32, h.Hidden), nil }
	if _, err := New(fixedTokenizer{}, nil, Options{Head: h, Hidden: hidden, Template: TemplateChat}); err == nil ||
		!strings.Contains(err.Error(), "trained under template") {
		t.Fatalf("a head under chat-v1: %v", err)
	}
	d, err := New(fixedTokenizer{}, nil, Options{Head: h, Hidden: hidden})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, want string
		r          Request
	}{
		{"score of 4 levels", "exactly 6 levels", Request{Kind: KindScore, State: "s", Question: "q", Options: ScoreOptions(4)}},
		{"descriptions", "descriptions", Request{Kind: KindChoice, State: "s", Question: "q", Options: []string{"a", "b"}, Descriptions: []string{"x", "y"}}},
	} {
		if _, err := d.Decide(context.Background(), tc.r); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want an error containing %q", tc.name, err, tc.want)
		}
	}
}
