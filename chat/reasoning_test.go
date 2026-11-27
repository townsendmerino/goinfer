package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

type thinkGoldenCase struct {
	Conv   string `json:"conv"`
	Mode   string `json:"mode"`
	Tools  bool   `json:"tools"`
	Prompt string `json:"prompt"`
	IDs    []int  `json:"ids"`
}
type thinkGolden struct {
	Checkpoint   string            `json:"checkpoint"`
	ChatTemplate string            `json:"chat_template"`
	Cases        []thinkGoldenCase `json:"cases"`
}

func loadThinkGoldens(t *testing.T) []thinkGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "think_modes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []thinkGolden
	if err := json.Unmarshal(raw, &gs); err != nil {
		t.Fatal(err)
	}
	if len(gs) != 4 {
		t.Fatalf("expected the four pinned checkpoints, got %d", len(gs))
	}
	return gs
}

func convTurns(conv string) (string, []Turn) {
	switch conv {
	case "user":
		return "", []Turn{{Role: "user", Content: "Hi"}}
	case "sys_user":
		return "Be brief.", []Turn{{Role: "user", Content: "Hi"}}
	case "multi":
		return "", []Turn{{Role: "user", Content: "Hi"}, {Role: "assistant", Content: "Hello!"}, {Role: "user", Content: "And?"}}
	}
	panic(conv)
}

var goldenMode = map[string]ThinkMode{"unset": ThinkTemplate, "false": ThinkOff, "true": ThinkOn}

var goldenTools = []Tool{{Name: "get_weather", Description: "Weather for a city",
	Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}}

// TestThinkModes_matchHF pins the PROMPT half, per checkpoint: Detect runs on the real chat template text, and every
// thinking setting renders byte-for-byte what HuggingFace renders from that same template. One checkpoint proving nothing
// about another is the lesson here — the 0.8B and 9B differ only in which way the enable_thinking test points.
func TestThinkModes_matchHF(t *testing.T) {
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Fatalf("%s: Detect: %v", g.Checkpoint, err)
		}
		if tmpl.Reasoning() == nil {
			t.Fatalf("%s: thinking control not recognised from its real template", g.Checkpoint)
		}
		for _, c := range g.Cases {
			if c.Tools {
				continue // below
			}
			system, turns := convTurns(c.Conv)
			got := tmpl.WithThinking(goldenMode[c.Mode]).Render(system, turns)
			ran++
			if got != c.Prompt {
				t.Errorf("%s %s enable_thinking=%s:\n got %q\nwant %q", g.Checkpoint, c.Conv, c.Mode, got, c.Prompt)
			}
		}
	}
	if ran != 4*9 {
		t.Fatalf("ran %d text cases, want 36", ran)
	}
}

// TestThinkModes_toolsMatchHF: Gemma 4's tool prompt, byte-exact in every thinking setting. (Qwen's own tool syntax is
// XML, goinfer's is Hermes JSON — a known, separate gap — so for Qwen only the generation-prompt ending is checked, below.)
func TestThinkModes_toolsMatchHF(t *testing.T) {
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		if !strings.HasPrefix(g.Checkpoint, "gemma-4") {
			continue
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for _, c := range g.Cases {
			if !c.Tools {
				continue
			}
			system, turns := convTurns(c.Conv)
			got := tmpl.WithThinking(goldenMode[c.Mode]).RenderTools(system, turns, goldenTools)
			ran++
			if got != c.Prompt {
				t.Errorf("%s tools %s enable_thinking=%s:\n got %q\nwant %q", g.Checkpoint, c.Conv, c.Mode, got, c.Prompt)
			}
		}
	}
	if ran != 6 {
		t.Fatalf("ran %d tool cases, want 6", ran)
	}
}

// Every Qwen tool prompt must end with the same think suffix the no-tools prompt does in that mode.
func TestThinkModes_chatmlToolsEndLikeText(t *testing.T) {
	for _, g := range loadThinkGoldens(t) {
		if strings.HasPrefix(g.Checkpoint, "gemma-4") {
			continue
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for mode, m := range goldenMode {
			text := tmpl.WithThinking(m).Render("", []Turn{{Role: "user", Content: "Hi"}})
			tools := tmpl.WithThinking(m).RenderTools("", []Turn{{Role: "user", Content: "Hi"}}, goldenTools)
			tail := text[strings.LastIndex(text, "<|im_start|>assistant\n"):]
			if !strings.HasSuffix(tools, tail) {
				t.Errorf("%s enable_thinking=%s: tool prompt ends %q, text prompt ends %q", g.Checkpoint, mode, tools[len(tools)-40:], tail)
			}
		}
	}
}

// ThinkAsIs is today's bytes: it must equal the generic rendering for every checkpoint, and be a strict prefix of
// every thinking rendering of a ChatML-family template (a suffix is only ever appended).
func TestThinkModes_asIsIsTodaysBytes(t *testing.T) {
	for _, g := range loadThinkGoldens(t) {
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		var generic *Template
		if tmpl.Name() == "gemma4" {
			generic = Gemma4()
		} else {
			generic = ChatML()
		}
		for _, conv := range []string{"user", "sys_user", "multi"} {
			system, turns := convTurns(conv)
			asis := tmpl.Render(system, turns)
			if want := generic.Render(system, turns); asis != want {
				t.Errorf("%s %s: as-is %q != generic %q", g.Checkpoint, conv, asis, want)
			}
			if tmpl.Name() != "chatml" {
				continue
			}
			for _, m := range []ThinkMode{ThinkTemplate, ThinkOn, ThinkOff} {
				if got := tmpl.WithThinking(m).Render(system, turns); !strings.HasPrefix(got, asis) {
					t.Errorf("%s %s mode %s: %q does not extend the as-is prompt %q", g.Checkpoint, conv, m, got, asis)
				}
			}
		}
	}
}

// The defaults and the open/closed shape per checkpoint, stated as facts about the real templates.
func TestThinkModes_detectedFacts(t *testing.T) {
	want := map[string]struct {
		defaultOn bool
		opensOn   bool // PromptOpensThink under ThinkTemplate
	}{
		"qwen3-4b":           {true, false},
		"qwen3.5-0.8b":       {false, false},
		"qwen3.5-9b":         {true, true},
		"gemma-4-26b-a4b-it": {false, false},
	}
	for _, g := range loadThinkGoldens(t) {
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		w := want[g.Checkpoint]
		if got := tmpl.Reasoning().DefaultOn(); got != w.defaultOn {
			t.Errorf("%s: DefaultOn=%v, want %v", g.Checkpoint, got, w.defaultOn)
		}
		if got := tmpl.WithThinking(ThinkTemplate).PromptOpensThink(); got != w.opensOn {
			t.Errorf("%s: PromptOpensThink(template)=%v, want %v", g.Checkpoint, got, w.opensOn)
		}
		if tmpl.WithThinking(ThinkOff).PromptOpensThink() {
			t.Errorf("%s: a thinking-off prompt must never end inside an open block", g.Checkpoint)
		}
	}
}

// The fail-toward-today's-bytes rule, from both directions: templates that do not think get no spec, and a ChatML
// template whose control is not one of the three shapes read from real checkpoints is left unmanaged rather than guessed.
func TestThinkModes_unrecognisedIsUnmanaged(t *testing.T) {
	plain := "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
	weird := "<|im_start|>{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is none %}{{ '<think>\\n' }}{% endif %}{% endif %}"
	for name, tmplText := range map[string]string{"qwen2.5-like": plain, "unrecognised control": weird} {
		tmpl, err := Detect(Meta{ChatTemplate: tmplText})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if tmpl.Reasoning() != nil {
			t.Errorf("%s: got a reasoning spec, want nil", name)
		}
		user := []Turn{{Role: "user", Content: "Hi"}}
		base := tmpl.Render("", user)
		for _, m := range []ThinkMode{ThinkTemplate, ThinkOn, ThinkOff} {
			if got := tmpl.WithThinking(m).Render("", user); got != base {
				t.Errorf("%s mode %s changed the bytes: %q vs %q", name, m, got, base)
			}
		}
		if tmpl.NewReasoningSplitter(nil) != nil {
			t.Errorf("%s: an unmanaged template must not split", name)
		}
	}
}

// TestThinkModes_idsMatchHF pins that the tagged segments tokenize to the ids HuggingFace produces — the
// <think> / </think> / <|think|> / channel markers must come out as their control tokens. Needs the real tokenizer files;
// skipped (visibly) without them, so run with -v and read for PASS.
func TestThinkModes_idsMatchHF(t *testing.T) {
	home, _ := os.UserHomeDir()
	ran := 0
	for _, g := range loadThinkGoldens(t) {
		dir := filepath.Join(home, "models", g.Checkpoint)
		if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
			t.Logf("SKIP %s: no tokenizer at %s", g.Checkpoint, dir)
			continue
		}
		tk, err := tokenizer.Load(dir)
		if err != nil {
			t.Fatalf("%s: %v", g.Checkpoint, err)
		}
		tmpl, _ := Detect(Meta{ChatTemplate: g.ChatTemplate})
		for _, c := range g.Cases {
			system, turns := convTurns(c.Conv)
			tm := tmpl.WithThinking(goldenMode[c.Mode])
			segs := tm.RenderSegments(system, turns)
			if c.Tools {
				segs = tm.RenderToolsSegments(system, turns, goldenTools)
			}
			ids, err := tk.EncodeSegments(segs, false)
			if err != nil {
				t.Fatal(err)
			}
			ran++
			if len(ids) != len(c.IDs) {
				t.Errorf("%s %s/%s tools=%v: %d ids, HF %d", g.Checkpoint, c.Conv, c.Mode, c.Tools, len(ids), len(c.IDs))
				continue
			}
			for i := range ids {
				if ids[i] != c.IDs[i] {
					t.Errorf("%s %s/%s tools=%v: id[%d]=%d, HF %d", g.Checkpoint, c.Conv, c.Mode, c.Tools, i, ids[i], c.IDs[i])
					break
				}
			}
		}
	}
	t.Logf("%d prompts tokenized and compared against HF ids", ran)
}

// TestThinkModes_delimitersDecodeToTheirSurfaceForm: the splitter works on DECODED text, so the real tokenizer must decode
// the delimiter tokens to their literal surface form (serve's streamTokens appends DecodePiece(id) per token). Checked
// against each checkpoint's real tokenizer, because the serve-level splitter tests use a synthetic vocabulary and would pass
// even if a real tokenizer dropped special tokens on decode — the one thing that would make the whole split a no-op.
// Skipped, visibly, without the tokenizer files; run with -v.
func TestThinkModes_delimitersDecodeToTheirSurfaceForm(t *testing.T) {
	home, _ := os.UserHomeDir()
	want := map[string][]string{
		"qwen3-4b":           {"<think>", "</think>"},
		"qwen3.5-0.8b":       {"<think>", "</think>"},
		"qwen3.5-9b":         {"<think>", "</think>"},
		"gemma-4-26b-a4b-it": {"<|channel>", "<channel|>"},
	}
	ran := 0
	for name, toks := range want {
		dir := filepath.Join(home, "models", name)
		if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
			t.Logf("SKIP %s: no tokenizer at %s", name, dir)
			continue
		}
		tk, err := tokenizer.Load(dir)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, lit := range toks {
			id, ok := tk.TokenID(lit)
			if !ok {
				t.Errorf("%s: %q is not a single token", name, lit)
				continue
			}
			piece, _ := tk.DecodePiece(id)
			ran++
			if piece != lit {
				t.Errorf("%s: DecodePiece(%d) = %q, want the literal %q — the splitter would never see the delimiter", name, id, piece, lit)
			}
		}
	}
	t.Logf("%d delimiter tokens decoded and compared", ran)
}
